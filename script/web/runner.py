"""Run smart_research.py as a subprocess and stream output to SSE listeners.

Two layers of streaming:

  1. Per-job stream — each Job has its own listener queue. Items are dicts:
       {"event": "line",    "text": "..."}
       {"event": "pause",   "results": [...]}
       {"event": "resumed"}
       {"event": "cascade", "topic": "...", "children": [...], ...}
       None                                     # sentinel: stream ended

  2. Queue stream — JOB_QUEUE.listeners receive queue-level events:
       {"event": "job-enqueued",            "item": {...}}
       {"event": "job-started",             "item": {...}}
       {"event": "job-finished",            "queue_id": "...", "job_id": "...", "exit_code": 0}
       {"event": "job-cancelled-pending",   "queue_id": "..."}
       {"event": "queue-cleared",           "n": 3}
       {"event": "concurrency-changed",     "n": 2}
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple


_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # the `script/` folder

_PAUSE_OPEN = "[pause:pick-sources]"
_PAUSE_CLOSE = "[/pause:pick-sources]"
_CASCADE_OPEN = "[cascade-context]"
_CASCADE_CLOSE = "[/cascade-context]"


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    query: str
    args: Dict
    lines: List[str] = field(default_factory=list)
    listeners: List["queue.Queue"] = field(default_factory=list)
    done: bool = False
    exit_code: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    process: Optional[subprocess.Popen] = None
    paused: bool = False
    pause_payload: Optional[Dict[str, Any]] = None
    cascade_context: Optional[Dict[str, Any]] = None
    on_finish: Optional[Callable[["Job"], None]] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    def _broadcast(self, item: Optional[Dict[str, Any]]) -> None:
        # Caller must hold _lock.
        for q in self.listeners:
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def append_line(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            self._broadcast({"event": "line", "text": line})

    def emit_pause(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.paused = True
            self.pause_payload = payload
            self._broadcast({"event": "pause", **payload})

    def emit_resumed(self) -> None:
        with self._lock:
            self.paused = False
            self.pause_payload = None
            self._broadcast({"event": "resumed"})

    def emit_cascade(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.cascade_context = payload
            self._broadcast({"event": "cascade", **payload})

    def finish(self, code: int) -> None:
        with self._lock:
            self.done = True
            self.exit_code = code
            self.finished_at = time.time()
            self.paused = False
            self._broadcast(None)
        cb = self.on_finish
        if cb:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                # Don't let a queue callback poison the worker thread.
                pass

    def attach_listener(self) -> "queue.Queue":
        with self._lock:
            q: queue.Queue = queue.Queue(maxsize=1024)
            self.listeners.append(q)
            return q

    def detach_listener(self, q: "queue.Queue") -> None:
        with self._lock:
            if q in self.listeners:
                self.listeners.remove(q)

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "id": self.id,
                "query": self.query,
                "args": self.args,
                "done": self.done,
                "exit_code": self.exit_code,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "line_count": len(self.lines),
                "paused": self.paused,
                "pause_payload": self.pause_payload,
                "cascade_context": self.cascade_context,
            }

    def resume(self, urls: List[str]) -> bool:
        """Send a curated URL list back to the paused subprocess."""
        with self._lock:
            if not self.paused or self.done or self.process is None:
                return False
            stdin = self.process.stdin
            if stdin is None or stdin.closed:
                return False
        try:
            stdin.write(json.dumps({"urls": list(urls)}) + "\n")
            stdin.flush()
        except (OSError, ValueError):
            return False
        self.emit_resumed()
        return True


# Global registry. Single-user local app; no eviction policy.
JOBS: Dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def list_jobs(limit: int = 50) -> List[Dict]:
    with _JOBS_LOCK:
        snaps = [j.snapshot() for j in JOBS.values()]
    snaps.sort(key=lambda j: j["started_at"], reverse=True)
    return snaps[:limit]


def get_job(job_id: str) -> Optional[Job]:
    return JOBS.get(job_id)


# ---------------------------------------------------------------------------
# Subprocess spawn
# ---------------------------------------------------------------------------

def _build_cmd(query: str, kwargs: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, "-u", "smart_research.py", query]
    if kwargs.get("provider"):
        cmd += ["--provider", kwargs["provider"]]
    if kwargs.get("extract_provider"):
        cmd += ["--extract-provider", kwargs["extract_provider"]]
    if kwargs.get("dry_run"):
        cmd += ["--dry-run"]
    if kwargs.get("parent"):
        cmd += ["--parent", kwargs["parent"]]
    if kwargs.get("no_parent"):
        cmd += ["--no-parent"]
    if kwargs.get("max_stubs") is not None:
        cmd += ["--max-stubs", str(kwargs["max_stubs"])]
    if kwargs.get("max_results") is not None:
        cmd += ["--max-results", str(kwargs["max_results"])]
    if kwargs.get("top_sources") is not None:
        cmd += ["--top-sources", str(kwargs["top_sources"])]
    if kwargs.get("pause_pick_sources"):
        cmd += ["--pause-pick-sources"]
    return cmd


def _spawn_now(
    query: str,
    *,
    kwargs: Dict[str, Any],
    on_finish: Optional[Callable[["Job"], None]] = None,
) -> Job:
    """Spawn smart_research.py immediately. Returns the Job; subprocess runs in
    a daemon thread."""
    job = Job(id=uuid.uuid4().hex, query=query, args=dict(kwargs), on_finish=on_finish)
    with _JOBS_LOCK:
        JOBS[job.id] = job

    cmd = _build_cmd(query, kwargs)
    job.append_line(f"$ {' '.join(cmd)}")

    def worker() -> None:
        try:
            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(_SCRIPT_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            job.process = proc
            assert proc.stdout is not None

            # Three-state line consumer: normal | pause | cascade.
            mode = "normal"
            buf: List[str] = []
            for raw in proc.stdout:
                line = raw.rstrip("\n")

                if mode == "normal":
                    if line == _PAUSE_OPEN:
                        mode = "pause"; buf = []
                        continue
                    if line == _CASCADE_OPEN:
                        mode = "cascade"; buf = []
                        continue
                    job.append_line(line)
                    continue

                close_marker = _PAUSE_CLOSE if mode == "pause" else _CASCADE_CLOSE
                if line == close_marker:
                    joined = "\n".join(buf).strip()
                    try:
                        payload = json.loads(joined) if joined else {}
                    except json.JSONDecodeError:
                        # Surface the raw block so it isn't silently lost.
                        open_marker = _PAUSE_OPEN if mode == "pause" else _CASCADE_OPEN
                        job.append_line(open_marker)
                        for b in buf:
                            job.append_line(b)
                        job.append_line(close_marker)
                        job.append_line(
                            f"[runner] could not parse {mode} payload as JSON"
                        )
                    else:
                        if mode == "pause":
                            job.emit_pause(payload)
                        else:
                            job.emit_cascade(payload)
                    mode = "normal"; buf = []
                    continue
                buf.append(line)

            code = proc.wait()
        except Exception as e:  # noqa: BLE001
            job.append_line(f"[runner error] {type(e).__name__}: {e}")
            code = -1
        job.finish(code)

    t = threading.Thread(target=worker, daemon=True, name=f"job-{job.id[:8]}")
    t.start()
    return job


def cancel_job(job_id: str) -> bool:
    job = JOBS.get(job_id)
    if not job or job.done or job.process is None:
        return False
    try:
        job.process.terminate()
        job.append_line("[runner] cancellation requested")
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(r"[^a-z0-9]+")


def _norm_query_key(query: str) -> str:
    """Lightweight filename-ish normalization for dedup keys. Doesn't need to
    match stub_writer.sanitize_filename perfectly — only needs to be stable."""
    return _FILENAME_RE.sub("-", query.strip().lower()).strip("-")


@dataclass
class QueueItem:
    queue_id: str
    query: str
    kwargs: Dict[str, Any]
    enqueued_at: float
    job_id: Optional[str] = None
    cascade_depth_remaining: int = 0
    cascade_picker: bool = True
    parent_label: Optional[str] = None  # "auto-cascade from <X>" or "mass" etc.

    def to_summary(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "query": self.query,
            "kwargs": self.kwargs,
            "enqueued_at": self.enqueued_at,
            "job_id": self.job_id,
            "cascade_depth_remaining": self.cascade_depth_remaining,
            "cascade_picker": self.cascade_picker,
            "parent_label": self.parent_label,
        }


class JobQueue:
    def __init__(self) -> None:
        self.pending: Deque[QueueItem] = deque()
        self.running: Dict[str, QueueItem] = {}     # job_id → item
        self.recent_done: List[QueueItem] = []
        self.concurrency: int = 1
        self.seen_keys: Set[Tuple[str, str]] = set()
        self.listeners: List["queue.Queue"] = []
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        t = threading.Thread(
            target=self._dispatcher_loop, daemon=True, name="queue-dispatcher",
        )
        t.start()

    # ------------------------------------------------------------------
    # Listener fanout (queue-level events)
    # ------------------------------------------------------------------

    def attach_listener(self) -> "queue.Queue":
        with self._lock:
            q: queue.Queue = queue.Queue(maxsize=512)
            self.listeners.append(q)
            return q

    def detach_listener(self, q: "queue.Queue") -> None:
        with self._lock:
            if q in self.listeners:
                self.listeners.remove(q)

    def _broadcast(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            ls = list(self.listeners)
        for q in ls:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # Public mutation
    # ------------------------------------------------------------------

    def _key_for(self, query: str, parent: Optional[str]) -> Tuple[str, str]:
        return (_norm_query_key(query), _norm_query_key(parent or ""))

    def enqueue_one(
        self,
        query: str,
        kwargs: Dict[str, Any],
        *,
        cascade_depth_remaining: int = 0,
        cascade_picker: bool = True,
        parent_label: Optional[str] = None,
    ) -> Optional[str]:
        """Returns queue_id if enqueued, None if duplicate-skipped."""
        query = query.strip()
        if not query:
            return None
        key = self._key_for(query, kwargs.get("parent"))
        item = QueueItem(
            queue_id=uuid.uuid4().hex,
            query=query,
            kwargs=dict(kwargs),
            enqueued_at=time.time(),
            cascade_depth_remaining=max(0, min(3, int(cascade_depth_remaining))),
            cascade_picker=cascade_picker,
            parent_label=parent_label,
        )
        with self._lock:
            if key in self.seen_keys:
                return None
            self.seen_keys.add(key)
            self.pending.append(item)
        self._broadcast({"event": "job-enqueued", "item": item.to_summary()})
        self._wakeup.set()
        return item.queue_id

    def enqueue_many(
        self,
        queries: List[Dict[str, Any]],
        shared_kwargs: Dict[str, Any],
        *,
        cascade_depth_remaining: int = 0,
        cascade_picker: bool = True,
        parent_label: Optional[str] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Bulk enqueue. Each entry is {query: str, parent?: str}.
        Returns (enqueued_queue_ids, skipped)."""
        enqueued: List[str] = []
        skipped: List[Dict[str, Any]] = []
        for q in queries:
            query = (q.get("query") or "").strip()
            if not query:
                skipped.append({**q, "reason": "empty query"})
                continue
            kwargs = dict(shared_kwargs)
            if q.get("parent"):
                kwargs["parent"] = q["parent"]
            qid = self.enqueue_one(
                query, kwargs,
                cascade_depth_remaining=cascade_depth_remaining,
                cascade_picker=cascade_picker,
                parent_label=parent_label,
            )
            if qid:
                enqueued.append(qid)
            else:
                skipped.append({**q, "reason": "duplicate"})
        return enqueued, skipped

    def cancel_pending(self, queue_id: str) -> bool:
        with self._lock:
            for i, item in enumerate(self.pending):
                if item.queue_id == queue_id:
                    del self.pending[i]
                    self.seen_keys.discard(
                        self._key_for(item.query, item.kwargs.get("parent"))
                    )
                    found = item
                    break
            else:
                return False
        self._broadcast({"event": "job-cancelled-pending", "queue_id": queue_id})
        return True

    def clear_pending(self) -> int:
        with self._lock:
            n = len(self.pending)
            for item in self.pending:
                self.seen_keys.discard(
                    self._key_for(item.query, item.kwargs.get("parent"))
                )
            self.pending.clear()
        self._broadcast({"event": "queue-cleared", "n": n})
        return n

    def set_concurrency(self, n: int) -> int:
        n = max(1, min(4, int(n)))
        with self._lock:
            self.concurrency = n
        self._broadcast({"event": "concurrency-changed", "n": n})
        self._wakeup.set()
        return n

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    def state_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "concurrency": self.concurrency,
                "pending": [it.to_summary() for it in self.pending],
                "running": [it.to_summary() for it in self.running.values()],
                "recent_done": [it.to_summary() for it in self.recent_done[-20:]],
            }

    def wait_for_job_id(self, queue_id: str, timeout: float = 0.5) -> Optional[str]:
        """Block (lightly) until the dispatcher assigns a job_id to this queue
        item. Returns None on timeout (queue is full)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for item in self.running.values():
                    if item.queue_id == queue_id:
                        return item.job_id
                # If the item is no longer pending or running, it was cancelled
                # or completed already.
            time.sleep(0.02)
        return None

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatcher_loop(self) -> None:
        while True:
            self._wakeup.wait()
            self._wakeup.clear()
            self._spawn_until_full()

    def _spawn_until_full(self) -> None:
        while True:
            with self._lock:
                if not self.pending or len(self.running) >= self.concurrency:
                    return
                item = self.pending.popleft()
            # Spawn outside the lock — Popen is slow-ish.
            job = _spawn_now(
                item.query, kwargs=item.kwargs, on_finish=self._on_job_finish,
            )
            item.job_id = job.id
            with self._lock:
                self.running[job.id] = item
            self._broadcast({"event": "job-started", "item": item.to_summary()})

    def _on_job_finish(self, job: Job) -> None:
        with self._lock:
            item = self.running.pop(job.id, None)
            if item:
                self.recent_done.append(item)
        if item:
            self._broadcast({
                "event": "job-finished",
                "queue_id": item.queue_id,
                "job_id": job.id,
                "exit_code": job.exit_code,
            })
            # Auto-cascade: enqueue children with one less depth, no pause/picker.
            if (
                job.exit_code == 0
                and item.cascade_depth_remaining > 0
                and isinstance(job.cascade_context, dict)
            ):
                children = job.cascade_context.get("children") or []
                base = dict(item.kwargs)
                base["parent"] = job.query
                base["pause_pick_sources"] = False
                for child in children:
                    if isinstance(child, str) and child.strip():
                        self.enqueue_one(
                            child,
                            base,
                            cascade_depth_remaining=item.cascade_depth_remaining - 1,
                            cascade_picker=False,
                            parent_label=f"auto-cascade from {item.query}",
                        )
        self._wakeup.set()


# Singleton queue (the dispatcher thread starts on import).
JOB_QUEUE = JobQueue()


# ---------------------------------------------------------------------------
# Public API used by web/app.py
# ---------------------------------------------------------------------------

def enqueue_research(
    query: str,
    *,
    provider: Optional[str] = None,
    extract_provider: Optional[str] = None,
    dry_run: bool = False,
    parent: Optional[str] = None,
    no_parent: bool = False,
    max_stubs: Optional[int] = None,
    max_results: Optional[int] = None,
    top_sources: Optional[int] = None,
    pause_pick_sources: bool = False,
    cascade_depth: int = 0,
    cascade_picker: bool = True,
    parent_label: Optional[str] = None,
) -> Optional[str]:
    """Enqueue a single research job. Returns queue_id (None if duplicate)."""
    kwargs = {
        "provider": provider,
        "extract_provider": extract_provider,
        "dry_run": dry_run,
        "parent": parent,
        "no_parent": no_parent,
        "max_stubs": max_stubs,
        "max_results": max_results,
        "top_sources": top_sources,
        "pause_pick_sources": pause_pick_sources,
    }
    return JOB_QUEUE.enqueue_one(
        query, kwargs,
        cascade_depth_remaining=cascade_depth,
        cascade_picker=cascade_picker,
        parent_label=parent_label,
    )
