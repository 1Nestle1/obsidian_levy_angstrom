"""FastAPI app for the local research-pipeline UI.

Run with:
    python -m web                # uses default host/port
    python -m web --port 9000    # custom port

Or from inside the script/ folder:
    uvicorn web.app:app --reload --port 8765
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow `python web/app.py` from the script/ folder by inserting the parent
# on sys.path (so `import config` and `from src...` resolve).
_HERE = Path(__file__).resolve().parent
_SCRIPT_DIR = _HERE.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import json
import queue as _queue

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config as _cfg
from src.modules.llm_factory import describe_provider, list_providers, resolve_provider
from src.modules.llm_probe import (
    ProbeResult,
    load_last_provider,
    probe_all,
    probe_provider,
    save_last_provider,
)
from src.modules.openai_compat_client import LLMError
from web.runner import (
    JOB_QUEUE,
    cancel_job,
    enqueue_research,
    get_job,
    list_jobs,
)


app = FastAPI(title="Obsidian Research Pipeline", version="0.1.0")

_STATIC_DIR = _HERE / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ProbePayload(BaseModel):
    name: str
    status: str
    ready: bool
    reachable: bool
    auth_ok: bool
    available_models: List[str]
    extract_model: str
    synthesize_model: str
    extract_available: bool
    synthesize_available: bool
    latency_ms: int
    error: str

    @classmethod
    def from_result(cls, r: ProbeResult) -> "ProbePayload":
        return cls(
            name=r.name,
            status=r.status,
            ready=r.ready,
            reachable=r.reachable,
            auth_ok=r.auth_ok,
            available_models=r.available_models,
            extract_model=r.extract_model,
            synthesize_model=r.synthesize_model,
            extract_available=r.extract_available,
            synthesize_available=r.synthesize_available,
            latency_ms=r.latency_ms,
            error=r.error,
        )


class SelectProviderRequest(BaseModel):
    name: str


class StateResponse(BaseModel):
    selected_provider: Optional[str]
    extract_provider: Optional[str]
    vault_path: str
    description: str


class ResearchRequest(BaseModel):
    query: str
    provider: Optional[str] = None
    extract_provider: Optional[str] = None
    dry_run: bool = False
    parent: Optional[str] = None
    no_parent: bool = False
    max_stubs: Optional[int] = None
    max_results: Optional[int] = None
    top_sources: Optional[int] = None
    pause_pick_sources: bool = True  # gate Pass-1 LLM costs by default
    cascade_depth: int = 0
    cascade_picker: bool = True


class JobStartResponse(BaseModel):
    queue_id: str
    job_id: Optional[str] = None


class ResumeRequest(BaseModel):
    urls: List[str]


class MassQueueQuery(BaseModel):
    query: str
    parent: Optional[str] = None


class MassQueueShared(BaseModel):
    provider: Optional[str] = None
    extract_provider: Optional[str] = None
    dry_run: bool = False
    no_parent: bool = False
    max_stubs: Optional[int] = None
    max_results: Optional[int] = None
    top_sources: Optional[int] = None
    pause_pick_sources: bool = False  # mass mode auto-picks all sources
    cascade_depth: int = 0
    cascade_picker: bool = False


class MassQueueRequest(BaseModel):
    queries: List[MassQueueQuery]
    shared: MassQueueShared = MassQueueShared()


class ConcurrencyRequest(BaseModel):
    n: int


class CascadeSubmitRequest(BaseModel):
    children: List[str] = []
    siblings: List[str] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/state", response_model=StateResponse)
def get_state() -> StateResponse:
    cached = load_last_provider()
    return StateResponse(
        selected_provider=cached,
        extract_provider=None,
        vault_path=_cfg.VAULT_PATH,
        description=describe_provider(cached) if cached else describe_provider(),
    )


@app.get("/api/providers", response_model=List[ProbePayload])
def get_providers() -> List[ProbePayload]:
    """Probe every configured provider in parallel and return the status table."""
    results = probe_all(timeout=8.0)
    return [ProbePayload.from_result(r) for r in results]


@app.get("/api/providers/{name}", response_model=ProbePayload)
def get_provider(name: str) -> ProbePayload:
    if name not in list_providers():
        raise HTTPException(status_code=404, detail=f"Unknown provider: {name}")
    return ProbePayload.from_result(probe_provider(name, timeout=10.0))


@app.post("/api/select-provider", response_model=StateResponse)
def select_provider(body: SelectProviderRequest) -> StateResponse:
    if body.name not in list_providers():
        raise HTTPException(status_code=400, detail=f"Unknown provider: {body.name}")
    try:
        # Validate it's at least configurable.
        resolve_provider(body.name)
    except LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    save_last_provider(body.name)
    return get_state()


@app.get("/api/healthz")
def healthz() -> dict:
    return {"ok": True}


# ---------------------------------------------------------------------------
# Research jobs
# ---------------------------------------------------------------------------

def _validate_provider(name: Optional[str], label: str = "provider") -> None:
    if name and name not in list_providers():
        raise HTTPException(status_code=400, detail=f"Unknown {label}: {name}")


@app.post("/api/research", response_model=JobStartResponse)
def start_research(body: ResearchRequest) -> JobStartResponse:
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    provider = body.provider or load_last_provider()
    _validate_provider(provider)
    _validate_provider(body.extract_provider, label="extract provider")

    queue_id = enqueue_research(
        query,
        provider=provider,
        extract_provider=body.extract_provider,
        dry_run=body.dry_run,
        parent=body.parent,
        no_parent=body.no_parent,
        max_stubs=body.max_stubs,
        max_results=body.max_results,
        top_sources=body.top_sources,
        pause_pick_sources=body.pause_pick_sources,
        cascade_depth=body.cascade_depth,
        cascade_picker=body.cascade_picker,
    )
    if not queue_id:
        raise HTTPException(status_code=409, detail="duplicate (already in queue)")
    job_id = JOB_QUEUE.wait_for_job_id(queue_id, timeout=0.5)
    return JobStartResponse(queue_id=queue_id, job_id=job_id)


@app.post("/api/research/queue")
def queue_many(body: MassQueueRequest) -> JSONResponse:
    if not body.queries:
        raise HTTPException(status_code=400, detail="queries list is empty")

    shared = body.shared
    provider = shared.provider or load_last_provider()
    _validate_provider(provider)
    _validate_provider(shared.extract_provider, label="extract provider")

    shared_kwargs = {
        "provider": provider,
        "extract_provider": shared.extract_provider,
        "dry_run": shared.dry_run,
        "no_parent": shared.no_parent,
        "max_stubs": shared.max_stubs,
        "max_results": shared.max_results,
        "top_sources": shared.top_sources,
        "pause_pick_sources": shared.pause_pick_sources,
    }
    queries = [{"query": q.query, "parent": q.parent} for q in body.queries]
    enqueued, skipped = JOB_QUEUE.enqueue_many(
        queries,
        shared_kwargs,
        cascade_depth_remaining=shared.cascade_depth,
        cascade_picker=shared.cascade_picker,
        parent_label="mass",
    )
    return JSONResponse({"queue_ids": enqueued, "skipped": skipped})


@app.get("/api/queue")
def queue_state() -> JSONResponse:
    return JSONResponse(JOB_QUEUE.state_summary())


@app.post("/api/queue/clear")
def queue_clear() -> JSONResponse:
    n = JOB_QUEUE.clear_pending()
    return JSONResponse({"ok": True, "cleared": n})


@app.post("/api/queue/concurrency")
def queue_concurrency(body: ConcurrencyRequest) -> JSONResponse:
    n = JOB_QUEUE.set_concurrency(body.n)
    return JSONResponse({"concurrency": n})


@app.delete("/api/queue/{queue_id}")
def queue_cancel_pending(queue_id: str) -> JSONResponse:
    if not JOB_QUEUE.cancel_pending(queue_id):
        raise HTTPException(status_code=404, detail="not pending (already started or unknown)")
    return JSONResponse({"ok": True})


@app.get("/api/queue/stream")
def queue_stream() -> StreamingResponse:
    def _sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    def gen():
        listener = JOB_QUEUE.attach_listener()
        try:
            # Replay current state once on connect so a fresh tab doesn't need
            # to issue a separate GET.
            yield _sse("snapshot", JOB_QUEUE.state_summary())
            while True:
                try:
                    item = listener.get(timeout=15.0)
                except _queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                ev = item.get("event") if isinstance(item, dict) else None
                if not ev:
                    continue
                payload = {k: v for k, v in item.items() if k != "event"}
                yield _sse(ev, payload)
        finally:
            JOB_QUEUE.detach_listener(listener)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/research/{job_id}/cascade")
def cascade_submit(job_id: str, body: CascadeSubmitRequest) -> JSONResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if not job.cascade_context:
        raise HTTPException(status_code=400, detail="job has no cascade context")

    parent_topic = job.cascade_context.get("topic_display") or job.query
    domain_parent = job.cascade_context.get("parent")
    base_kwargs = dict(job.args)
    base_kwargs["pause_pick_sources"] = False  # cascaded jobs shouldn't gate the queue

    enqueued: List[str] = []
    skipped: List[Dict] = []

    # Children: become topics under THIS job's topic.
    children_kwargs = dict(base_kwargs)
    children_kwargs["parent"] = parent_topic
    children_kwargs["no_parent"] = False
    for name in body.children:
        if not name.strip():
            continue
        qid = JOB_QUEUE.enqueue_one(
            name, children_kwargs,
            cascade_depth_remaining=0, cascade_picker=False,
            parent_label=f"child of {parent_topic}",
        )
        if qid:
            enqueued.append(qid)
        else:
            skipped.append({"query": name, "reason": "duplicate"})

    # Siblings: peers under the SAME parent as this topic.
    sibling_kwargs = dict(base_kwargs)
    if domain_parent:
        sibling_kwargs["parent"] = domain_parent
        sibling_kwargs["no_parent"] = False
    else:
        sibling_kwargs.pop("parent", None)
        sibling_kwargs["no_parent"] = True
    for name in body.siblings:
        if not name.strip():
            continue
        qid = JOB_QUEUE.enqueue_one(
            name, sibling_kwargs,
            cascade_depth_remaining=0, cascade_picker=False,
            parent_label=f"sibling of {parent_topic}",
        )
        if qid:
            enqueued.append(qid)
        else:
            skipped.append({"query": name, "reason": "duplicate"})

    return JSONResponse({"queue_ids": enqueued, "skipped": skipped})


# ---------------------------------------------------------------------------
# Vault siblings (used by the cascade picker)
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_PARENT_FIELD_RE = re.compile(
    r'^\s*parent\s*:\s*"?\[\[([^\]"]+)\]\]"?\s*$', re.MULTILINE
)
_THEME_FIELD_RE = re.compile(
    r'^\s*theme\s*:\s*"?\[\[([^\]"]+)\]\]"?\s*$', re.MULTILINE
)


def _read_topic_frontmatter(path: Path) -> Dict[str, Optional[str]]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"parent": None, "theme": None}
    fm_match = _FRONTMATTER_RE.search(content)
    fm = fm_match.group(1) if fm_match else ""
    parent_match = _PARENT_FIELD_RE.search(fm)
    theme_match = _THEME_FIELD_RE.search(fm)
    return {
        "parent": parent_match.group(1) if parent_match else None,
        "theme": theme_match.group(1) if theme_match else None,
    }


@app.get("/api/vault/siblings")
def vault_siblings(topic: str) -> JSONResponse:
    """Return topics that share a parent with the given topic. If the given
    topic has no parent, returns other root topics in the same domain."""
    topics_dir = Path(_cfg.VAULT_PATH) / "01_Topics"
    if not topics_dir.is_dir():
        return JSONResponse([])

    target_path = topics_dir / f"{topic}.md"
    target_fm = _read_topic_frontmatter(target_path)
    target_parent = target_fm.get("parent")
    target_theme = target_fm.get("theme")

    siblings: List[Dict[str, Optional[str]]] = []
    for path in sorted(topics_dir.glob("*.md")):
        if path.stem == topic:
            continue
        fm = _read_topic_frontmatter(path)
        if target_parent:
            if fm.get("parent") == target_parent:
                siblings.append({
                    "filename_stem": path.stem,
                    "title": path.stem,
                    "parent": fm.get("parent"),
                })
        else:
            # Root topics in the same domain (theme).
            if fm.get("parent") in (None, "") and (
                not target_theme or fm.get("theme") == target_theme
            ):
                siblings.append({
                    "filename_stem": path.stem,
                    "title": path.stem,
                    "parent": None,
                })
    return JSONResponse(siblings)


@app.post("/api/research/{job_id}/resume")
def resume(job_id: str, body: ResumeRequest) -> JSONResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if not job.paused:
        raise HTTPException(status_code=400, detail="job is not paused")
    if not job.resume(body.urls):
        raise HTTPException(status_code=400, detail="resume failed (process not writable)")
    return JSONResponse({"ok": True, "kept": len(body.urls)})


@app.get("/api/jobs")
def get_jobs(limit: int = 50) -> JSONResponse:
    return JSONResponse(list_jobs(limit=limit))


@app.get("/api/research/{job_id}")
def get_job_snapshot(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    snap = job.snapshot()
    snap["lines"] = list(job.lines)
    return JSONResponse(snap)


@app.post("/api/research/{job_id}/cancel")
def cancel(job_id: str) -> JSONResponse:
    ok = cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="cannot cancel (missing or finished)")
    return JSONResponse({"ok": True})


@app.get("/api/research/{job_id}/stream")
def stream(job_id: str) -> StreamingResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    def _sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    def gen():
        # Replay everything we've already buffered, then attach a live listener.
        # We snapshot the current line count so we don't double-deliver lines that
        # arrive between the replay and the listener attach.
        listener = job.attach_listener()
        try:
            # Replay buffered lines.
            replayed = list(job.lines)
            for line in replayed:
                yield _sse("line", {"text": line})
            # If the job is currently paused, replay the pause event so a
            # fresh tab can render the picker.
            if job.paused and job.pause_payload is not None:
                yield _sse("pause", job.pause_payload)
            # If the job already produced a cascade context, replay it.
            if job.cascade_context is not None:
                yield _sse("cascade", job.cascade_context)
            # If the job finished while we were replaying (or before we attached),
            # emit the done event and stop.
            if job.done:
                yield _sse(
                    "done",
                    {"exit_code": job.exit_code, "finished_at": job.finished_at},
                )
                return
            # Live tail. The listener gets every event; sentinel None ends it.
            # Only `line` events are appended to job.lines, so we count those to
            # skip ones we already replayed.
            replayed_count = len(replayed)
            seen_lines = 0
            while True:
                try:
                    item = listener.get(timeout=15.0)
                except _queue.Empty:
                    # Heartbeat keeps proxies and the EventSource alive.
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    yield _sse(
                        "done",
                        {"exit_code": job.exit_code, "finished_at": job.finished_at},
                    )
                    return
                ev = item.get("event")
                if ev == "line":
                    seen_lines += 1
                    if seen_lines <= replayed_count:
                        # Already delivered via replay.
                        continue
                    yield _sse("line", {"text": item.get("text", "")})
                elif ev == "pause":
                    # Forward everything except the event tag.
                    payload = {k: v for k, v in item.items() if k != "event"}
                    yield _sse("pause", payload)
                elif ev == "resumed":
                    yield _sse("resumed", {})
                elif ev == "cascade":
                    payload = {k: v for k, v in item.items() if k != "event"}
                    yield _sse("cascade", payload)
                else:
                    # Unknown event: drop silently.
                    continue
        finally:
            job.detach_listener(listener)

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ---------------------------------------------------------------------------
# Entry point: `python -m web` or `python web/app.py`
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Local web UI for the research pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args()

    print(f"\n  Open http://{args.host}:{args.port} in your browser.\n")
    uvicorn.run(
        "web.app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
