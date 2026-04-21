#!/usr/bin/env python3
"""Research pipeline: query -> search -> extract -> theme -> 2-pass LLM -> note pack."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from config import (
    DEFAULT_MAX_RESULTS,
    DEFAULT_TOP_SOURCES,
    MODEL_EXTRACT,
    MODEL_SYNTHESIZE,
    RUNS_DIR,
    VAULT_PATH,
)
from src.modules.llm_factory import describe_provider, get_llm_client
from src.modules.llm_note_generator import LLMNoteGenerator
from src.modules.multi_source_extractor import MultiSourceExtractor
from src.modules.openai_compat_client import LLMError
from src.modules.pack_parser import parse as parse_pack
from src.modules.pack_parser import validate_pack, write_pack
from src.modules.search import SearchClient
from src.modules.theme_detector import ThemeDetector
from src.modules.vault_index import build_index, ensure_vault_structure
from src.utils.filenames import sanitize_filename


def banner(line: str) -> None:
    print("\n" + "=" * 70)
    print(line)
    print("=" * 70)


def run(
    query: str,
    vault_path: str = VAULT_PATH,
    max_results: int = DEFAULT_MAX_RESULTS,
    top_sources: int = DEFAULT_TOP_SOURCES,
    dry_run: bool = False,
) -> int:
    banner(f"RESEARCH: {query}")

    vault_root = Path(vault_path)
    ensure_vault_structure(vault_root)

    # --- Step 1: vault index + theme detection ---
    print("\n[step 1] Indexing vault + detecting theme")
    vault = build_index(vault_root)
    print(f"  vault: {len(vault.mocs)} MOCs, {len(vault.concepts)} concepts, "
          f"{len(vault.research)} research notes")

    print(f"  llm: {describe_provider()}")
    try:
        llm = get_llm_client()
    except LLMError as e:
        print(f"  [abort] {e}")
        return 1

    available = llm.list_models()
    if available:
        print(f"  models available: {len(available)}")
    else:
        print("  [warn] model list empty — provider may be unreachable or auth missing.")

    detector = ThemeDetector(vault_index=vault, ollama=llm)
    theme = detector.detect(query)
    print(f"  domain:   {theme.domain} (new={theme.domain_is_new})")
    print(f"  subtheme: {theme.subtheme}")
    print(f"  confidence: {theme.confidence:.2f}")
    print(f"  existing links: {theme.linked_existing_nodes}")
    print(f"  new concepts:   {theme.proposed_new_concepts}")

    # --- Step 2: web search ---
    print("\n[step 2] Searching")
    search = SearchClient()
    results = search.search(query, max_results=max_results, domain=theme.domain)
    if not results:
        print("  [abort] no search results")
        return 2
    print(f"  {len(results)} results")

    # --- Step 3: extract sources ---
    print("\n[step 3] Extracting sources")
    extractor = MultiSourceExtractor()
    sources = extractor.extract_multiple(
        [r["url"] for r in results],
        top_n=top_sources,
    )
    if not sources:
        print("  [abort] no usable sources")
        return 3

    # --- Step 4: Pass 1 — per-source JSON extraction ---
    print("\n[step 4] Pass 1 — per-source extraction")
    generator = LLMNoteGenerator(ollama=llm)
    summaries = generator.extract_all(sources)
    if not summaries:
        print("  [abort] no summaries produced")
        return 4
    print(f"  {len(summaries)} summaries")

    # --- Step 5: Pass 2 — synthesize pack ---
    print("\n[step 5] Pass 2 — synthesizing note pack")
    topic_filename = sanitize_filename(query.title())
    existing_vault_titles = {n.filename_stem for n in vault.all_notes}

    raw = generator.synthesize_pack(
        topic=query,
        topic_filename=topic_filename,
        domain=theme.domain,
        subtheme=theme.subtheme,
        summaries=summaries,
        existing_links=theme.linked_existing_nodes,
        new_concepts=theme.proposed_new_concepts,
    )
    if not raw:
        print("  [abort] synthesis failed")
        return 5

    pack = parse_pack(raw)
    if not pack.files:
        print("  [abort] parser found no === FILE: === blocks in output")
        _dump_debug_output(query, raw)
        return 6

    # Override the Sources block with a deterministic version built from
    # the actual summaries — small models tend to echo template guidance here.
    _inject_deterministic_sources(pack, query, topic_filename, sources, summaries)

    print(f"  parsed {len(pack.files)} file blocks:")
    for f in pack.files:
        print(f"    - {f.rel_path}")

    # --- Step 6: validate ---
    print("\n[step 6] Validating pack")
    validation = validate_pack(
        pack,
        required_concepts=theme.proposed_new_concepts,
        existing_vault_titles=existing_vault_titles,
    )
    if validation.missing_concepts:
        print(f"  [warn] missing concept file blocks: {validation.missing_concepts}")
    if validation.unknown_links:
        print(f"  [warn] unknown wikilinks: {validation.unknown_links}")
    if validation.orphan_concept_files:
        print(f"  [warn] orphan concept files: {validation.orphan_concept_files}")

    # --- Step 7: write ---
    if dry_run:
        print("\n[dry-run] not writing to vault. Pack preview:")
        for f in pack.files:
            print(f"\n--- {f.rel_path} ---")
            print(f.content[:400])
        return 0

    print("\n[step 7] Writing pack")
    report = write_pack(pack, vault_root)
    for p in report.written:
        print(f"  wrote {p.relative_to(vault_root)}")
    for rel, reason in report.skipped:
        print(f"  [skip] {rel}: {reason}")

    _save_run_manifest(query, theme, sources, summaries, pack, report)

    banner("DONE")
    print(f"  files written: {len(report.written)}")
    print(f"  skipped:       {len(report.skipped)}")
    return 0


def _inject_deterministic_sources(
    pack, query: str, topic_filename: str, sources: list, summaries: list
) -> None:
    """Replace the Sources file content with a deterministic listing built from
    Pass 1 summaries + extracted source titles."""
    from src.modules.pack_parser import PackFile

    access_date = datetime.now(timezone.utc).date().isoformat()
    title_by_url = {s["url"]: s.get("title") or s["url"] for s in sources}
    blurb_by_url = {s.get("url", ""): (s.get("one_line_summary") or "").strip() for s in summaries}

    lines = [
        "---",
        f"topic: {query}",
        "tags: [sources]",
        "---",
        f"# Sources - {query}",
        "",
    ]
    for s in sources:
        url = s["url"]
        title = title_by_url.get(url, url)
        blurb = blurb_by_url.get(url, "")
        lines.append(f"- [{title}]({url})")
        if blurb:
            lines.append(f"  - {blurb}")
        lines.append(f"  - Access date: {access_date}")

    new_content = "\n".join(lines) + "\n"
    rel = f"05_Sources/Sources - {topic_filename}.md"

    for i, f in enumerate(pack.files):
        if f.rel_path.startswith("05_Sources/"):
            pack.files[i] = PackFile(rel_path=rel, content=new_content)
            return

    pack.files.append(PackFile(rel_path=rel, content=new_content))


def _dump_debug_output(query: str, raw: str) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dump = RUNS_DIR / f"debug-{stamp}-{sanitize_filename(query)}.txt"
    dump.write_text(raw, encoding="utf-8")
    print(f"  raw output saved to {dump}")


def _save_run_manifest(query, theme, sources, summaries, pack, report) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"run-{stamp}-{sanitize_filename(query)}.json"
    payload = {
        "query": query,
        "timestamp": stamp,
        "theme": theme.to_dict(),
        "models": {"extract": MODEL_EXTRACT, "synthesize": MODEL_SYNTHESIZE},
        "sources": [{"url": s["url"], "title": s["title"], "score": s["quality_score"]} for s in sources],
        "summary_count": len(summaries),
        "files_written": [str(p) for p in report.written],
        "files_skipped": [{"path": r, "reason": why} for r, why in report.skipped],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research pipeline for Obsidian vault.")
    parser.add_argument("query", nargs="+", help="Research query / topic.")
    parser.add_argument("--vault", default=VAULT_PATH, help="Obsidian vault root.")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--top-sources", type=int, default=DEFAULT_TOP_SOURCES)
    parser.add_argument("--dry-run", action="store_true", help="Don't write to vault.")
    args = parser.parse_args(argv)

    query = " ".join(args.query).strip()
    if not query:
        print("empty query")
        return 1

    return run(
        query,
        vault_path=args.vault,
        max_results=args.max_results,
        top_sources=args.top_sources,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
