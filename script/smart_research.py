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
from src.modules.llm_factory import (
    describe_provider,
    get_llm_client,
    interactive_pick,
    list_providers,
    resolve_provider,
)
from src.modules.llm_probe import (
    choose_llm,
    load_last_provider,
    probe_provider,
    save_last_provider,
)
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
    provider: str | None = None,
    extract_provider: str | None = None,
    ask_hybrid: bool = False,
) -> int:
    banner(f"RESEARCH: {query}")

    vault_root = Path(vault_path)
    ensure_vault_structure(vault_root)

    # --- Step 1: vault index + theme detection ---
    print("\n[step 1] Indexing vault + detecting theme")
    vault = build_index(vault_root)
    print(f"  vault: {len(vault.mocs)} MOCs, {len(vault.concepts)} concepts, "
          f"{len(vault.research)} research notes")

    print(f"  llm: {describe_provider(provider)}")
    try:
        resolved = resolve_provider(provider)
    except LLMError as e:
        print(f"  [abort] {e}")
        return 1

    probe = probe_provider(resolved.name)
    if not probe.ready:
        print(f"  [warn] provider '{resolved.name}' not ready: {probe.status} ({probe.error or '—'})")
        fallback = choose_llm()
        if not fallback:
            return 1
        provider = fallback
        resolved = resolve_provider(provider)
        probe = probe_provider(provider)

    if probe.available_models and not probe.extract_available:
        print(f"  [warn] extract model '{resolved.extract_model}' not in provider — requests may 404")
    if probe.available_models and not probe.synthesize_available:
        print(f"  [warn] synthesize model '{resolved.synthesize_model}' not in provider — requests may 404")

    try:
        llm = get_llm_client(provider)
    except LLMError as e:
        print(f"  [abort] {e}")
        return 1

    save_last_provider(resolved.name)
    print(f"  models available: {len(probe.available_models)}  latency: {probe.latency_ms}ms")

    # --- Optional hybrid mode: different provider for Pass 1 (extract) ---
    extract_llm = llm
    extract_resolved = resolved
    if ask_hybrid and not extract_provider:
        ans = input("  Use a different provider for Pass 1 extract (saves cloud tokens)? [y/N]: ").strip().lower()
        if ans == "y":
            print("  → pick the EXTRACT provider:")
            extract_provider = choose_llm()
    if extract_provider and extract_provider != resolved.name:
        try:
            extract_resolved = resolve_provider(extract_provider)
            extract_llm = get_llm_client(extract_provider)
            print(f"  hybrid: extract={extract_provider}/{extract_resolved.extract_model}  "
                  f"synth={resolved.name}/{resolved.synthesize_model}")
        except LLMError as e:
            print(f"  [warn] hybrid extract provider unusable ({e}); falling back to single-provider")
            extract_llm = llm
            extract_resolved = resolved

    detector = ThemeDetector(
        vault_index=vault, ollama=extract_llm, model=extract_resolved.extract_model,
    )
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
    generator = LLMNoteGenerator(
        ollama=extract_llm,
        extract_model=extract_resolved.extract_model,
        synthesize_model=resolved.synthesize_model,
        synth_client=llm,
    )
    summaries = generator.extract_all(sources)
    if not summaries:
        print("  [abort] no summaries produced")
        return 4
    print(f"  {len(summaries)} summaries")

    # --- Step 5: Pass 2 — synthesize hub DOC ---
    print(f"\n[step 5] Pass 2 — synthesizing hub ({theme.template_family} template)")
    display_title = _display_title(query)
    topic_filename = sanitize_filename(display_title)
    existing_vault_titles = {n.filename_stem for n in vault.all_notes}

    raw = generator.synthesize_pack(
        topic=display_title,
        topic_filename=topic_filename,
        domain=theme.domain,
        subtheme=theme.subtheme,
        summaries=summaries,
        existing_links=theme.linked_existing_nodes,
        new_concepts=theme.proposed_new_concepts,
        template_family=theme.template_family,
    )
    if not raw:
        print("  [abort] synthesis failed")
        return 5

    pack = parse_pack(raw)
    if not pack.files:
        print("  [abort] parser found no === FILE: === blocks in output")
        _dump_debug_output(query, raw)
        return 6

    # Inject code examples (harvested, no LLM) and sources bibliography
    # into the hub file placeholders.
    _fill_placeholders(pack, sources, summaries)

    print(f"  parsed {len(pack.files)} file block(s):")
    for f in pack.files:
        print(f"    - {f.rel_path}")

    # --- Step 6: validate (light — we no longer require the old triplet) ---
    print("\n[step 6] Validating pack")
    validation = validate_pack(
        pack,
        required_concepts=[],  # folded into hub; no separate concept files required
        existing_vault_titles=existing_vault_titles,
    )
    if validation.unknown_links:
        print(f"  [warn] unknown wikilinks: {validation.unknown_links}")

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

    # Link the new hub into its domain MOC.
    _link_topic_in_domain_moc(vault_root, theme.domain, topic_filename, theme.subtheme)

    _save_run_manifest(query, theme, sources, summaries, pack, report, resolved)

    banner("DONE")
    print(f"  files written: {len(report.written)}")
    print(f"  skipped:       {len(report.skipped)}")
    return 0


def _build_examples_section(sources: list) -> str:
    """Build the Examples section from harvested code_blocks (no LLM).
    Returns '' if no source has any code blocks."""
    labels = ["Basic Example", "Advanced Example", "Additional Example"]
    picked: list = []
    # Prefer code blocks from higher-ranked sources (sources arrive already sorted).
    for s in sources:
        for block in s.get("code_blocks", []) or []:
            picked.append((s, block))
            if len(picked) >= 3:
                break
        if len(picked) >= 3:
            break

    if not picked:
        return ""

    # If only one block, call it "Example" instead of "Basic Example".
    if len(picked) == 1:
        labels = ["Example"]

    out: list[str] = []
    for i, (source, block) in enumerate(picked):
        label = labels[i] if i < len(labels) else f"Example {i+1}"
        ctx = block.get("context", "").strip()
        ctx_line = f"*From the section “{ctx}” in the source.*\n\n" if ctx else ""
        out.append(
            f"### {label}\n\n"
            f"{ctx_line}"
            f"```{block.get('language', 'text')}\n"
            f"{block.get('code', '').rstrip()}\n"
            f"```\n\n"
            f"*Source: [{source.get('title') or source['url']}]({source['url']})*\n"
        )
    return "\n".join(out).rstrip() + "\n"


def _build_sources_section(sources: list, summaries: list) -> str:
    """Deterministic annotated bibliography appended at bottom of hub."""
    access_date = datetime.now(timezone.utc).date().isoformat()
    blurb_by_url = {s.get("url", ""): (s.get("one_line_summary") or "").strip() for s in summaries}
    lines: list[str] = []
    for s in sources:
        url = s["url"]
        title = s.get("title") or url
        blurb = blurb_by_url.get(url, "")
        lines.append(f"- [{title}]({url})")
        if blurb:
            lines.append(f"  - {blurb}")
        lines.append(f"  - Accessed: {access_date}")
    return "\n".join(lines) + "\n"


def _fill_placeholders(pack, sources: list, summaries: list) -> None:
    """Replace <!-- EXAMPLES_PLACEHOLDER --> and <!-- SOURCES_PLACEHOLDER -->
    inside the hub file with deterministic content. Mutates pack in place."""
    examples = _build_examples_section(sources)
    bibliography = _build_sources_section(sources, summaries)

    for i, f in enumerate(pack.files):
        if not f.rel_path.startswith("01_Topics/"):
            continue
        content = f.content
        if "<!-- EXAMPLES_PLACEHOLDER -->" in content:
            if examples:
                content = content.replace("<!-- EXAMPLES_PLACEHOLDER -->", examples)
            else:
                # Drop the "## 4. Examples" section entirely when no code was harvested.
                content = _drop_section(content, "## 4. Examples")
        content = content.replace("<!-- SOURCES_PLACEHOLDER -->", bibliography)
        # Type-safe replace via dataclass
        from src.modules.pack_parser import PackFile
        pack.files[i] = PackFile(rel_path=f.rel_path, content=content)


def _drop_section(content: str, header: str) -> str:
    """Remove a `## header` section and everything under it up to the next `## ` header."""
    lines = content.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.strip() == header:
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _display_title(query: str) -> str:
    """Preserve camelCase / explicit capitalization; title-case only if all lowercase."""
    q = query.strip()
    if any(c.isupper() for c in q):
        return q
    return q.title()


def _link_topic_in_domain_moc(
    vault_root: Path, domain: str, topic_filename: str, subtheme: str
) -> None:
    """Add `- [[Topic]]` to MOC - {Domain}.md. Creates the file if missing. Idempotent."""
    moc_path = vault_root / "04_MOCs" / f"MOC - {sanitize_filename(domain)}.md"
    link_line = f"- [[{topic_filename}]]"
    if subtheme:
        link_line += f" — {subtheme}"

    if moc_path.exists():
        existing = moc_path.read_text(encoding="utf-8")
        if f"[[{topic_filename}]]" in existing:
            return
        new_content = existing.rstrip() + "\n" + link_line + "\n"
    else:
        moc_path.parent.mkdir(parents=True, exist_ok=True)
        new_content = (
            f"---\ndomain: {domain}\ntags: [moc]\n---\n"
            f"# MOC - {domain}\n\n"
            f"Topics in this domain:\n\n"
            f"{link_line}\n"
        )
    moc_path.write_text(new_content, encoding="utf-8")
    print(f"  linked in {moc_path.relative_to(vault_root)}")


def _dump_debug_output(query: str, raw: str) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dump = RUNS_DIR / f"debug-{stamp}-{sanitize_filename(query)}.txt"
    dump.write_text(raw, encoding="utf-8")
    print(f"  raw output saved to {dump}")


def _save_run_manifest(query, theme, sources, summaries, pack, report, resolved) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"run-{stamp}-{sanitize_filename(query)}.json"
    payload = {
        "query": query,
        "timestamp": stamp,
        "theme": theme.to_dict(),
        "models": {
            "provider": resolved.name,
            "extract": resolved.extract_model,
            "synthesize": resolved.synthesize_model,
        },
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
    parser.add_argument(
        "--provider",
        choices=list_providers(),
        default=None,
        help="LLM provider (default: $LLM_PROVIDER or ollama). Use --pick to choose interactively.",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Interactively pick a provider (no probing) before running.",
    )
    parser.add_argument(
        "--choose-llm",
        action="store_true",
        help="Probe all providers, show status table, and pick interactively.",
    )
    parser.add_argument(
        "--extract-provider",
        choices=list_providers(),
        default=None,
        help="Separate provider for Pass 1 extract (hybrid mode). "
             "E.g. --provider groq --extract-provider ollama to save cloud tokens.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Prompt at runtime whether to use a different provider for extract.",
    )
    args = parser.parse_args(argv)

    query = " ".join(args.query).strip()
    if not query:
        print("empty query")
        return 1

    provider = args.provider
    if args.choose_llm and not provider:
        provider = choose_llm()
        if not provider:
            return 1
    elif args.pick and not provider:
        provider = interactive_pick()
    elif not provider:
        cached = load_last_provider()
        if cached:
            provider = cached
            print(f"llm: using cached provider '{cached}' (override with --choose-llm or --provider)")

    return run(
        query,
        vault_path=args.vault,
        max_results=args.max_results,
        top_sources=args.top_sources,
        dry_run=args.dry_run,
        provider=provider,
        extract_provider=args.extract_provider,
        ask_hybrid=args.hybrid,
    )


if __name__ == "__main__":
    # Only call sys.exit for non-zero codes so a clean success run doesn't
    # surface SystemExit as a traceback in some terminals (Python 3.14+).
    _code = main()
    if _code:
        sys.exit(_code)
