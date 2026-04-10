#!/usr/bin/env python3
"""
Smart auto-research system.
Usage: python smart_research.py "your search query"
"""

import sys
import requests
from pathlib import Path
from slugify import slugify

from config import SEARXNG_URL, VAULT_PATH, FOLDERS
from src.modules.theme_detector import ThemeDetector
from src.modules.multi_source_extractor import MultiSourceExtractor
from src.modules.llm_note_generator import LLMNoteGenerator
from src.modules.auto_linker import AutoLinker


def search_web(query: str, max_results: int = 8) -> list:
    """Search SearXNG."""
    print(f"🔍 Searching: {query}")
    
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={'q': query, 'format': 'json'},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        results = [
            {'title': r.get('title'), 'url': r.get('url')}
            for r in data.get('results', [])[:max_results]
            if r.get('url')
        ]
        
        print(f"✓ Found {len(results)} results")
        return results
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []


def save_note(content: str, filename: str) -> Path:
    """Save note to vault."""
    vault = Path(VAULT_PATH)
    target_dir = vault / FOLDERS['research']
    target_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = target_dir / filename
    filepath.write_text(content, encoding='utf-8')
    
    return filepath


def main():
    if len(sys.argv) < 2:
        print("Usage: python smart_research.py 'your search query'")
        sys.exit(1)
    
    query = ' '.join(sys.argv[1:])
    
    print("\n" + "="*70)
    print(f"🎯 SMART RESEARCH: {query}")
    print("="*70 + "\n")
    
    # Step 1: Detect theme
    print("📊 STEP 1: Theme Detection")
    detector = ThemeDetector()
    theme = detector.detect_from_query(query)
    
    if theme:
        print(f"   ✓ Theme: {theme['name']}")
    else:
        theme = {'id': 'general', 'name': 'General', 'tags': ['research']}
    
    # Step 2: Web search
    print(f"\n🌐 STEP 2: Web Search")
    search_results = search_web(query)
    
    if not search_results:
        print("❌ No results")
        return
    
    # Step 3: Extract sources
    print(f"\n📚 STEP 3: Multi-Source Extraction")
    extractor = MultiSourceExtractor()
    sources = extractor.extract_multiple([r['url'] for r in search_results], top_n=3)
    
    if not sources:
        print("❌ No content extracted")
        return
    
    # Step 4: Generate note with LLM
    print(f"\n🤖 STEP 4: LLM Note Generation")
    generator = LLMNoteGenerator()
    note_content = generator.generate_lecture_note(query, sources, theme['name'])
    
    if not note_content:
        print("❌ Generation failed")
        return
    
    # Step 5: Auto-linking
    print(f"\n🔗 STEP 5: Auto-Linking")
    linker = AutoLinker(VAULT_PATH)
    linkable_terms = linker.find_linkable_terms(note_content, theme['id'])
    print(f"   Found {len(linkable_terms)} linkable terms")
    
    linked_content = linker.add_wikilinks_to_content(note_content, linkable_terms)
    final_content = generator.add_metadata_footer(linked_content, sources, theme['name'])
    
    # Step 6: Save
    print(f"\n💾 STEP 6: Saving")
    filename = slugify(query) + ".md"
    filepath = save_note(final_content, filename)
    
    print(f"   ✓ Saved: {filepath}")
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"📄 {filepath.name}")
    print(f"🎯 {theme['name']}")
    print(f"📚 {len(sources)} sources")
    print(f"🔗 {len(linkable_terms)} links")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()