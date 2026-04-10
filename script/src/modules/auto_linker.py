"""Auto-detect and add wikilinks."""
import re
from typing import Set
from src.modules.vault_scanner import VaultScanner
from config import CONCEPT_RELATIONSHIPS


class AutoLinker:
    def __init__(self, vault_path: str):
        self.scanner = VaultScanner(vault_path)
    
    def find_linkable_terms(self, content: str, theme_id: str) -> Set[str]:
        """Find terms that should be linked."""
        linkable = set()
        
        # Get existing notes
        existing_notes = self.scanner.find_notes_by_theme(theme_id)
        content_lower = content.lower()
        
        # Check for existing note titles
        for note in existing_notes:
            if note['title'].lower() in content_lower:
                linkable.add(note['title'])
        
        # Check for known concepts
        concepts = CONCEPT_RELATIONSHIPS.get(theme_id, {})
        for category, concept_list in concepts.items():
            for concept in concept_list:
                if concept in content_lower:
                    # Format as note name
                    note_name = f"{theme_id.capitalize()} - {concept.capitalize()}"
                    linkable.add(note_name)
        
        return linkable
    
    def add_wikilinks_to_content(self, content: str, linkable_terms: Set[str]) -> str:
        """Add [[wikilinks]] to content."""
        modified = content
        
        for term in sorted(linkable_terms, key=len, reverse=True):
            if f"[[{term}]]" in modified:
                continue
            
            # Replace first occurrence
            pattern = r'\b' + re.escape(term) + r'\b'
            modified = re.sub(pattern, f"[[{term}]]", modified, count=1, flags=re.IGNORECASE)
        
        return modified