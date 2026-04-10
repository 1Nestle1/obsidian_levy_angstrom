"""Scan vault for existing notes."""
import re
from pathlib import Path
from typing import List, Dict
from config import VAULT_PATH, FOLDERS


class VaultScanner:
    def __init__(self, vault_path: str = VAULT_PATH):
        self.vault_path = Path(vault_path)
        self.research_dir = self.vault_path / FOLDERS['research']
    
    def find_notes_by_theme(self, theme_id: str) -> List[Dict]:
        """Find all notes related to a theme."""
        notes = []
        
        if not self.research_dir.exists():
            return notes
        
        # Look for notes starting with theme name
        pattern = f"{theme_id}*.md"
        
        for note_path in self.research_dir.glob(pattern):
            if note_path.is_file():
                content = note_path.read_text(encoding='utf-8')
                title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                
                notes.append({
                    'path': note_path,
                    'filename': note_path.name,
                    'title': title_match.group(1) if title_match else note_path.stem,
                    'links': re.findall(r'\[\[([^\]]+)\]\]', content)
                })
        
        return notes