"""Detect programming theme from query."""
import re
from typing import Optional, Dict
from config import TECH_THEMES


class ThemeDetector:
    def __init__(self):
        self.themes = TECH_THEMES
    
    def detect_from_query(self, query: str) -> Optional[Dict]:
        """Detect theme from search query."""
        query_lower = query.lower()
        
        for theme_id, theme_data in self.themes.items():
            for keyword in theme_data['keywords']:
                if keyword in query_lower:
                    return {
                        'id': theme_id,
                        'name': theme_data['hub_name'],
                        'tags': theme_data['tags'],
                        'subtopics': theme_data['subtopics']
                    }
        return None