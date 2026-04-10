"""
Extract key concepts from content.
"""
import re
from typing import List, Dict
from script.config import CONCEPT_RELATIONSHIPS


class ConceptExtractor:
    """Extract programming concepts from text."""
    
    def __init__(self):
        self.relationships = CONCEPT_RELATIONSHIPS
    
    def extract_concepts(self, content: str, theme_id: str) -> List[str]:
        """
        Extract key concepts from content.
        
        Args:
            content: Text content
            theme_id: Theme identifier (e.g., 'react', 'python')
            
        Returns:
            List of concept names
        """
        concepts = []
        content_lower = content.lower()
        
        # Get known concepts for this theme
        theme_concepts = self.relationships.get(theme_id, {})
        
        for category, concept_list in theme_concepts.items():
            for concept in concept_list:
                # Look for concept mentions
                pattern = r'\b' + re.escape(concept.replace('-', r'[\s-]')) + r'\b'
                if re.search(pattern, content_lower):
                    concepts.append(concept)
        
        # Also check for category names
        for category in theme_concepts.keys():
            pattern = r'\b' + re.escape(category) + r'\b'
            if re.search(pattern, content_lower):
                if category not in concepts:
                    concepts.append(category)
        
        return list(set(concepts))
    
    def extract_code_blocks(self, content: str) -> List[Dict]:
        """Extract code blocks from markdown content."""
        # Pattern for fenced code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        code_blocks = []
        for lang, code in matches:
            code_blocks.append({
                'language': lang or 'text',
                'code': code.strip()
            })
        
        return code_blocks
    
    def get_related_concepts(self, concept: str, theme_id: str) -> List[str]:
        """Get concepts related to a given concept."""
        theme_concepts = self.relationships.get(theme_id, {})
        
        # Find which category this concept belongs to
        for category, concept_list in theme_concepts.items():
            if concept in concept_list:
                return concept_list
            if concept == category:
                return concept_list
        
        return []


# Test
if __name__ == "__main__":
    extractor = ConceptExtractor()
    
    sample_content = """
    React hooks are a powerful feature. The useState hook manages state,
    while useEffect handles side effects. Custom hooks can combine both.
    """
    
    concepts = extractor.extract_concepts(sample_content, 'react')
    print("Extracted concepts:", concepts)
    
    related = extractor.get_related_concepts('useState', 'react')
    print("Related to useState:", related)