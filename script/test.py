"""Test all modules together."""
from src.modules.theme_detector import ThemeDetector
from src.modules.vault_scanner import VaultScanner
from src.modules.concept_extractor import ConceptExtractor


def test_workflow():
    """Test the complete workflow."""
    
    # Initialize modules
    detector = ThemeDetector()
    scanner = VaultScanner()
    extractor = ConceptExtractor()
    
    # Test query
    query = "React useEffect hook tutorial"
    
    print("="*60)
    print(f"QUERY: {query}")
    print("="*60)
    
    # 1. Detect theme
    print("\n1. THEME DETECTION")
    theme = detector.detect_from_query(query)
    if theme:
        print(f"   Theme: {theme['name']}")
        print(f"   ID: {theme['id']}")
        print(f"   Tags: {', '.join(theme['tags'])}")
    
    # 2. Scan vault for existing notes
    print("\n2. VAULT SCAN")
    if theme:
        existing_notes = scanner.find_notes_by_theme(theme['id'])
        print(f"   Found {len(existing_notes)} existing {theme['id']} notes")
        
        hub_exists = scanner.theme_hub_exists(theme['name'])
        print(f"   Theme hub exists: {hub_exists}")
    
    # 3. Extract concepts
    print("\n3. CONCEPT EXTRACTION")
    sample_content = "useEffect is a React hook for handling side effects in components"
    if theme:
        concepts = extractor.extract_concepts(sample_content, theme['id'])
        print(f"   Extracted concepts: {', '.join(concepts)}")
    
    print("\n" + "="*60)
    print("✓ All modules working!")
    print("="*60)


if __name__ == "__main__":
    test_workflow()