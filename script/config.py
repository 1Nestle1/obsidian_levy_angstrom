"""Configuration for hierarchical knowledge base."""
from pathlib import Path

# === PATHS ===
VAULT_PATH = r"C:\Users\user\Documents\levy_angstrom"  # UPDATE THIS!
CACHE_DIR = Path("data/cache")
KNOWLEDGE_GRAPH_DIR = Path("data/knowledge_graph")

# === SEARXNG ===
SEARXNG_URL = "http://localhost:8080"

# === VAULT STRUCTURE ===
FOLDERS = {
    'research': '02_Research',
    'concepts': '03_Concepts',
    'mocs': '04_MOCs',
    'code': '06_Code'
}

# === THEME CONFIGURATION ===
# Programming languages and frameworks
TECH_THEMES = {
    # Languages
    'python': {
        'keywords': ['python', 'py', 'pip', 'virtualenv', 'conda'],
        'hub_name': 'Code - Python',
        'tags': ['code/python', 'programming'],
        'subtopics': ['basics', 'oop', 'functional', 'async', 'web', 'data-science']
    },
    'javascript': {
        'keywords': ['javascript', 'js', 'node', 'npm', 'ecmascript'],
        'hub_name': 'Code - JavaScript',
        'tags': ['code/javascript', 'programming'],
        'subtopics': ['basics', 'es6', 'async', 'dom', 'frameworks']
    },
    'typescript': {
        'keywords': ['typescript', 'ts', 'tsc'],
        'hub_name': 'Code - TypeScript',
        'tags': ['code/typescript', 'programming'],
        'subtopics': ['types', 'interfaces', 'generics', 'decorators']
    },
    
    # Frameworks/Libraries
    'react': {
        'keywords': ['react', 'reactjs', 'jsx', 'tsx'],
        'hub_name': 'Code - React',
        'tags': ['code/react', 'framework', 'frontend'],
        'subtopics': ['components', 'hooks', 'state', 'routing', 'performance']
    },
    'vue': {
        'keywords': ['vue', 'vuejs', 'nuxt'],
        'hub_name': 'Code - Vue',
        'tags': ['code/vue', 'framework', 'frontend'],
        'subtopics': ['components', 'composition-api', 'directives', 'routing']
    },
    'django': {
        'keywords': ['django', 'drf', 'django-rest'],
        'hub_name': 'Code - Django',
        'tags': ['code/django', 'framework', 'backend'],
        'subtopics': ['models', 'views', 'templates', 'orm', 'rest']
    },
    'fastapi': {
        'keywords': ['fastapi', 'starlette', 'pydantic'],
        'hub_name': 'Code - FastAPI',
        'tags': ['code/fastapi', 'framework', 'backend'],
        'subtopics': ['routing', 'models', 'async', 'validation']
    },
}

# Concept relationships (will be learned over time)
CONCEPT_RELATIONSHIPS = {
    'react': {
        'components': ['jsx', 'props', 'state'],
        'hooks': ['useState', 'useEffect', 'useContext', 'custom-hooks'],
        'state': ['props', 'context', 'redux'],
    },
    'python': {
        'decorators': ['functions', 'closures', 'wrappers'],
        'asyncio': ['coroutines', 'event-loop', 'tasks'],
        'oop': ['classes', 'inheritance', 'polymorphism'],
    }
}

# Trusted code sources
CODE_SOURCES = [
    'github.com',
    'stackoverflow.com',
    'docs.python.org',
    'react.dev',
    'developer.mozilla.org',
    'realpython.com',
    'freecodecamp.org',
    'dev.to',
    'medium.com'
]