"""Generate notes using Ollama LLM."""
import requests
from typing import List, Dict, Optional


class LLMNoteGenerator:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:1b"
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    def generate_lecture_note(self, query: str, sources: List[Dict], theme_name: str = None) -> Optional[str]:
        """Generate comprehensive lecture-style note."""
        
        # Build sources
        sources_list = []
        for s in sources:
            sources_list.append("SOURCE: " + s['title'])
            sources_list.append(s['content'][:2000])
            sources_list.append("")
        
        sources_text = "\n".join(sources_list)
        
        # Build prompt parts
        prompt_parts = [
            "Create comprehensive technical documentation for: ",
            query,
            "\n\nSOURCES:\n",
            sources_text,
            "\n\nCreate a complete note with:\n\n",
            "# ", query, "\n\n",
            "## Overview\n",
            "[2-3 paragraphs explaining clearly]\n\n",
            "## Key Concepts\n",
            "[Detailed explanations with examples]\n\n",
            "## How It Works\n",
            "[Step-by-step explanation]\n\n",
            "## Examples\n\n",
            "### Basic Example\n",
            "```\n[working code example]\n```\n\n",
            "### Advanced Example\n",
            "```\n[complex example]\n```\n\n",
            "## Best Practices\n",
            "- Good practice\n",
            "- Common mistake to avoid\n\n",
            "## Common Pitfalls\n",
            "[What to avoid and why]\n\n",
            "Use ONLY information from sources. Include working code examples."
        ]
        
        prompt = "".join(prompt_parts)

        try:
            print("🤖 Generating note with LLM...")
            
            response = requests.post(
                self.base_url + "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.4, "num_predict": 3000}
                },
                timeout=180
            )
            
            response.raise_for_status()
            result = response.json()
            generated = result.get('response', '').strip()
            
            print("✓ Generated " + str(len(generated)) + " characters")
            return generated
            
        except Exception as e:
            print("❌ LLM failed: " + str(e))
            return None
    
    def add_metadata_footer(self, note_content: str, sources: List[Dict], theme_name: str = None) -> str:
        """Add metadata footer."""
        footer_parts = ["\n\n---\n", "## Sources\n"]
        
        for source in sources:
            footer_parts.append("- [" + source['title'] + "](" + source['url'] + ")")
        
        if theme_name:
            footer_parts.append("\n**Theme**: [[" + theme_name + "]]\n")
        
        footer_parts.append("\n#research #documentation\n")
        
        return note_content + "\n".join(footer_parts)