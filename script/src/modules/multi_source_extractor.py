"""Extract content from multiple sources."""
import trafilatura
import requests
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from config import CODE_SOURCES


class MultiSourceExtractor:
    def __init__(self, max_workers: int = 5, timeout: int = 15):
        self.max_workers = max_workers
        self.timeout = timeout
    
    def extract_single(self, url: str) -> Optional[Dict]:
        """Extract content from one URL."""
        try:
            print(f"  📥 {url[:60]}...")
            
            # Download with requests first (with timeout)
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={'User-Agent': 'Mozilla/5.0 (Research Bot)'}
            )
            response.raise_for_status()
            downloaded = response.text
            
            # Extract text using trafilatura
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                output_format='markdown'
            )
            
            if not text or len(text) < 300:
                print(f"  ⚠️  Content too short")
                return None
            
            # Score quality
            quality = 50
            for trusted in CODE_SOURCES:
                if trusted in url:
                    quality += 30
                    break
            
            if len(text) > 2000:
                quality += 10
            
            if '```' in text or 'code>' in text:
                quality += 5
            
            if quality < 40:
                print(f"  ⚠️  Low quality ({quality}/100)")
                return None
            
            print(f"  ✓ {len(text)} chars (quality: {quality}/100)")
            
            # Extract title from URL or content
            title = url.split('//')[-1].split('/')[0]
            if '<title>' in downloaded:
                import re
                title_match = re.search(r'<title>([^<]+)</title>', downloaded)
                if title_match:
                    title = title_match.group(1).strip()
            
            return {
                'url': url,
                'title': title,
                'content': text,
                'quality_score': quality,
                'word_count': len(text.split())
            }
            
        except requests.Timeout:
            print(f"  ⏱️  Timeout")
            return None
        except requests.RequestException as e:
            print(f"  ❌ Request failed: {str(e)[:30]}")
            return None
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:50]}")
            return None
    
    def extract_multiple(self, urls: List[str], top_n: int = 3) -> List[Dict]:
        """Extract from multiple URLs in parallel."""
        print(f"\n🔄 Extracting from {len(urls)} sources...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.extract_single, url) for url in urls]
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        
        if not results:
            print("\n⚠️  No content extracted from any source")
            return []
        
        # Sort by quality
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        top_results = results[:top_n]
        
        print(f"\n✓ Successfully extracted {len(top_results)} high-quality sources")
        for r in top_results:
            print(f"   - {r['title'][:50]} ({r['quality_score']}/100)")
        
        return top_results