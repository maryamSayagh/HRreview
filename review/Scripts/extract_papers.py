
# --- ABSTRACTION LAYER ---
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Type
import pandas as pd
from review.Scripts.paper import Paper
# --- GOOGLE SCHOLAR IMPORT & CHECK ---
def start_gs():
    try:
       from serpapi import GoogleSearch
    except ImportError as e:
       raise ImportError(
        "The 'serpapi' package is required for Google Scholar extraction. "
        "Install it with 'pip install serpapi'."
        ) from e

# --- ELSEVIER IMPORT & CHECK ---
def start_elsevier():
    try:
        from elsevier_api import ElsevierClient  # Replace with your actual Elsevier API client library
    except ImportError as e:
        pass



try:
    import pybliometrics
    pybliometrics.init()
    from pybliometrics.scopus import ScopusSearch
except ImportError as e:
    raise ImportError(
        "The 'pybliometrics' package is required for Scopus extraction. "
            "Install it with 'pip install pybliometrics'."
        ) from e





# --- ABSTRACTION LAYER ---
class PaperExtractor(ABC):
    """Abstract base class for extracting paper data from various sources."""
    @abstractmethod
    def extract(self, query: str, api_key: str, batch_size: int = 10, max_batches: Optional[int] = None) -> List[Paper]:
        pass

# --- GOOGLE SCHOLAR IMPLEMENTATION ---
class GoogleScholarExtractor(PaperExtractor):
    start_gs()
    def extract(self, query: str, api_key: str, batch_size: int = 10, max_batches: Optional[int] = None) -> List[Paper]:
        search = GoogleSearch({
            "q": query,
            "engine": "google_scholar",
            "api_key": api_key
        })
        all_results = []
        page = 1
        batches = 0
        while True:
            if max_batches and batches >= max_batches:
                break
            print(f"Searching page {page} (Google Scholar)...")
            search.params_dict["start"] = (page - 1) * batch_size
            results = search.get_dict()
            if "organic_results" in results:
                all_results.extend(results["organic_results"])
                if "pagination" in results and "next" in results["pagination"]:
                    page += 1
                    batches += 1
                else:
                    break
            else:
                print("No more results or fetch error:", results)
                break

        papers = []
        for result in all_results:
            paper = Paper(
                title=result.get("title", ""),
                authors=[author.get("name", "") for author in result.get("publication_info", {}).get("authors", [])],
                publication_year=result.get("year") or None,
                journal=result.get("publication_info", {}).get("summary", ""),
                doi=(result.get("resources", [{}])[0].get("doi") if result.get("resources") else None),
                abstract=None,  # Not provided
                source="Google Scholar (SerpAPI)",
                keywords=[]
            )
            papers.append(paper)
        return papers

# --- SCOPUS IMPLEMENTATION (MINIMAL) ---
class ScopusExtractor(PaperExtractor):

    def extract(self, query: str, api_key: str, batch_size: int = 25, max_batches: Optional[int] = None) -> List[Paper]:
        # Assumes ScopusSearch is configured to use your API Key (pybliometrics uses config/scopus.ini)
        scopus_search = ScopusSearch(query)
        results = scopus_search.results or []
        papers = []
        for res in results:
            paper = Paper(
                title=getattr(res, "title", None),
                authors=(getattr(res, "author_names", "") or "").split(", "),
                publication_year=getattr(res, "coverDate", "")[:4] if getattr(res, "coverDate", None) else None,
                journal=getattr(res, "publicationName", None),
                doi=getattr(res, "doi", None),
                abstract=None,
                source="Scopus",
                keywords=[]
            )
            papers.append(paper)
        return papers

# --- ELSEVIER IMPLEMENTATION (DEMO) ---
class ElsevierExtractor(PaperExtractor):
    start_elsevier()
    def extract(self, query: str, api_key: str, batch_size: int = 25, max_batches: Optional[int] = None) -> List[Paper]:
        # Replace this with your actual Elsevier API client logic
        client = ElsevierClient(api_key=api_key)
        all_results = []
        start = 0
        batch_count = 0
        while True:

            if max_batches and batch_count >= max_batches:
                break  #works if max_batches is provided, not none or 0
            response = client.article_search(query, count=batch_size, start=start)
            articles = getattr(response, "results", [])
            if not articles:
                break
            all_results.extend(articles)
            if len(articles) < batch_size:
                break
            start += batch_size
            batch_count += 1

        papers = []
        for art in all_results:
            paper = Paper(
                title=getattr(art, "title", None),
                authors=[auth["name"] for auth in getattr(art, "authors", [])] if getattr(art, "authors", None) else [],
                publication_year=getattr(art, "publication_year", None),
                journal=getattr(art, "journal", None),
                doi=getattr(art, "doi", None),
                abstract=getattr(art, "abstract", None),
                source="Elsevier",
                keywords=getattr(art, "keywords", [])
            )
            papers.append(paper)
        return papers

# --- REGISTRY/FACTORY ---
EXTRACTORS: Dict[str, Type[PaperExtractor]] = {
    "googlescholar": GoogleScholarExtractor,
    "scopus": ScopusExtractor,
    "elsevier": ElsevierExtractor
}

def get_extractor(provider: str) -> PaperExtractor:
    if provider.lower() not in EXTRACTORS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {list(EXTRACTORS.keys())}")
    return EXTRACTORS[provider.lower()]()

def main(query: str, api_key: str, provider: str = "googlescholar", batch_size: int = 10, max_batches: Optional[int] = None) -> pd.DataFrame:
    """
    Main entrypoint to extract papers as DataFrame.
    Use provider: googlescholar, scopus, elsevier.
    """
    extractor = get_extractor(provider)
    papers = extractor.extract(query, api_key, batch_size=batch_size, max_batches=max_batches)
    data = [paper.to_dict() for paper in papers]
    df = pd.DataFrame(data)
    return df

