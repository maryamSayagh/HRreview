# paper.py
# Class definition for research papers collected from Scopus and Elsevier

from typing import List, Optional

class Paper:
    """
    Represents a research paper collected from Scopus or Elsevier databases.
    Adheres to Single Responsibility Principle by handling only paper metadata.
    """
    
    def __init__(self,
                 title: str,
                 authors: List[str],
                 publication_year: int,
                 journal: str,
                 doi: Optional[str] = None,
                 abstract: Optional[str] = None,
                 source: Optional[str] = None,
                 keywords: Optional[List[str]] = None):
        self.title = title
        self.authors = authors
        self.publication_year = publication_year
        self.journal = journal
        self.doi = doi
        self.abstract = abstract
        self.source = source  # 'Scopus', 'Elsevier', etc.
        self.keywords = keywords if keywords else []
    
    def __repr__(self):
        return f"<Paper title='{self.title}', year={self.publication_year}, source={self.source}>"
    
    def add_keyword(self, keyword: str):
        if keyword not in self.keywords:
            self.keywords.append(keyword)
    
    def to_dict(self):
        """Serializes the paper metadata to a dictionary."""
        return {
            'title': self.title,
            'authors': self.authors,
            'publication_year': self.publication_year,
            'journal': self.journal,
            'doi': self.doi,
            'abstract': self.abstract,
            'source': self.source,
            'keywords': self.keywords
        }
