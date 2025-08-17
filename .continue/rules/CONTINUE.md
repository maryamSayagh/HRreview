# CONTINUE.md - Project Guide

## 1. Project Overview

This project is a framework for extracting, classifying, and filtering academic papers from various scholarly sources, likely for literature review or bias analysis. It includes components for interacting with Google Scholar (through SerpAPI), Scopus (through pybliometrics), and Elsevier APIs, along with local journal filtering using GPT-based classification via OpenAI.

**Key technologies:**  
- Python  
- pandas  
- OpenAI GPT API  
- SerpAPI  
- pybliometrics (Scopus)  
- (Elsevier client library)  

**High-level architecture:**
- Modular extractors for different data providers (Google Scholar, Scopus, Elsevier)
- Shared abstraction layer for paper data
- GPT-driven data cleaning/classification tools
- Jupyter Notebooks for exploratory analysis

---

## 2. Getting Started
### Prerequisites
- Python 3.10+
- `pip` package manager
- API keys for SerpAPI, Scopus (configured for pybliometrics), Elsevier, and OpenAI

### Installation
```bash
pip install -r review/requirements.txt
```
(You may need to manually install proprietary/specialized libraries referenced in the code, such as `serpapi`, `pybliometrics`, or a specific Elsevier client. See code comments for guidance.)

### Usage Example
- Run extractors as Python scripts or import their classes in notebooks / workflows
- Example (from a notebook):
    ```python
    from review.Scripts.extract_papers import main
    df = main(query="machine learning bias", api_key=API_KEY, provider="scopus")
    print(df.head())
    ```
- For journal filtering using GPT (see `journal_filter_gpt.py`):
    ```python
    import openai
    from review.Scripts.journal_filter_gpt import filter_journal_dataframe
    df_filtered = filter_journal_dataframe(df, client=openai.OpenAI(...), column_names=[...], context_text="...")
    ```
### Running Tests
- **Tests not detected:** Recommend creating a `tests/` folder with basic coverage.

---

## 3. Project Structure

- **review/Scripts/extract_papers.py**  
  Entry point & implementations for extracting papers from Google Scholar, Scopus, and Elsevier. Includes abstract `PaperExtractor` interface and provider-specific classes.
- **review/Scripts/paper.py**  
  Core `Paper` data structure with serialization and helper methods.
- **review/Scripts/journal_filter_gpt.py**  
  Filtering/cleaning utilities utilizing OpenAI GPT for classification of journals or other metadata.
- **review/Scripts/bias_scopus.ipynb**  
  Jupyter exploratory analysis (Scopus lists, filtering logic, demo)
- **review/data/**  
  Source and processed CSV/XLSX paper/journal/lists data
- **review/requirements.txt**  
  Python dependency list (not always complete)

**Important config files:**
- `review/requirements.txt` — Python dependencies
- `.env`/local API key config — not directly present but referenced in code

---

## 4. Development Workflow
- **Coding Standards:**  
  Follow PEP8 and SOLID principles for class design (see code comments).
- **Testing:**  
  No tests present; add tests for extractors and filter utilities.
- **Build/Deployment:**  
  Not automated; recommend using a Makefile or simple bash scripts for routine tasks.
- **Contribution Guidelines:**  
  Not found—add a `CONTRIBUTING.md` describing code/environments/PR process.

---

## 5. Key Concepts
- **Paper extraction:**  
  Modular, provider-driven approach to acquiring structured paper metadata
- **Abstraction layer:**  
  Interfaces and base classes minimize duplication/spaghetti code
- **GPT-based filtering:**  
  Large language model inference is used to handle ambiguous/subjective filtering
- **Chunked processing:**  
  Large lists broken into smaller groups for manageable GPT API calls

---

## 6. Common Tasks
- **Extract papers from a provider:**  
  Use `main()` in `extract_papers.py`; specify provider (`googlescholar`, `scopus`, `elsevier`)
- **Add a new extractor/provider:**  
  Inherit from `PaperExtractor`, add to `EXTRACTORS` dictionary
- **Filter a DataFrame of journals:**  
  Use `filter_journal_dataframe()` with required client/context/columns
- **Extend paper features:**  
  Modify `Paper` in `paper.py` (add fields/methods as required)

---

## 7. Troubleshooting
- **ImportError for provider APIs:**  
  Install missing library or check import path per error message
- **OpenAI API errors:**  
  Ensure API key is set & correct, check quota, and test with trivial prompt
- **Incomplete data extraction:**  
  Some APIs rate-limit or restrict output—try smaller batch sizes/queries
- **Missing/extra dependencies:**  
  `requirements.txt` may need syncing with your environment!

---

## 8. References
- [pandas Documentation](https://pandas.pydata.org/)
- [OpenAI API](https://platform.openai.com/docs/)
- [pybliometrics](https://pybliometrics.readthedocs.io/en/stable/)
- [SerpAPI (Google Scholar)](https://serpapi.com/)
- Elsevier API Docs: See client library documentation

---

*This file was generated automatically. Please review for accuracy and update with any project-specific details not captured above. You may create further `.md` rules files in subfolders for deeper documentation.*
