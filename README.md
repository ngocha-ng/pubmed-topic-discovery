# PubMed Research Cluster Explorer

An end-to-end NLP pipeline that fetches, processes, clusters, and visualizes biomedical literature from PubMed — deployed as an interactive app on Hugging Face Spaces.

## Project overview

This project collects 20,000+ research articles per domain from the NCBI PubMed database, generates biomedical embeddings, clusters them by topic, and surfaces insights through an interactive visualization.

**Domains covered**
- Drug discovery — drug discovery, small molecule, target identification, lead optimization
- Oncology — oncology, immunotherapy, CAR-T, tumor microenvironment, mRNA cancer vaccine

## Pipeline phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 — Data collection | Bulk fetch from PubMed via NCBI E-utilities API | ✅ Done |
| 2 — Preprocessing | Clean text, generate embeddings (BiomedBERT) | 🔜 Next |
| 3 — Clustering | UMAP dimensionality reduction + HDBSCAN clustering | 🔜 Planned |
| 4 — App | Gradio interactive cluster explorer | 🔜 Planned |
| 5 — Deploy | Hugging Face Spaces | 🔜 Planned |

## Data schema

Each record contains:

| Column | Description |
|--------|-------------|
| `pmid` | PubMed article ID |
| `title` | Article title |
| `abstract` | Full abstract text |
| `pub_date` | Publication date |
| `keywords` | Author-supplied keywords |
| `mesh_terms` | MeSH controlled vocabulary terms |
| `domain` | Domain label (drug_discovery / oncology) |

## Setup

```bash
git clone https://github.com/your-username/pubmed-cluster-explorer
cd pubmed-cluster-explorer
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
NCBI_API_KEY=your_key_here
```

Get a free NCBI API key at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/).

## Run data collection

```bash
python pubmed_fetcher.py
```

Outputs: `drug_discovery.csv` and `oncology.csv` (~20,000 records each).

> Runtime: ~45–60 min with an API key (10 req/sec rate limit).

## Tech stack

- **Data collection** — NCBI E-utilities API, `requests`
- **Embeddings** — `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
- **Clustering** — UMAP, HDBSCAN
- **App** — Gradio
- **Deployment** — Hugging Face Spaces

## Notes

- Raw CSV files are excluded from this repo (see `.gitignore`) — they will be published as a Hugging Face Dataset in a later phase
- The `.env` file is never committed — never share your API key publicly
