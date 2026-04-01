import os
import time
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY           = os.getenv("NCBI_API_KEY")
BASE_URL          = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE        = 200      # PMIDs per efetch call (NCBI recommended max)
SEARCH_PAGE       = 500      # PMIDs per esearch page
TARGET_PER_DOMAIN = 20_000
RATE_DELAY        = 0.11     # ~9 req/sec — safe with API key (limit is 10)

DOMAINS = {
    "drug_discovery": [
        "drug discovery",
        "small molecule",
        "target identification",
        "lead optimization",
    ],
    "oncology": [
        "oncology",
        "immunotherapy",
        "CAR-T",
        "tumor microenvironment",
        "mRNA cancer vaccine",
    ],
}

CSV_COLUMNS = ["pmid", "title", "abstract", "pub_date", "keywords", "mesh_terms", "domain"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_json(endpoint: str, params: dict) -> dict:
    params["api_key"] = API_KEY
    time.sleep(RATE_DELAY)
    r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _get_xml(endpoint: str, params: dict) -> str:
    params["api_key"] = API_KEY
    time.sleep(RATE_DELAY)
    r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
    r.raise_for_status()
    return r.text


def esearch_all(term: str, max_results: int) -> list:
    """Paginate esearch to collect up to max_results PMIDs for a term."""
    pmids = []
    retstart = 0

    params = {
        "db": "pubmed", "term": term, "retmax": SEARCH_PAGE,
        "retstart": 0, "sort": "relevance", "retmode": "json",
    }
    data  = _get_json("esearch.fcgi", params)
    total = int(data["esearchresult"].get("count", 0))
    pmids += data["esearchresult"]["idlist"]
    retstart += SEARCH_PAGE

    while len(pmids) < min(max_results, total):
        params["retstart"] = retstart
        chunk = _get_json("esearch.fcgi", params)["esearchresult"]["idlist"]
        if not chunk:
            break
        pmids += chunk
        retstart += SEARCH_PAGE

    return pmids[:max_results]


def efetch_batch(pmids: list) -> str:
    return _get_xml("efetch.fcgi", {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
    })


def _txt(element, path: str) -> str:
    el = element.find(path)
    return el.text.strip() if el is not None and el.text else ""


def parse_xml(xml_text: str, domain: str) -> list:
    root = ET.fromstring(xml_text)
    records = []

    for art in root.findall(".//PubmedArticle"):
        pmid     = _txt(art, ".//PMID")
        title    = _txt(art, ".//ArticleTitle")
        abstract = " ".join(
            el.text.strip()
            for el in art.findall(".//AbstractText")
            if el.text
        )

        year  = _txt(art, ".//PubDate/Year")
        month = _txt(art, ".//PubDate/Month")
        med   = _txt(art, ".//MedlineDate")
        pub_date = med if med else (f"{year}-{month}" if month else year)

        keywords = "; ".join(
            el.text.strip()
            for el in art.findall(".//KeywordList/Keyword")
            if el.text
        )
        mesh_terms = "; ".join(
            el.text.strip()
            for el in art.findall(".//MeshHeading/DescriptorName")
            if el.text
        )

        if pmid and title:
            records.append({
                "pmid":       pmid,
                "title":      title,
                "abstract":   abstract,
                "pub_date":   pub_date,
                "keywords":   keywords,
                "mesh_terms": mesh_terms,
                "domain":     domain,
            })

    return records


def save_csv(records: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


# ── per-domain fetch ──────────────────────────────────────────────────────────

def fetch_domain(domain: str, keywords: list) -> list:
    seen_pmids = set()
    all_records = []
    per_keyword = TARGET_PER_DOMAIN // len(keywords)

    for kw in keywords:
        print(f"\n  [{domain}] keyword: '{kw}'")
        print(f"    Searching up to {per_keyword:,} PMIDs...")

        pmids     = esearch_all(kw, per_keyword)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        print(f"    Found {len(pmids):,} · {len(new_pmids):,} new after dedup")
        seen_pmids.update(new_pmids)

        fetched = 0
        for i in range(0, len(new_pmids), BATCH_SIZE):
            batch = new_pmids[i : i + BATCH_SIZE]
            try:
                records = parse_xml(efetch_batch(batch), domain)
                all_records.extend(records)
                fetched += len(records)
                print(
                    f"    Batch {i // BATCH_SIZE + 1}: "
                    f"+{len(records)} · domain total: {len(all_records):,}",
                    end="\r",
                )
            except Exception as e:
                print(f"\n    Warning — batch {i // BATCH_SIZE + 1} failed: {e}")

        print(f"\n    Done '{kw}': {fetched:,} records")

        if len(all_records) >= TARGET_PER_DOMAIN:
            print(f"  Target of {TARGET_PER_DOMAIN:,} reached.")
            break

    return all_records


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        raise ValueError("NCBI_API_KEY not found — check your .env file.")

    print(f"PubMed bulk fetcher  |  target: {TARGET_PER_DOMAIN:,} records/domain\n")

    for domain, keywords in DOMAINS.items():
        print("=" * 52)
        print(f"  Domain: {domain}")
        print("=" * 52)

        records  = fetch_domain(domain, keywords)
        out_path = Path(f"{domain}.csv")
        save_csv(records, out_path)
        print(f"\n  Saved {len(records):,} records → {out_path}\n")

    print("All done.")


if __name__ == "__main__":
    main()
