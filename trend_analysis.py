"""
Phase 3 — Trend Analysis
Reads processed metadata CSVs and produces interactive Plotly charts:
  1. Publication volume over time (animated bar race)
  2. MeSH term heatmap (year x term)
  3. Top-10 keyword frequency bar chart per year
  4. Domain split comparison
"""

import re
import ast
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict, Counter

# ── config ────────────────────────────────────────────────────────────────────

DATA_FILES = {
    "drug_discovery": "data/processed/drug_discovery_metadata.csv",
    "oncology":       "data/processed/oncology_metadata.csv",
}

OUTPUT_DIR = Path("charts")
OUTPUT_DIR.mkdir(exist_ok=True)

# Only keep articles from this year range
YEAR_MIN = 2000
YEAR_MAX = 2024

# Top N MeSH terms and keywords to show in charts
TOP_N_MESH     = 20
TOP_N_KEYWORDS = 10


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_year(pub_date: str) -> int | None:
    """Pull the 4-digit year out of any pub_date string."""
    match = re.search(r"(19|20)\d{2}", str(pub_date))
    return int(match.group()) if match else None


def split_terms(cell: str) -> list:
    """Split a semicolon-separated terms cell into a clean list."""
    if not cell or pd.isna(cell):
        return []
    return [t.strip() for t in str(cell).split(";") if t.strip()]


def load_and_prepare() -> pd.DataFrame:
    frames = []
    for domain, filepath in DATA_FILES.items():
        if not Path(filepath).exists():
            print(f"  Warning: {filepath} not found — skipping {domain}")
            continue
        df = pd.read_csv(filepath)
        df["domain"] = domain
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No metadata CSVs found. Run process_abstracts.py first.")

    df = pd.concat(frames, ignore_index=True)

    # Parse year
    df["year"] = df["pub_date"].apply(extract_year)
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].copy()

    # Parse list columns
    df["mesh_list"]    = df["mesh_terms"].apply(split_terms)
    df["keyword_list"] = df["keywords"].apply(split_terms)

    print(f"  Loaded {len(df):,} articles ({YEAR_MIN}–{YEAR_MAX})")
    return df


# ── chart 1: publication volume over time ────────────────────────────────────

def chart_volume_over_time(df: pd.DataFrame):
    print("  Building chart 1: publication volume over time...")

    vol = (
        df.groupby(["year", "domain"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        vol,
        x="year",
        y="count",
        color="domain",
        barmode="group",
        labels={"year": "Year", "count": "Articles published", "domain": "Domain"},
        title="Publication volume over time — drug discovery vs oncology",
        color_discrete_map={
            "drug_discovery": "#378ADD",
            "oncology":       "#D85A30",
        },
        template="plotly_white",
    )
    fig.update_layout(
        legend_title_text="Domain",
        font_family="Arial",
        title_font_size=16,
        hovermode="x unified",
    )
    fig.write_html(OUTPUT_DIR / "1_volume_over_time.html")
    print("    Saved → charts/1_volume_over_time.html")


# ── chart 2: animated bar race (cumulative volume) ───────────────────────────

def chart_bar_race(df: pd.DataFrame):
    print("  Building chart 2: animated bar race...")

    vol = (
        df.groupby(["year", "domain"])
        .size()
        .reset_index(name="count")
        .sort_values("year")
    )

    # Cumulative sum per domain
    vol["cumulative"] = vol.groupby("domain")["count"].cumsum()

    fig = px.bar(
        vol,
        x="cumulative",
        y="domain",
        color="domain",
        animation_frame="year",
        orientation="h",
        range_x=[0, vol["cumulative"].max() * 1.1],
        labels={"cumulative": "Total articles", "domain": "Domain"},
        title="Cumulative publication growth — animated",
        color_discrete_map={
            "drug_discovery": "#378ADD",
            "oncology":       "#D85A30",
        },
        template="plotly_white",
    )
    fig.update_layout(
        font_family="Arial",
        title_font_size=16,
        showlegend=False,
    )
    fig.write_html(OUTPUT_DIR / "2_bar_race.html")
    print("    Saved → charts/2_bar_race.html")


# ── chart 3: MeSH term heatmap (year × term) ─────────────────────────────────

def chart_mesh_heatmap(df: pd.DataFrame, domain: str):
    print(f"  Building chart 3: MeSH heatmap — {domain}...")

    sub = df[df["domain"] == domain].copy()

    # Explode mesh terms
    exploded = sub[["year", "mesh_list"]].explode("mesh_list")
    exploded = exploded.rename(columns={"mesh_list": "mesh_term"})
    exploded = exploded[exploded["mesh_term"].notna() & (exploded["mesh_term"] != "")]

    # Find top N overall MeSH terms
    top_terms = (
        exploded["mesh_term"]
        .value_counts()
        .head(TOP_N_MESH)
        .index.tolist()
    )

    # Pivot: year x term
    pivot = (
        exploded[exploded["mesh_term"].isin(top_terms)]
        .groupby(["year", "mesh_term"])
        .size()
        .reset_index(name="count")
        .pivot(index="mesh_term", columns="year", values="count")
        .fillna(0)
    )

    fig = px.imshow(
        pivot,
        labels={"x": "Year", "y": "MeSH term", "color": "Article count"},
        title=f"MeSH term frequency heatmap — {domain.replace('_', ' ').title()}",
        color_continuous_scale="Blues",
        aspect="auto",
        template="plotly_white",
    )
    fig.update_layout(
        font_family="Arial",
        title_font_size=16,
        xaxis_nticks=10,
    )
    fig.write_html(OUTPUT_DIR / f"3_mesh_heatmap_{domain}.html")
    print(f"    Saved → charts/3_mesh_heatmap_{domain}.html")


# ── chart 4: top-10 keywords per year (animated) ─────────────────────────────

def chart_top_keywords(df: pd.DataFrame, domain: str):
    print(f"  Building chart 4: top keywords per year — {domain}...")

    sub = df[df["domain"] == domain].copy()

    # Explode keywords
    exploded = sub[["year", "keyword_list"]].explode("keyword_list")
    exploded = exploded.rename(columns={"keyword_list": "keyword"})
    exploded = exploded[exploded["keyword"].notna() & (exploded["keyword"] != "")]
    exploded["keyword"] = exploded["keyword"].str.lower().str.strip()

    # Count per year
    counts = (
        exploded.groupby(["year", "keyword"])
        .size()
        .reset_index(name="count")
    )

    # Keep top N keywords per year (sort + head avoids pandas apply dropping "year")
    top_per_year = (
        counts.sort_values(["year", "count"], ascending=[True, False])
        .groupby("year", group_keys=False)
        .head(TOP_N_KEYWORDS)
    )

    fig = px.bar(
        top_per_year,
        x="count",
        y="keyword",
        animation_frame="year",
        orientation="h",
        range_x=[0, top_per_year["count"].max() * 1.1],
        labels={"count": "Articles", "keyword": "Keyword"},
        title=f"Top {TOP_N_KEYWORDS} keywords per year — {domain.replace('_', ' ').title()}",
        template="plotly_white",
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        font_family="Arial",
        title_font_size=16,
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        coloraxis_showscale=False,
    )
    fig.write_html(OUTPUT_DIR / f"4_top_keywords_{domain}.html")
    print(f"    Saved → charts/4_top_keywords_{domain}.html")


# ── chart 5: domain split over time (area chart) ─────────────────────────────

def chart_domain_split(df: pd.DataFrame):
    print("  Building chart 5: domain split comparison...")

    vol = (
        df.groupby(["year", "domain"])
        .size()
        .reset_index(name="count")
    )

    # Compute share %
    total_per_year = vol.groupby("year")["count"].transform("sum")
    vol["share"] = (vol["count"] / total_per_year * 100).round(1)

    fig = px.area(
        vol,
        x="year",
        y="share",
        color="domain",
        labels={"year": "Year", "share": "Publication share (%)", "domain": "Domain"},
        title="Domain publication share over time (%)",
        color_discrete_map={
            "drug_discovery": "#378ADD",
            "oncology":       "#D85A30",
        },
        template="plotly_white",
    )
    fig.update_layout(
        font_family="Arial",
        title_font_size=16,
        hovermode="x unified",
        legend_title_text="Domain",
    )
    fig.write_html(OUTPUT_DIR / "5_domain_split.html")
    print("    Saved → charts/5_domain_split.html")


# ── chart 6: specific term spotlight (story angles) ──────────────────────────

def chart_term_spotlight(df: pd.DataFrame):
    """
    Track specific high-interest terms over time.
    Story angles: mRNA spike post-2020, CAR-T growth, checkpoint inhibitors.
    """
    print("  Building chart 6: term spotlight...")

    SPOTLIGHT_TERMS = {
        "mRNA":                  r"mrna",
        "CAR-T":                 r"car-t|cart|chimeric antigen",
        "Checkpoint inhibitors": r"checkpoint inhibitor|pd-1|pd-l1|ctla",
        "Immunotherapy":         r"immunotherapy",
        "Lead optimization":     r"lead optimization",
        "Small molecule":        r"small molecule",
    }

    sub = df.copy()
    sub["text"] = (
        sub["title"].fillna("") + " " +
        sub["mesh_terms"].fillna("") + " " +
        sub["keywords"].fillna("")
    ).str.lower()

    rows = []
    for label, pattern in SPOTLIGHT_TERMS.items():
        matched = sub[sub["text"].str.contains(pattern, regex=True, na=False)]
        yearly  = matched.groupby("year").size().reset_index(name="count")
        yearly["term"] = label
        rows.append(yearly)

    spotlight = pd.concat(rows, ignore_index=True)

    fig = px.line(
        spotlight,
        x="year",
        y="count",
        color="term",
        markers=True,
        labels={"year": "Year", "count": "Articles mentioning term", "term": "Term"},
        title="Key term spotlight — publication volume over time",
        template="plotly_white",
    )
    fig.update_layout(
        font_family="Arial",
        title_font_size=16,
        hovermode="x unified",
        legend_title_text="Term",
    )
    fig.write_html(OUTPUT_DIR / "6_term_spotlight.html")
    print("    Saved → charts/6_term_spotlight.html")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Phase 3 — Trend analysis\n")

    df = load_and_prepare()
    print()

    chart_volume_over_time(df)
    chart_bar_race(df)

    for domain in ["drug_discovery", "oncology"]:
        if domain in df["domain"].unique():
            chart_mesh_heatmap(df, domain)
            chart_top_keywords(df, domain)

    chart_domain_split(df)
    chart_term_spotlight(df)

    print(f"\nAll charts saved to ./charts/")
    print("Open any .html file in your browser to explore interactively.")


if __name__ == "__main__":
    main()
