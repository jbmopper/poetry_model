import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PAGE_SIZE = 25

DEFAULT_DATA_DIR = "/mnt/raid/datas/output_v4_modern"


def resolve_data_dir() -> Path | None:
    """Resolve data directory from CLI args, env var, session state, or default."""
    # CLI: uv run streamlit run viewer.py -- /path/to/data
    cli_args = sys.argv[1:]
    for arg in cli_args:
        p = Path(arg)
        if p.is_dir():
            return p

    if "data_dir" in st.session_state and st.session_state.data_dir:
        p = Path(st.session_state.data_dir)
        if p.is_dir():
            return p

    if env := os.environ.get("POETRY_DATA_DIR"):
        p = Path(env)
        if p.is_dir():
            return p

    p = Path(DEFAULT_DATA_DIR)
    if p.is_dir():
        return p

    return None


@st.cache_data
def load_data(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            meta = r.get("metadata", {})
            records.append(
                {
                    "id": r["id"],
                    "title": r.get("title") or "(untitled)",
                    "author": r.get("author") or "",
                    "domain": r["domain"],
                    "source_type": r.get("source_type", ""),
                    "confidence": r["confidence"],
                    "line_count": meta.get("line_count", 0),
                    "word_count": meta.get("word_count", 0),
                    "char_count": meta.get("char_count", 0),
                    "url": r.get("url", ""),
                    "text": r["text"],
                }
            )
    df = pd.DataFrame(records)
    df["confidence"] = df["confidence"].round(3)
    return df


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    domains = st.sidebar.multiselect(
        "Source",
        options=sorted(df["domain"].unique()),
        default=sorted(df["domain"].unique()),
    )

    source_types = st.sidebar.multiselect(
        "Source type",
        options=sorted(df["source_type"].unique()),
        default=sorted(df["source_type"].unique()),
    )

    conf_min, conf_max = float(df["confidence"].min()), float(df["confidence"].max())
    conf_range = st.sidebar.slider(
        "Confidence range",
        min_value=conf_min,
        max_value=conf_max,
        value=(conf_min, conf_max),
        step=0.01,
    )

    word_min, word_max = int(df["word_count"].min()), int(df["word_count"].max())
    word_range = st.sidebar.slider(
        "Word count range",
        min_value=word_min,
        max_value=word_max,
        value=(word_min, word_max),
    )

    titled_only = st.sidebar.checkbox("Has title only", value=False)

    mask = (
        df["domain"].isin(domains)
        & df["source_type"].isin(source_types)
        & df["confidence"].between(*conf_range)
        & df["word_count"].between(*word_range)
    )
    if titled_only:
        mask = mask & (df["title"] != "(untitled)")

    return df[mask]


def page_browse(df: pd.DataFrame, filtered: pd.DataFrame):
    st.header("Browse Poems")
    st.caption(f"{len(filtered):,} poems match current filters")

    col_sort, col_dir = st.columns([2, 1])
    with col_sort:
        sort_by = st.selectbox(
            "Sort by",
            ["title", "confidence", "word_count", "domain", "line_count"],
        )
    with col_dir:
        ascending = st.selectbox("Order", ["Ascending", "Descending"]) == "Ascending"

    sorted_df = filtered.sort_values(sort_by, ascending=ascending).reset_index(
        drop=True
    )

    total_pages = max(1, math.ceil(len(sorted_df) / PAGE_SIZE))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page - 1) * PAGE_SIZE
    page_df = sorted_df.iloc[start : start + PAGE_SIZE]

    st.caption(f"Page {page} of {total_pages}")

    display_cols = [
        "title",
        "author",
        "domain",
        "source_type",
        "confidence",
        "word_count",
        "line_count",
    ]
    st.dataframe(
        page_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Read a poem")

    poem_options = {
        f"{row['title']}  [{row['domain']}]  (conf={row['confidence']})": row["id"]
        for _, row in page_df.iterrows()
    }
    if poem_options:
        selected_label = st.selectbox("Select from this page", list(poem_options.keys()))
        if selected_label:
            render_poem_detail(df, poem_options[selected_label])


def render_poem_detail(df: pd.DataFrame, poem_id: str):
    row = df[df["id"] == poem_id].iloc[0]

    st.markdown(f"### {row['title']}")
    if row["author"]:
        st.markdown(f"*by {row['author']}*")

    meta_cols = st.columns(4)
    meta_cols[0].metric("Confidence", f"{row['confidence']:.3f}")
    meta_cols[1].metric("Words", f"{row['word_count']:,}")
    meta_cols[2].metric("Lines", row["line_count"])
    meta_cols[3].metric("Source", row["source_type"])

    st.caption(f"**Domain:** {row['domain']}")
    if row["url"]:
        st.caption(f"**URL:** [{row['url']}]({row['url']})")

    st.divider()
    st.text(row["text"])


def page_detail(df: pd.DataFrame, filtered: pd.DataFrame):
    st.header("Poem Detail")

    poem_id = st.text_input("Enter poem ID")
    if poem_id and poem_id in df["id"].values:
        render_poem_detail(df, poem_id)
    elif poem_id:
        st.warning("Poem ID not found.")


def page_search(df: pd.DataFrame, filtered: pd.DataFrame):
    st.header("Search")

    query = st.text_input("Search titles and text")
    if not query:
        st.info("Enter a search term to find poems.")
        return

    query_lower = query.lower()
    mask = filtered["title"].str.lower().str.contains(query_lower, na=False) | filtered[
        "text"
    ].str.lower().str.contains(query_lower, na=False)
    results = filtered[mask]
    st.caption(f"{len(results):,} results for **{query}**")

    if results.empty:
        st.warning("No results found.")
        return

    total_pages = max(1, math.ceil(len(results) / PAGE_SIZE))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page - 1) * PAGE_SIZE
    page_results = results.iloc[start : start + PAGE_SIZE]

    for _, row in page_results.iterrows():
        with st.expander(
            f"**{row['title']}** — {row['domain']} (conf={row['confidence']:.2f}, {row['word_count']} words)"
        ):
            if row["author"]:
                st.markdown(f"*by {row['author']}*")

            text = row["text"]
            # Highlight context around match
            idx = text.lower().find(query_lower)
            if idx >= 0:
                context_start = max(0, idx - 200)
                context_end = min(len(text), idx + len(query) + 200)
                snippet = text[context_start:context_end]
                if context_start > 0:
                    snippet = "..." + snippet
                if context_end < len(text):
                    snippet = snippet + "..."
                st.text(snippet)
            else:
                st.text(text[:500] + ("..." if len(text) > 500 else ""))

            if st.button("Full text", key=f"full_{row['id']}"):
                st.text(text)


def page_stats(df: pd.DataFrame, filtered: pd.DataFrame):
    st.header("Dataset Statistics")
    st.caption(f"Stats for {len(filtered):,} poems matching filters")

    # Top-level metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total poems", f"{len(filtered):,}")
    m2.metric("With titles", f"{(filtered['title'] != '(untitled)').sum():,}")
    m3.metric("With authors", f"{(filtered['author'] != '').sum():,}")
    m4.metric("Avg confidence", f"{filtered['confidence'].mean():.3f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Poems by source")
        domain_counts = filtered["domain"].value_counts().reset_index()
        domain_counts.columns = ["domain", "count"]
        fig = px.bar(
            domain_counts,
            x="count",
            y="domain",
            orientation="h",
            color="domain",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Source type")
        type_counts = filtered["source_type"].value_counts().reset_index()
        type_counts.columns = ["source_type", "count"]
        fig = px.pie(
            type_counts,
            values="count",
            names="source_type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Confidence distribution")
        fig = px.histogram(
            filtered,
            x="confidence",
            nbins=30,
            color_discrete_sequence=["#6C8EBF"],
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Word count distribution")
        # Cap at 99th percentile for readability
        cap = filtered["word_count"].quantile(0.99)
        fig = px.histogram(
            filtered[filtered["word_count"] <= cap],
            x="word_count",
            nbins=40,
            color_discrete_sequence=["#82B366"],
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Confidence vs. word count")
    sample = filtered.sample(min(500, len(filtered)), random_state=42)
    fig = px.scatter(
        sample,
        x="word_count",
        y="confidence",
        color="domain",
        hover_data=["title"],
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Per-source statistics")
    stats = (
        filtered.groupby("domain")
        .agg(
            count=("id", "size"),
            avg_confidence=("confidence", "mean"),
            avg_words=("word_count", "mean"),
            avg_lines=("line_count", "mean"),
            titled=("title", lambda s: (s != "(untitled)").sum()),
        )
        .round(2)
        .sort_values("count", ascending=False)
    )
    st.dataframe(stats, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Poetry Corpus Viewer",
        page_icon="📖",
        layout="wide",
    )

    st.title("Poetry Corpus Viewer")

    data_dir = resolve_data_dir()

    if data_dir is None:
        st.warning("Could not find the poetry dataset automatically.")
        st.text_input(
            "Enter the path to the dataset directory (the one containing corpus/):",
            key="data_dir",
            placeholder="/path/to/output_v4_modern",
        )
        st.info(
            "You can also pass the path as a CLI argument:\n\n"
            "    uv run streamlit run viewer.py -- /path/to/output_v4_modern"
        )
        return

    corpus_jsonl = data_dir / "corpus" / "poetry_corpus.jsonl"
    if not corpus_jsonl.exists():
        st.error(
            f"Found directory `{data_dir}` but missing expected file:\n\n"
            f"    {corpus_jsonl}\n\n"
            "Make sure the directory contains `corpus/poetry_corpus.jsonl`."
        )
        st.text_input(
            "Enter a different path:",
            key="data_dir",
            placeholder="/path/to/output_v4_modern",
        )
        return

    st.caption(f"Dataset: `{data_dir}`")

    df = load_data(str(corpus_jsonl))
    filtered = render_sidebar_filters(df)

    tab_browse, tab_search, tab_stats = st.tabs(["Browse", "Search", "Stats"])

    with tab_browse:
        page_browse(df, filtered)
    with tab_search:
        page_search(df, filtered)
    with tab_stats:
        page_stats(df, filtered)


if __name__ == "__main__":
    main()
