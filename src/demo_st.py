import importlib.util
import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.resolve().parent
SRC = ROOT / "src"
EXP = ROOT / "exp"
CORPUS_ANALYSIS = SRC / "data" / "corpus_analysis_new"
sys.path.append(str(SRC))
sys.path.append(str(EXP))
PREFIX_DATA = "batiment"


def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


saved_queries = import_from_path("saved_queries", SRC / "query" / "queries.py")
run_sparql_query = import_from_path(
    "run_sparql_query", SRC / "query" / "run_sparql_query.py"
)


@st.cache_resource
def load_kg_cached(kg_path):
    return run_sparql_query.load_kg(kg_path)


@st.cache_data
def load_parcourir_data():
    articles = pd.read_csv(CORPUS_ANALYSIS / f"{PREFIX_DATA}_uri.csv")
    triples = pd.read_csv(CORPUS_ANALYSIS / f"{PREFIX_DATA}_triples.csv")
    contents_path = CORPUS_ANALYSIS / f"{PREFIX_DATA}_articles_content.csv"
    contents = pd.read_csv(contents_path) if contents_path.exists() else pd.DataFrame()
    return articles.fillna(""), triples.fillna(""), contents.fillna("")


def short_uri(uri):
    value = str(uri or "")
    return re.split(r"[/#]", value)[-1] or value


def normalize_text(value):
    return str(value or "").casefold()


def filter_contains(df, columns, query):
    if not query:
        return pd.Series(True, index=df.index)
    query = normalize_text(query)
    haystack = df[columns].astype(str).agg(" ".join, axis=1).map(normalize_text)
    return haystack.str.contains(re.escape(query), na=False)


def entity_label(row, role):
    for column in (
        f"{role}_prefLabel",
        f"{role}_pref_label",
        f"{role}_label",
        f"{role}Label",
        f"{role}_rdfs_label",
    ):
        value = row.get(column, "")
        if value and not str(value).startswith("http"):
            return str(value)
    return short_uri(row.get(role, ""))


def build_entity_rows(triples):
    rows = []
    for _, row in triples.iterrows():
        predicate = short_uri(row.get("predicate_local") or row.get("predicate"))
        for role in ("subject", "object"):
            rows.append(
                {
                    "uri": row.get(role, ""),
                    "label": entity_label(row, role),
                    "class": row.get(f"{role}_class", "Unknown") or "Unknown",
                    "role": role,
                    "predicate": predicate,
                    "article": row.get("article_uri", ""),
                }
            )
    return pd.DataFrame(rows)


def make_graphviz(triples, article_uri, max_edges=35):
    rows = triples[triples["article_uri"] == article_uri].head(max_edges)
    lines = [
        "digraph G {",
        'graph [rankdir=LR, bgcolor="transparent", pad="0.2"];',
        'node [shape=ellipse, style="filled", fillcolor="#e9f4ef", color="#0e7c66", fontname="Arial"];',
        'edge [color="#9ca6ac", fontname="Arial", fontsize=10];',
    ]
    seen = set()
    for _, row in rows.iterrows():
        subject_id = f's{abs(hash(row["subject"]))}'
        object_id = f'o{abs(hash(row["object"]))}'
        subject_label = entity_label(row, "subject")[:42].replace('"', "'")
        object_label = entity_label(row, "object")[:42].replace('"', "'")
        predicate = short_uri(row.get("predicate_local") or row.get("predicate"))[:34]
        if subject_id not in seen:
            lines.append(f'"{subject_id}" [label="{subject_label}"];')
            seen.add(subject_id)
        if object_id not in seen:
            lines.append(
                f'"{object_id}" [label="{object_label}", fillcolor="#eef2fb", color="#4267ac"];'
            )
            seen.add(object_id)
        lines.append(f'"{subject_id}" -> "{object_id}" [label="{predicate}"];')
    lines.append("}")
    return "\n".join(lines)


def render_parcourir_page():
    st.header("Parcourir", icon=":material/account_tree:")
    st.caption(
        "Exploration des articles portes/portails et des triplets RDF extraits, "
        "d'apres la maquette dashboard_knowledge_graphs_by_article.html."
    )

    try:
        articles, triples, contents = load_parcourir_data()
    except FileNotFoundError as exc:
        st.error(f"Fichier CSV introuvable: {exc}", icon=":material/error:")
        return

    predicates = sorted(triples["predicate_local"].replace("", pd.NA).dropna().unique())
    if not predicates:
        predicates = sorted(triples["predicate"].map(short_uri).unique())

    with st.container(horizontal=True):
        st.metric("RDF triples in KG", "1,936,288", "Notebook load result", border=True)
        st.metric(
            "Matched articles", f"{len(articles):,}", f"{PREFIX_DATA}_uri.csv", border=True
        )
        st.metric(
            "Extracted triples",
            f"{len(triples):,}",
            f"{PREFIX_DATA}_triples.csv",
            border=True,
        )
        st.metric(
            "Predicates", f"{len(predicates):,}", "Distinct RDF predicates", border=True
        )

    triples = triples.copy()
    triples["article"] = triples["article_uri"].map(short_uri)
    triples["predicate_name"] = triples["predicate_local"].where(
        triples["predicate_local"].astype(bool), triples["predicate"].map(short_uri)
    )
    triples["subject_label"] = triples.apply(
        lambda row: entity_label(row, "subject"), axis=1
    )
    triples["object_label"] = triples.apply(
        lambda row: entity_label(row, "object"), axis=1
    )

    tab_triples, tab_classes, tab_articles = st.tabs(
        [
            ":material/table_chart: Triples",
            ":material/category: Description par classe",
            ":material/article: Entites par article",
        ]
    )

    with tab_triples:
        col_left, col_right = st.columns(2)
        with col_left.container(border=True, height="stretch"):
            st.subheader("Distribution des predicats")
            predicate_counts = (
                triples["predicate_name"]
                .value_counts()
                .head(15)
                .rename_axis("Predicat")
                .reset_index(name="Triples")
            )
            st.bar_chart(predicate_counts, x="Predicat", y="Triples", horizontal=True)
        with col_right.container(border=True, height="stretch"):
            st.subheader("Articles avec le plus de triples")
            article_counts = (
                triples["article"]
                .value_counts()
                .head(12)
                .rename_axis("Article")
                .reset_index(name="Triples")
            )
            st.bar_chart(article_counts, x="Article", y="Triples", horizontal=True)

        with st.container(border=True):
            st.subheader("Explorer les triples extraits")
            search_col, predicate_col = st.columns([2, 1])
            query = search_col.text_input(
                "Rechercher sujet/objet/article", key="triple_search"
            )
            selected_predicate = predicate_col.selectbox(
                "Predicat", ["Tous"] + predicates, key="predicate_filter"
            )
            visible = triples[
                filter_contains(
                    triples,
                    ["article", "subject_label", "object_label", "predicate_name"],
                    query,
                )
            ]
            if selected_predicate != "Tous":
                visible = visible[visible["predicate_name"] == selected_predicate]
            st.dataframe(
                visible[
                    [
                        "article",
                        "predicate_name",
                        "subject_label",
                        "subject_class",
                        "object_label",
                        "object_class",
                    ]
                ],
                hide_index=True,
                height=430,
            )
            st.caption(f"{len(visible):,} triples affiches.")

        with st.container(border=True):
            st.subheader("Sujets des articles trouves")
            article_view = articles.copy()
            article_view["article"] = article_view["article_uri"].map(short_uri)
            st.dataframe(
                article_view[
                    [
                        "article",
                        "dcterms_subject",
                        "matches",
                        "match_count",
                        "article_uri",
                    ]
                ],
                hide_index=True,
                height=330,
            )

    with tab_classes:
        entity_rows = build_entity_rows(triples)
        classes = sorted(entity_rows["class"].dropna().unique())
        class_col, mode_col, search_col = st.columns([1, 1, 2])
        selected_class = class_col.selectbox(
            "Classe", classes, index=classes.index("Actor") if "Actor" in classes else 0
        )
        mode = mode_col.selectbox("Grouper", ["Par entite", "Par article"])
        class_query = search_col.text_input(
            "Rechercher dans la classe", key="class_search"
        )
        scoped_entities = entity_rows[entity_rows["class"] == selected_class]
        scoped_entities = scoped_entities[
            filter_contains(
                scoped_entities, ["label", "uri", "predicate", "article"], class_query
            )
        ]

        col_left, col_right = st.columns([1.2, 0.8])
        with col_left.container(border=True, height="stretch"):
            if mode == "Par article":
                article_entities = (
                    scoped_entities.groupby("article")
                    .agg(
                        designations=(
                            "label",
                            lambda values: " | ".join(sorted(set(values))[:8]),
                        ),
                        triples=("predicate", "size"),
                        predicates=(
                            "predicate",
                            lambda values: ", ".join(sorted(set(values))[:8]),
                        ),
                    )
                    .reset_index()
                    .sort_values("triples", ascending=False)
                )
                st.subheader("Articles")
                st.dataframe(article_entities, hide_index=True, height=430)
            else:
                entity_summary = (
                    scoped_entities.groupby(["uri", "label", "class"])
                    .agg(
                        roles=("role", lambda values: " + ".join(sorted(set(values)))),
                        triples=("predicate", "size"),
                        predicates=(
                            "predicate",
                            lambda values: ", ".join(sorted(set(values))[:8]),
                        ),
                    )
                    .reset_index()
                    .sort_values("triples", ascending=False)
                )
                st.subheader("Entites")
                st.dataframe(
                    entity_summary[["label", "roles", "triples", "predicates", "uri"]],
                    hide_index=True,
                    height=430,
                )
        with col_right.container(border=True, height="stretch"):
            st.subheader("Predicats lies")
            predicate_scope = (
                scoped_entities["predicate"]
                .value_counts()
                .head(15)
                .rename_axis("Predicat")
                .reset_index(name="Triples")
            )
            st.bar_chart(predicate_scope, x="Predicat", y="Triples", horizontal=True)

    with tab_articles:
        article_summaries = (
            triples.groupby(["article_uri", "article"])
            .agg(
                entities=("subject_label", "count"),
                triples=("predicate_name", "size"),
                predicates=(
                    "predicate_name",
                    lambda values: ", ".join(sorted(set(values))[:10]),
                ),
            )
            .reset_index()
            .sort_values("triples", ascending=False)
        )
        article_query = st.text_input(
            "Rechercher article, entite ou predicat", key="article_search"
        )
        if article_query:
            matching_articles = triples[
                filter_contains(
                    triples,
                    ["article", "subject_label", "object_label", "predicate_name"],
                    article_query,
                )
            ]["article_uri"].unique()
            article_summaries = article_summaries[
                article_summaries["article_uri"].isin(matching_articles)
            ]

        if article_summaries.empty:
            st.warning("Aucun article ne correspond aux filtres.")
            return

        col_articles, col_graph = st.columns([0.95, 1.05])
        with col_articles.container(border=True, height="stretch"):
            st.subheader("Articles")
            selected_article = st.selectbox(
                "Article selectionne",
                article_summaries["article_uri"].tolist(),
                format_func=short_uri,
            )
            st.dataframe(
                article_summaries[["article", "entities", "triples", "predicates"]],
                hide_index=True,
                height=380,
            )
        with col_graph.container(border=True, height="stretch"):
            st.subheader("Graphe de l'article")
            st.graphviz_chart(make_graphviz(triples, selected_article))
            if len(triples[triples["article_uri"] == selected_article]) > 35:
                st.caption(
                    "Le graphe affiche les 35 premiers liens pour garder la lecture fluide."
                )

        with st.container(border=True):
            st.subheader("Texte de l'article")
            if not contents.empty:
                row = contents[contents["article_uri"] == selected_article]
                st.write(
                    row.iloc[0]["content"] if not row.empty else "Contenu non trouve."
                )
            else:
                st.caption(
                    f"Le fichier {PREFIX_DATA}_articles_content.csv n'est pas disponible."
                )


st.set_page_config(
    page_title="LegiMaintLex GUI",
    page_icon=":material/account_tree:",
    layout="wide",
)
st.title("LegiMaintLex - KG exploration")

page = st.sidebar.selectbox("Navigation", ["Parcourir", "Description", "GraphRAG"])

if page == "Description":
    st.header(
        "Description - Knowledge graph description", icon=":material/query_stats:"
    )
    KG_EXPORT_DIR_BASE = EXP / "kg" / "full"
    with st.sidebar:
        provider_choice = st.selectbox(
            "Choisir un provider", ["Mistral", "Openai"], index=0
        )
        KG_EXPORT_DIR = KG_EXPORT_DIR_BASE / provider_choice.lower()
        kg_files = sorted([p.name for p in KG_EXPORT_DIR.glob("*.ttl")])

        if not kg_files:
            st.warning(
                f"Aucun fichier .ttl trouve dans {KG_EXPORT_DIR}. Genere un KG d'abord."
            )
            kg_choice = None
            g = None
        else:
            default_index = min(2, len(kg_files) - 1)
            kg_choice = st.selectbox(
                "Choisir un KG (turtle)", kg_files, index=default_index
            )
            kg_path = KG_EXPORT_DIR / kg_choice
            with st.spinner("Chargement du Knowledge Graph..."):
                start_time = time.perf_counter()
                g = load_kg_cached(str(kg_path))
                elapsed = time.perf_counter() - start_time
            st.success(f"Knowledge Graph charge en {elapsed:.2f} s")
            st.write(f"Nombre de triples: {len(g)}")

    query_map = {
        name: getattr(saved_queries, name)
        for name in dir(saved_queries)
        if not name.startswith("_")
        and isinstance(getattr(saved_queries, name), str)
        and "\n" in getattr(saved_queries, name)
    }

    if not query_map:
        st.warning("Aucune query SPARQL definie dans src/query/queries.py")
    elif g is not None:
        st.write(f"Nombre de requetes SPARQL detectees: {len(query_map)}")
        with st.form("sparql_form"):
            selected_queries = st.multiselect(
                "Sparql queries:",
                options=list(query_map.keys()),
                format_func=lambda name: f"{name}: {len(query_map[name].splitlines())} lignes",
            )
            submitted = st.form_submit_button("Executer les requetes selectionnees")

        if submitted:
            if not selected_queries:
                st.warning("Veuillez selectionner au moins une requete SPARQL.")
            else:
                with st.spinner(
                    f"Execution des {len(selected_queries)} requetes SPARQL..."
                ):
                    summary_rows = []
                    full_results = {}
                    start_time = time.perf_counter()
                    for qname in selected_queries:
                        query_text = query_map[qname]
                        try:
                            rows, vars_ = run_sparql_query.run_sparql_query(
                                g, query_text
                            )
                            summary_rows.append(
                                {
                                    "query_name": qname,
                                    "nb_results": len(rows),
                                    "variables": ", ".join(vars_) if vars_ else None,
                                    "status": "OK",
                                }
                            )
                            full_results[qname] = {"variables": vars_, "rows": rows}
                        except Exception as exc:
                            summary_rows.append(
                                {
                                    "query_name": qname,
                                    "nb_results": None,
                                    "variables": None,
                                    "status": f"ERROR: {exc}",
                                }
                            )
                            full_results[qname] = {
                                "variables": None,
                                "rows": None,
                                "status": str(exc),
                            }

                        elapsed = time.perf_counter() - start_time
                        st.caption(f"Query ({qname}) en {elapsed:.2f} s")

                    st.caption(f"Temps total {time.perf_counter() - start_time:.2f} s")
                    df_summary = pd.DataFrame(summary_rows)
                    st.success("Resume de description genere")
                    st.dataframe(df_summary, hide_index=True)

                    csv_bytes = df_summary.to_csv(index=False, encoding="utf-8").encode(
                        "utf-8"
                    )
                    st.download_button(
                        "Telecharger le resume (CSV)",
                        data=csv_bytes,
                        file_name=f"{kg_choice}_description_summary.csv",
                        mime="text/csv",
                    )

                    st.write("Resultats complets des requetes SPARQL:")
                    for qname, results in full_results.items():
                        st.subheader(f"Resultats pour la requete: {qname}")
                        if results["rows"] is not None:
                            st.dataframe(pd.DataFrame(results["rows"]), hide_index=True)
                        else:
                            st.warning(
                                f"Erreur lors de l'execution de la requete: {results['status']}"
                            )

elif page == "Parcourir":
    render_parcourir_page()

elif page == "GraphRAG":
    st.header("GraphRAG", icon=":material/chat:")
    st.write("Cette section est en cours de developpement.")
