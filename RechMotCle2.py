import streamlit as st
from rdflib import Graph, URIRef, Literal, RDF, RDFS
from datasketch import MinHash
import itertools
import os
import tempfile
from pyvis.network import Network
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Chargement RDF
@st.cache_resource
def load_graph():
    g = Graph()
    g.parse("./test-rdf.ttl", format="ttl")
    return g


g = load_graph()

# Prétraitement : KMV Dp, Rp, Sc
KMV_D, KMV_R, KMV_C = {}, {}, {}


def update_kmv(store, key, value):
    if key not in store:
        store[key] = MinHash(num_perm=128)
    store[key].update(str(value).encode("utf-8"))


for s, p, o in g:
    update_kmv(KMV_D, p, s)
    if isinstance(o, URIRef):
        update_kmv(KMV_R, p, o)
    if p == RDF.type and isinstance(o, URIRef):
        update_kmv(KMV_C, o, s)


def search_keywords(keywords):
    results = set()
    for s, p, o in g:
        for k in keywords:
            if isinstance(o, Literal) and k.lower() in str(o).lower():
                results.add((s, p, o, k))
            elif p == RDFS.label and k.lower() in str(o).lower():
                results.add((s, "rdfs:label", o, k))
    return list(results)


def jaccard_similarity(mh1, mh2):
    return mh1.jaccard(mh2)


def fuse_nodes(matches, threshold=0.5):
    groups = []
    seen_keywords = set()
    for (s1, p1, _, k1), (s2, p2, _, k2) in itertools.combinations(matches, 2):
        if p1 in KMV_D and p2 in KMV_D:
            sim = jaccard_similarity(KMV_D[p1], KMV_D[p2])
            if sim >= threshold:
                groups.append({k1, k2})
                seen_keywords |= {k1, k2}
    for m in matches:
        if m[3] not in seen_keywords:
            groups.append({m[3]})
    return groups


def containment_score(mh_a, mh_b):
    return len(set(mh_a.digest()).intersection(mh_b.digest())) / len(mh_a.digest())


def add_edges(groups, matches, threshold=0.3):
    edges = set()
    for g1, g2 in itertools.combinations(groups, 2):
        for m1 in matches:
            for m2 in matches:
                if m1[3] in g1 and m2[3] in g2:
                    p, q = m1[1], m2[1]
                    if p in KMV_D and q in KMV_R:
                        score = containment_score(KMV_D[p], KMV_R[q])
                        if score >= threshold:
                            edges.add((str(m1[0]), str(p), str(m2[0])))
    return list(edges)


def tree_expansion(nodes, existing_edges, threshold=0.3):
    added = set()
    for s, p, o in g:
        if str(s) in nodes or str(o) in nodes:
            if p in KMV_D:
                for other in KMV_D:
                    if containment_score(KMV_D[other], KMV_D[p]) >= threshold:
                        added.add((str(s), str(p), str(o)))
    return list(added - set(existing_edges))


def generate_sparql(matches, edges):
    vars_map = {}
    counter = 1
    where = set()

    def get_var(x):
        nonlocal counter
        if str(x) not in vars_map:
            vars_map[str(x)] = f"?v{counter}"
            counter += 1
        return vars_map[str(x)]

    for s, p, o, k in matches:
        vs = get_var(s)
        vo = get_var(o)
        if p == "rdfs:label":
            where.add(
                f'{vs} rdfs:label {vo} FILTER CONTAINS(LCASE(STR({vo})), "{k.lower()}") .'
            )
        else:
            where.add(
                f'{vs} <{p}> {vo} FILTER CONTAINS(LCASE(STR({vo})), "{k.lower()}") .'
            )

    for s, p, o in edges:
        vs = get_var(s)
        vo = get_var(o)
        where.add(f"{vs} <{p}> {vo} .")

    return "SELECT DISTINCT * WHERE {\n  " + "\n  ".join(sorted(where)) + "\n}"


def info_rank(graph):
    scores = {}
    for node in set(graph.subjects()) | set(graph.objects()):
        literals = [
            o for _, _, o in graph.triples((node, None, None)) if isinstance(o, Literal)
        ]
        scores[str(node)] = len(literals)
    return sorted(scores.items(), key=lambda x: -x[1])


def execute(query):
    results = g.query(query)
    entities = []
    for r in results:
        line = [str(x) for x in r]
        entities.extend(line)
    return list(set(entities))


def export_interactive_graph(graph, result_entities=None, filename="graph.html"):
    net = Network(height="600px", width="100%", notebook=False, directed=True)
    highlighted = set(result_entities or [])

    for s, p, o in graph:
        s_str, o_str = str(s), str(o)
        p_str = str(p).split("/")[-1]

        for node in [s_str, o_str]:
            label = node.split("/")[-1]
            color = "orange" if node in highlighted else "#97c2fc"
            net.add_node(node, label=label, title=node, color=color)

        net.add_edge(s_str, o_str, label=p_str)

    path = os.path.join(tempfile.gettempdir(), filename)
    net.save_graph(path)

    return path


# --- Streamlit UI ---
for key in ["matches", "groups", "edges", "query", "result_entities"]:
    if key not in st.session_state:
        st.session_state[key] = []

st.title("🔎 RDF Keyword Search & SPARQL Generator")

user_input = st.text_input("Entrez vos mots-clés (séparés par espace):")

# --- Bouton de recherche
if st.button("Lancer la recherche") and user_input:
    keywords = user_input.split()
    st.session_state.matches = search_keywords(keywords)

    if not st.session_state.matches:
        st.warning("❌ Aucun mot-clé trouvé.")
    else:
        st.success(f"✅ {len(st.session_state.matches)} correspondances trouvées.")
        st.write("🎯 Triplets correspondants :", st.session_state.matches)

        st.session_state.groups = fuse_nodes(st.session_state.matches)
        st.write("🧩 Groupes fusionnés :", st.session_state.groups)

        st.session_state.edges = add_edges(
            st.session_state.groups, st.session_state.matches
        )

        if not st.session_state.edges:
            nodes = {str(m[0]) for m in st.session_state.matches}
            st.session_state.edges = tree_expansion(nodes, [])
            st.info("🌲 Expansion d’arbre effectuée (fallback).")

        st.session_state.query = generate_sparql(
            st.session_state.matches, st.session_state.edges
        )
        st.code(st.session_state.query, language="sparql")

# --- Affichage des boutons seulement si des correspondances existent
if st.session_state.matches:
    if st.button("Exécuter la requête SPARQL"):
        if st.session_state.query:
            st.session_state.result_entities = execute(st.session_state.query)
            if st.session_state.result_entities:
                st.success("🎯 Entités trouvées :")
                st.write(st.session_state.result_entities)
                ranked = info_rank(g)
                st.subheader("📊 Classement InfoRank des entités :")
                for ent, score in ranked:
                    if ent in st.session_state.result_entities:
                        st.write(f"→ {ent} : {score}")
            else:
                st.warning("Aucun résultat SPARQL.")
        else:
            st.warning("Veuillez d'abord lancer une recherche.")

    if st.button("📤 Exporter graphe interactif"):
        if st.session_state.result_entities:
            path = export_interactive_graph(g, st.session_state.result_entities)

            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)
            st.download_button(
                label="📂 Télécharger le graphe",
                data=html_content,
                file_name="graph.html",
                mime="text/html",
            )
        else:
            st.warning("Aucun résultat à exporter.")
