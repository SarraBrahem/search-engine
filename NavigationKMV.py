from concurrent.futures import ThreadPoolExecutor
from rdflib import Graph, URIRef, Literal, RDF, RDFS
from datasketch import MinHash
from collections import defaultdict
import threading
import itertools
import streamlit as st
from pyvis.network import Network
import os
import tempfile
import networkx as nx

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")  # léger, rapide, efficace

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Chargement RDF ---
g = Graph()
g.parse("./test-rdf.ttl", format="ttl")

# --- Prétraitement parallèle : KMV Dp, Rp, Sc ---
KMV_D = defaultdict(lambda: MinHash(num_perm=128))
KMV_R = defaultdict(lambda: MinHash(num_perm=128))
KMV_C = defaultdict(lambda: MinHash(num_perm=128))
lock = threading.Lock()

def process_triple(triple):
    s, p, o = triple
    encoded_s = str(s).encode("utf-8")
    encoded_o = str(o).encode("utf-8")

    with lock:
        KMV_D[p].update(encoded_s)
        if isinstance(o, URIRef):
            KMV_R[p].update(encoded_o)
        if p == RDF.type and isinstance(o, URIRef):
            KMV_C[o].update(encoded_s)

with ThreadPoolExecutor() as executor:
    executor.map(process_triple, g)

# --- Matching enrichi : littéraux, URI, rdfs:label/comment ---
def search_keywords(keywords):
    results = set()
    for s, p, o in g:
        for k in keywords:
            k_low = k.lower()
            # Literal exact
            if isinstance(o, Literal) and k_low == str(o).lower():
                results.add((s, p, o, k, 1.0))
            # Valeur du mot-clé inclut dans le Literal
            elif isinstance(o, Literal) and k_low in str(o).lower():
                results.add((s, p, o, k, 0.8))
            # rdfs:label direct
            elif p == RDFS.label and k_low in str(o).lower():
                results.add((s, "rdfs:label", o, k, 0.7))
            # URI de propriété contenant le mot-clé
            elif isinstance(p, URIRef) and k_low in str(p).lower():
                results.add((s, p, o, k, 0.5))
            else:
                # label/comment associé à la propriété
                for label_pred in [RDFS.label, RDFS.comment]:
                    for _, _, label_val in g.triples((p, label_pred, None)):
                        if k_low in str(label_val).lower():
                            results.add((s, p, o, k, 0.4))
    return list(results)

def semantic_search_keywords(keywords, threshold=0.7):
    results = set()
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

    for s, p, o in g:
        candidates = []

        # Cas 1 : matcher contre le littéral o
        if isinstance(o, Literal):
            candidates.append((s, p, o, str(o)))

        # Cas 2 : matcher contre le label ou commentaire du prédicat
        for label_pred in [RDFS.label, RDFS.comment]:
            for _, _, label_val in g.triples((p, label_pred, None)):
                candidates.append((s, p, o, str(label_val)))

        # Cas 3 : matcher contre le nom du prédicat lui-même
        if isinstance(p, URIRef):
            candidates.append((s, p, o, str(p).split("/")[-1]))

        # Cas 4 : matcher contre le nom du sujet ou objet URI
        for ent in [s, o]:
            if isinstance(ent, URIRef):
                candidates.append((s, p, o, str(ent).split("/")[-1]))

        # Cas 5 : matcher sur rdf:type lui-même
        if (p == RDF.type and isinstance(o, URIRef)):
            candidates.append((s, p, o, str(o).split("/")[-1]))  # ex: "Movie"

            # aussi labels du type si dispo
            for _, _, label_val in g.triples((o, RDFS.label, None)):
                candidates.append((s, p, o, str(label_val)))

        # Appliquer le matching sémantique
        for s_, p_, o_, text in candidates:
            text_emb = model.encode(text, convert_to_tensor=True)
            sim_scores = util.cos_sim(keyword_embeddings, text_emb)
            for idx, score in enumerate(sim_scores):
                if score.item() >= threshold:
                    results.add((s_, p_, o_, keywords[idx], score.item()))
    return list(results)

# --- Fusion des nœuds (Jaccard sur KMV Dp) ---
def jaccard_similarity(mh1, mh2):
    return mh1.jaccard(mh2)

def fuse_nodes(matches, threshold=0.5):
    groups = []
    seen_keywords = set()
    for (s1, p1, _, k1, _), (s2, p2, _, k2, _) in itertools.combinations(matches, 2):
        if p1 in KMV_D and p2 in KMV_D:
            sim = jaccard_similarity(KMV_D[p1], KMV_D[p2])
            if sim >= threshold:
                groups.append({k1, k2})
                seen_keywords |= {k1, k2}
    for m in matches:
        if m[3] not in seen_keywords:
            groups.append({m[3]})
    return groups

# --- Ajout d’arêtes par containment Dp/Rp ---
def containment_score(mh_a, mh_b):
    return len(set(mh_a.digest()).intersection(mh_b.digest())) / len(mh_a.digest())

def add_edges(groups, matches, threshold=0.3, top_k=10):
    scored_edges = []
    for g1, g2 in itertools.combinations(groups, 2):
        for m1 in matches:
            for m2 in matches:
                if m1[3] in g1 and m2[3] in g2:
                    p, q = m1[1], m2[1]
                    if p in KMV_D and q in KMV_R:
                        score = containment_score(KMV_D[p], KMV_R[q])
                        if score >= threshold:
                            edge = (str(m1[0]), str(p), str(m2[0]), score)
                            scored_edges.append(edge)
    
    # Trier et garder les meilleurs
    top_edges = sorted(scored_edges, key=lambda x: -x[3])[:top_k]
    return [(s, p, o) for s, p, o, _ in top_edges]


# --- Expansion de graphe (fallback) ---
def tree_expansion(nodes, existing_edges, threshold=0.3):
    added = set()
    for s, p, o in g:
        if str(s) in nodes or str(o) in nodes:
            if p in KMV_D:
                for other in KMV_D:
                    if containment_score(KMV_D[other], KMV_D[p]) >= threshold:
                        added.add((str(s), str(p), str(o)))
    return list(added - set(existing_edges))

# --- Génération SPARQL ---
def generate_sparql(matches, edges):
    var_map = {}
    var_counter = 1
    required_triples = set()
    optional_triples = set()
    filters = []

    def get_var(entity):
        nonlocal var_counter
        if str(entity) not in var_map:
            var_map[str(entity)] = f"?v{var_counter}"
            var_counter += 1
        return var_map[str(entity)]

    seen = {}
    for s, p, o, k, score in matches:
        key = (s, p, o)
        if key not in seen or score > seen[key][1]:
            seen[key] = (k, score)

    for (s, p, o), (k, score) in seen.items():
        vs, vo = get_var(s), get_var(o)
        triple = f"{vs} <{p}> {vo} ."
        filter_ = f'FILTER(CONTAINS(LCASE(STR({vo})), "{k.lower()}"))'

        if score >= 0.9:
            required_triples.add(triple)
            filters.append(filter_)
        elif score >= 0.75:
            optional_triples.add((triple, filter_))

    for s, p, o in edges:
        vs, vo = get_var(s), get_var(o)
        required_triples.add(f"{vs} <{p}> {vo} .")

    query = "SELECT DISTINCT * WHERE {\n"
    for line in sorted(required_triples):
        query += f"  {line}\n"
    for line, filter_ in optional_triples:
        query += f"  OPTIONAL {{ {line} {filter_} }}\n"
    for f in filters:
        query += f"  {f}\n"
    query += "} LIMIT 100"

    return query


def generate_sparql_dynamic(matches, edges, context_filter=None):
    """
    Génère une requête SPARQL structurellement connectée, avec un contexte RDF optionnel.
    """
    var_map = {}
    var_counter = 1
    triple_patterns = set()
    optional_triples = set()
    filters = []

    def get_var(entity):
        nonlocal var_counter
        if str(entity) not in var_map:
            var_map[str(entity)] = f"?v{var_counter}"
            var_counter += 1
        return var_map[str(entity)]

    # 1. Sélection des meilleurs matches
    seen = {}
    for s, p, o, k, score in matches:
        key = (s, p, o)
        if key not in seen or score > seen[key][1]:
            seen[key] = (k, score)

    # 2. Triplets + filtres selon le score
    for (s, p, o), (k, score) in seen.items():
        vs, vo = get_var(s), get_var(o)
        triple = f"{vs} <{p}> {vo} ."
        filter_ = f'FILTER(CONTAINS(LCASE(STR({vo})), "{k.lower()}"))'

        if score >= 0.9:
            triple_patterns.add(triple)
            filters.append(filter_)
        elif score >= 0.75:
            optional_triples.add((triple, filter_))

    # 3. Ajout des arêtes de graphe
    for s, p, o in edges:
        vs, vo = get_var(s), get_var(o)
        triple_patterns.add(f"{vs} <{p}> {vo} .")

    # 4. Ajout du contexte dynamique + liaison vers les entités matchées
    if context_filter:
        anchor = context_filter["anchor_var"]
        for s_var, p_uri, o_var in context_filter["path"]:
            triple_patterns.add(f"{s_var} <{p_uri}> {o_var} .")

        lit_var = context_filter["filter_literal_var"]
        val = context_filter["filter_value"].lower()
        filters.append(f'FILTER(CONTAINS(LCASE(STR({lit_var})), "{val}"))')

        # 🔒 Lier toutes les entités matchées à l’ancre via un chemin connu
        for (s, p, o), (k, score) in seen.items():
            s_var = get_var(s)
            if str(s).startswith("http://example.org/actor"):  # ou autre filtre utile
                triple_patterns.add(f"{anchor} <http://example.org/hasActor> {s_var} .")

    # 5. Construction finale
    query = "SELECT DISTINCT * WHERE {\n"
    for line in sorted(triple_patterns):
        query += f"  {line}\n"
    for t, f in optional_triples:
        query += f"  OPTIONAL {{ {t} {f} }}\n"
    for f in filters:
        query += f"  {f}\n"
    query += "} LIMIT 100"
    return query


# --- Classement simplifié (InfoRank) ---
def info_rank(graph):
    scores = {}
    for node in set(graph.subjects()) | set(graph.objects()):
        literals = [o for _, _, o in graph.triples((node, None, None)) if isinstance(o, Literal)]
        scores[str(node)] = len(literals)
    return sorted(scores.items(), key=lambda x: -x[1])

# --- Exécution requête ---
def execute(query):
    print("\n📌 Requête SPARQL générée :\n", query)
    results = g.query(query)
    print("\n🎯 Résultats SPARQL :")
    entities = []
    for r in results:
        line = [str(x) for x in r]
        print(" →", line)
        entities.extend(line)
    return list(set(entities))

# Fonction de article 4
def export_interactive_graph(graph, result_entities=None, filename="graph.html"):
    net = Network(height="600px", width="100%", notebook=False, directed=True)

    highlighted = set(result_entities or [])

    for s, p, o in graph:
        s_str, o_str = str(s), str(o)
        p_str = str(p).split("/")[-1]

        # Couleur différente si le noeud est dans les résultats
        for node in [s_str, o_str]:
            label = node.split("/")[-1]
            color = "orange" if node in highlighted else "#97c2fc"
            net.add_node(node, label=label, title=node, color=color)

        net.add_edge(s_str, o_str, label=p_str)

    path = os.path.join(tempfile.gettempdir(), filename)
    net.save_graph(path)
    return path

def simplify_uri(uri):
    if isinstance(uri, str):
        return uri.split("/")[-1].split("#")[-1]
    return uri

def filter_matches_by_context_path(matches, graph, context_filter, max_depth=3):
    import networkx as nx

    g_nx = nx.Graph()
    for s, p, o in graph:
        g_nx.add_edge(str(s), str(o), label=str(p))

    # Déterminer les nœuds de contexte (via le chemin)
    target_literal = context_filter["filter_value"].lower()
    lit_var = context_filter["filter_literal_var"]

    context_nodes = set()

    for s, p, o in graph:
        if isinstance(o, Literal) and target_literal in str(o).lower():
            context_nodes.add(str(s))  # noeud qui contient le littéral

    if not context_nodes:
        return []

    # Filtrer les matches connectés à un noeud de contexte
    filtered = []
    for s, p, o, k, score in matches:
        if not isinstance(s, URIRef):
            continue
        try:
            for c in context_nodes:
                if nx.has_path(g_nx, c, str(s)):
                    if nx.shortest_path_length(g_nx, c, str(s)) <= max_depth:
                        filtered.append((s, p, o, k, score))
                        break
        except:
            continue

    return filtered


def rdflib_to_networkx(g_rdflib, directed=True):
    g_nx = nx.DiGraph() if directed else nx.Graph()
    for s, p, o in g_rdflib:
        g_nx.add_edge(str(s), str(o), label=str(p))
    return g_nx

def build_context_filter_from_matches(matches, g):
    """
    Analyse les correspondances (matches) pour trouver une valeur contextuelle (lieu, genre),
    et génère dynamiquement un chemin RDF jusqu'à une ancre (ex: ?film).
    """
    context_options = []

    for s, p, o, k, score in matches:
        k_low = k.lower()
        if isinstance(o, Literal):
            lit_str = str(o).lower()
            if k_low in lit_str:
                # LIEU via :lname
                if "lname" in str(p):
                    context_options.append({
                        "filter_value": k,
                        "anchor_var": "?film",
                        "path": [
                            ("?studio", "http://example.org/loc", "?loc"),
                            ("?loc", "http://example.org/lname", "?lname"),
                            ("?studio", "http://example.org/produces", "?film")
                        ],
                        "filter_literal_var": "?lname"
                    })
                # GENRE via :genre
                elif "genre" in str(p):
                    context_options.append({
                        "filter_value": k,
                        "anchor_var": "?film",
                        "path": [
                            ("?film", "http://example.org/genre", "?genre")
                        ],
                        "filter_literal_var": "?genre"
                    })

    return context_options[0] if context_options else None


# --- Programme principal ---
def main():
    st.title("🔍 RDF Keyword Search Assistant")
    st.markdown("Entrez des mots-clés pour rechercher dans le graphe RDF enrichi.")

    user_input = st.text_input("Mots-clés (séparés par espace)", placeholder="Ex: Brandon Western Hollywood")
    mode = st.selectbox("🔀 Mode de recherche", ["🧩 Sans liaison", "🔗 Avec liaison"])

    if user_input:
        keywords = user_input.split()
        with st.spinner("🔎 Recherche des correspondances..."):
            matches = semantic_search_keywords(keywords, 0.7)
            if mode == "🔗 Avec liaison":
              context_filter = build_context_filter_from_matches(matches, g)
              if context_filter:
                  matches = filter_matches_by_context_path(matches, g, context_filter)

        if not matches:
            st.error("❌ Aucun mot-clé trouvé dans les littéraux, URI ou labels.")
            return

        st.success("✅ Correspondances trouvées !")
        st.write("**🔗 Triplets correspondants (avec score) :**")
        for m in matches:
            s, p, o, k, score = m
            st.write(f"→ `{k}` matché sur ({simplify_uri(p)}) avec score {score:.2f}")

        with st.spinner("🧩 Fusion des nœuds similaires..."):
            groups = fuse_nodes(matches)
        st.write("📚 **Groupes de mots-clés fusionnés :**", groups)

        with st.spinner("➕ Ajout d’arêtes par containment..."):
            edges = add_edges(groups, matches)

        if not edges:
            with st.spinner("🌲 Expansion du graphe par similarité..."):
                nodes = {str(m[0]) for m in matches}
                edges = tree_expansion(nodes, edges)
            st.info("✨ Expansion appliquée, arêtes suggérées :")
        st.write("🧠 **Arêtes retenues :**", edges)

        with st.spinner("⚙️ Génération de la requête SPARQL..."):
            if mode == "🔗 Avec liaison":
              context_filter = build_context_filter_from_matches(matches, g)
              query = generate_sparql_dynamic(matches, edges, context_filter)
            else:
              query = generate_sparql(matches, edges)
        st.code(query, language="sparql")

        with st.spinner("🚀 Exécution de la requête..."):
            result_entities = execute(query)

        st.subheader("🎯 Résultats")
        if result_entities:
            st.success("Résultats obtenus :")
            st.write(result_entities)
        else:
            st.warning("Aucun résultat retourné.")

        st.subheader("📊 Classement InfoRank")
        ranks = info_rank(g)
        for ent, score in ranks:
            if ent in result_entities:
                st.write(f"🔸 `{ent}` : {score}")

        st.divider()
        with st.expander("🧪 Détails intermédiaires"):
            st.write("**Mots-clés détectés :**", [m[3] for m in matches])
            st.write("**Groupes fusionnés :**", groups)
            st.write("**Arêtes ajoutées :**", edges)

        with st.expander("🌐 Voir le graphe RDF (interactif)"):
          if st.button("🖼️ Générer le graphe interactif (résultats en orange)"):
            html_path = export_interactive_graph(g, result_entities, "graph.html")
            st.success(f"Graphe interactif généré.")
            with open(html_path, "r", encoding="utf-8") as f:
              html_content = f.read()
              st.components.v1.html(html_content, height=600, scrolling=True)
              st.download_button("📂 Télécharger le graphe", f, file_name="graph.html")

if __name__ == "__main__":
    main()