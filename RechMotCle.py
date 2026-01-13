import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph, Literal
from datasketch import MinHash
from SPARQLWrapper import SPARQLWrapper, JSON

# 🔹 Charger le fichier RDF
rdf_file = "./test-rdf.ttl"

# 🔹 Charger le graphe RDF
g = Graph()
g.parse(rdf_file, format="ttl")

# 🔹 Prétraitement : Calcul des KMV-Synopses
synopses = {}
for s, p, o in g.triples((None, None, None)):
    if p not in synopses:
        synopses[p] = MinHash(num_perm=128)
    synopses[p].update(str(o).encode("utf8"))

# 🔹 Fonction de recherche de mots-clés
def search_keyword(keyword):
    results = []
    for s, p, o in g.triples((None, None, None)):
        if isinstance(o, Literal) and keyword.lower() in str(o).lower():
            results.append({"Sujet": str(s), "Propriété": str(p), "Valeur": str(o)})
    return pd.DataFrame(results)

# 🔹 Construction d'une forêt de requêtes
def build_forest(matches):
    forest = []
    for index, row in matches.iterrows():
        g_sub = nx.Graph()
        g_sub.add_edge(row["Sujet"], row["Valeur"], label=row["Propriété"])
        forest.append(g_sub)
    return forest

# 🔹 Fusion des sous-graphes en un graphe final
def merge_graphs(forest):
    final_graph = nx.Graph()
    for subgraph in forest:
        final_graph = nx.compose(final_graph, subgraph)
    return final_graph

# 🔹 Expansion d'arbre (Tree Expansion)
def expand_graph(graph, g):
    for node in list(graph.nodes):
        for s, p, o in g.triples((None, None, None)):
            if str(s) in graph.nodes or str(o) in graph.nodes:
                graph.add_edge(str(s), str(o), label=str(p))
    return graph

# 🔹 Optimisation Steiner Tree
def optimize_steiner_tree(graph):
    pruned_graph = nx.minimum_spanning_tree(graph)
    return pruned_graph

# 🔹 Classement des résultats RDF
def rank_results(graph):
    scores = nx.degree_centrality(graph)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes

# 🔹 Affichage du graphe avec couleurs et sauvegarde
def draw_graph(graph, filename="graph.png", k_value=0.3):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42, k=k_value)  # Ajuste k pour modifier l'espace entre les nœuds
    node_types = {n: "film" if "Movie" in n else "person" if "Person" in n else "genre" for n in graph.nodes}
    colors = {"film": "lightblue", "person": "lightgreen", "genre": "lightcoral"}
    node_colors = [colors.get(node_types.get(n, "film"), "gray") for n in graph.nodes]
    
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=800)
    labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8)
    
    plt.axis('off')
    plt.savefig(filename)
    plt.show()  # Afficher le graphe

# 🔹 Génération de la requête SPARQL
def generate_sparql(query_graph):
    sparql_query = "SELECT DISTINCT ?s WHERE {\n"
    for edge in query_graph.edges(data=True):
        source = f"<{edge[0]}>" if edge[0].startswith("http") else f'"{edge[0]}"'
        target = f"<{edge[1]}>" if edge[1].startswith("http") else f'"{edge[1]}"'
        predicate = f"<{edge[2]['label']}>"

        sparql_query += f"  {source} {predicate} {target} .\n"
    sparql_query += "}"
    return sparql_query


# 🔹 Exécution de la requête SPARQL
def execute_sparql(query):
    endpoint = "http://dbpedia.org/sparql"
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return pd.DataFrame(results["results"]["bindings"])

# 🔹 Demander un mot-clé à l'utilisateur
keyword = input("Entrez un mot-clé : ")
matches = search_keyword(keyword)

# 🔹 Construire la forêt de requêtes puis fusionner les graphes
query_forest = build_forest(matches)
query_graph = merge_graphs(query_forest)
query_graph = expand_graph(query_graph, g)
query_graph = optimize_steiner_tree(query_graph)

# 🔹 Dessiner le graphe (avec k=0.3 pour espacer les nœuds)
draw_graph(query_graph, k_value=0.3)

# 🔹 Générer et exécuter la requête SPARQL
sparql_query = generate_sparql(query_graph)
print("Requête SPARQL générée :\n", sparql_query)
results = execute_sparql(sparql_query)

# 🔹 Classement et affichage des résultats
ranked_results = rank_results(query_graph)
print("Classement des résultats :")
for node, score in ranked_results:
    print(f"{node}: {score}")
print("Résultats de la requête SPARQL :")
print(results)
