# search-engine

Moteur de recherche simple sur un graphe **RDF (Turtle)** : tu donnes des mots-clés, le script retrouve les ressources pertinentes, génère une requête **SPARQL**, exécute la requête, puis affiche les résultats.  
Le projet peut aussi produire une visualisation du graphe.

![Demo](demo_search-engine.gif)

---

## Fonctionnalités

- Recherche par mots-clés dans un graphe RDF (URI, littéraux, `rdfs:label`)
- Regroupement de termes proches (MinHash / similarité)
- Génération automatique d’une requête SPARQL
- Exécution de la requête sur le RDF chargé
- Visualisation du graphe (ex : `graph.png`)

---

## Fichiers du repo

- `NavigationKMV.py` : app Streamlit principale (version “avancée”)
- `RechMotCle2.py` : app Streamlit plus simple
- `RechMotCle.py` : prototype / ancienne version
- `test-rdf.ttl` : dataset RDF d’exemple
- `requirements.txt` : dépendances Python
- `graph.png` : exemple de graphe
- `demo_search-engine.gif` : démo rapide

pip install -r requirements.txt
