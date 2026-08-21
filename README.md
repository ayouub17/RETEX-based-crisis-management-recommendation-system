# Moteur de recherche sémantique RETEX

Ce module indexe les retours d'expérience de la table PostgreSQL `retex` et
retrouve les cas les plus proches d'une nouvelle crise. Il utilise
`all-MiniLM-L6-v2`, FAISS et la similarité cosinus (embeddings normalisés +
produit scalaire).

## 1. Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Complétez ensuite les accès PostgreSQL dans `.env`. Ne versionnez jamais ce
fichier, car il contient le mot de passe.

## 2. Construction de l'index

Le texte vectorisé est construit à partir de `title`, `crisis_type`, `summary`,
`description`, `actions` et `recommendations`. Les champs vides sont ignorés.

```powershell
python generate_embeddings.py
python build_index.py
```

La seconde commande crée `data/faiss/retex.index` et
`data/faiss/retex_metadata.json`. Relancez-la après chaque ajout ou modification
de RETEX en base.

## 3. Recherche

```powershell
python search_engine.py "Inondations majeures avec évacuations et coupure d'électricité" --top-k 5
```

Chaque résultat contient un score de similarité cosinus, le titre,
l'organisation, les actions et les recommandations. Le score est compris entre
-1 et 1 ; plus il est élevé, plus les textes sont sémantiquement proches.

## 4. Intégration Python

```python
from main import find_similar_retex

results = find_similar_retex("Cyberattaque par ransomware sur un hôpital", top_k=3)
for result in results:
    print(result["title"], result["similarity_score"])
```

## 5. Génération de recommandations

Le module de recommandation regroupe les champs `actions` et `recommendations`
des RETEX récupérés. Chaque élément reçoit un poids égal à la somme des scores
de similarité des RETEX qui le contiennent. Les éléments les plus soutenus par
les cas les plus proches apparaissent donc en premier.

```powershell
python recommendation_engine.py "Cyberattaque ransomware sur un hôpital" --top-k 5 --limit 10
```

Le résultat doit être validé par un expert de la gestion de crise : le score
exprime une proximité sémantique, pas une garantie de réussite opérationnelle.

Depuis Python :

```python
from main import recommend_for_crisis

proposals = recommend_for_crisis("Inondation avec évacuations", top_k=5)
for action in proposals["actions"]:
    print(action["text"], action["weighted_score"])
```

## 6. Interface web

Installez les dépendances mises à jour puis démarrez le serveur :

```powershell
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Ouvrez ensuite `http://127.0.0.1:8000` dans votre navigateur. L'interface
appelle `POST /api/recommendations` et affiche les actions pondérées, les
recommandations ainsi que les RETEX qui les justifient. La documentation API
interactive est disponible sur `http://127.0.0.1:8000/docs`.
