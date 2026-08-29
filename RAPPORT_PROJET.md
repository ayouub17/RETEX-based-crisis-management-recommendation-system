# Rapport de projet
## CrisisLens RETEX : recherche sémantique et recommandation pour la gestion de crise

## 1. Résumé exécutif

CrisisLens RETEX est une application Python qui aide un utilisateur à analyser une nouvelle situation de crise à partir de retours d'expérience, ou RETEX. Le système recherche dans une base PostgreSQL les cas les plus proches du problème décrit, puis extrait et classe les actions et recommandations associées à ces cas.

Le projet combine :

- PostgreSQL pour stocker les RETEX;
- Sentence Transformers pour transformer les textes en vecteurs numériques;
- FAISS pour effectuer une recherche rapide par similarité vectorielle;
- FastAPI pour exposer une API HTTP;
- une interface web HTML/CSS/JavaScript pour saisir une crise et afficher les résultats.

Le système ne produit pas une décision automatique. Il fournit une aide à l'analyse : les résultats doivent être relus et validés par un expert de la gestion de crise.

## 2. Problématique et objectifs

Lorsqu'une organisation fait face à une crise, elle dispose souvent d'informations dispersées dans des rapports et des retours d'expérience. Une recherche par mots-clés peut manquer des documents qui utilisent un vocabulaire différent. L'objectif du projet est donc de permettre une recherche par sens plutôt que par correspondance exacte des mots.

À partir d'une description libre, l'application doit :

1. comprendre le contenu général de la crise;
2. retrouver les RETEX sémantiquement les plus proches;
3. afficher le score de proximité et les informations des cas trouvés;
4. regrouper les actions et recommandations issues de ces cas;
5. classer les propositions en fonction de la proximité et de la répétition dans les sources;
6. conserver la traçabilité vers les RETEX utilisés.

## 3. Architecture générale

Le projet est organisé en plusieurs couches.

```text
PostgreSQL - table retex
        |
        v
RetexRepository.fetch_all()
        |
        v
build_retex_text()
        |
        v
SentenceTransformer - embeddings normalisés
        |
        v
FAISS IndexFlatIP + metadata JSON
        |
        v
SemanticSearchEngine.search()
        |
        v
RecommendationEngine.recommend()
        |
        +--> API FastAPI
        |
        +--> interface web
        |
        +--> commandes CLI
```

La génération de l'index est effectuée en amont. Pendant une recherche, l'application charge l'index et ses métadonnées, encode uniquement la nouvelle description de crise, puis interroge FAISS.

## 4. Organisation des fichiers

### 4.1 Configuration et lancement

- `config.py` centralise les paramètres de l'application. Il charge le fichier `.env`, vérifie les variables PostgreSQL obligatoires et définit les chemins de l'index FAISS.
- `main.py` fournit la façade applicative. `get_search_engine()` charge le modèle et l'index une seule fois grâce à `lru_cache`.
- `app.py` expose l'API FastAPI et sert l'interface web.
- `requirements.txt` liste les dépendances Python.
- `README.md` contient les commandes rapides d'installation et d'utilisation.

### 4.2 Accès aux données

- `database/postgres.py` contient `RetexRepository`.
- La méthode `fetch_all()` lit la table `retex` dans un ordre stable par identifiant.
- Les colonnes attendues sont : `id`, `source`, `title`, `crisis_type`, `organization`, `country`, `report_date`, `summary`, `description`, `actions`, `recommendations` et `url`.
- La connexion est fermée automatiquement après la lecture et dispose d'un délai de connexion de dix secondes.

Le projet attend donc que la base PostgreSQL et la table `retex` existent déjà. Aucun script de création de schéma ou de migration SQL n'est fourni dans l'état actuel.

### 4.3 Préparation du texte et embeddings

- `embeddings/text_builder.py` construit la représentation textuelle de chaque RETEX.
- Les champs utilisés sont le titre, le type de crise, le résumé, la description, les actions et les recommandations.
- Les champs nuls ou vides sont ignorés.
- Un RETEX sans aucun contenu textuel provoque une erreur, ce qui évite de créer un vecteur vide.

`embeddings/embedder.py` encapsule `SentenceTransformer`. Le modèle par défaut est `all-MiniLM-L6-v2`. Chaque texte est converti en vecteur `float32`, puis normalisé selon la norme L2. Cette normalisation permet d'utiliser le produit scalaire comme équivalent de la similarité cosinus.

### 4.4 Index vectoriel

- `search/faiss_store.py` sauvegarde et recharge l'index.
- `FaissStore.save()` crée un index `faiss.IndexFlatIP`, ajoute les vecteurs et écrit deux fichiers :
  - `data/faiss/retex.index` contient les vecteurs;
  - `data/faiss/retex_metadata.json` contient les RETEX dans le même ordre.
- `FaissStore.load()` vérifie que le nombre de vecteurs est égal au nombre de métadonnées.

L'association entre un résultat FAISS et un RETEX repose sur la position du vecteur dans l'index et la position du dictionnaire correspondant dans le fichier JSON.

### 4.5 Recherche sémantique

`search/engine.py` contient `SemanticSearchEngine`.

Lors d'une recherche :

1. la description est contrôlée pour vérifier qu'elle n'est pas vide;
2. elle est encodée par le même modèle que celui utilisé pour l'index;
3. FAISS compare le vecteur de la requête aux vecteurs des RETEX;
4. les `top_k` positions les plus proches sont retournées;
5. chaque résultat reçoit un champ `similarity_score` et les métadonnées du RETEX.

Le score est un produit scalaire entre vecteurs normalisés, donc une similarité cosinus. Plus le score est élevé, plus le contenu est proche selon le modèle linguistique. Dans la recherche, les scores sont généralement compris entre -1 et 1.

### 4.6 Moteur de recommandation

`recommendations/engine.py` transforme les résultats de recherche en propositions opérationnelles.

Le moteur :

1. lit les champs `actions` et `recommendations` de chaque cas;
2. sépare les éléments sur les retours à la ligne, les points-virgules et les puces;
3. normalise les espaces et la casse pour regrouper les textes identiques;
4. remplace les scores négatifs par zéro pour le calcul des recommandations;
5. additionne les scores des cas contenant une même proposition;
6. trie les éléments par score décroissant;
7. conserve les sources, les titres et les scores des RETEX associés.

Ainsi, une action présente dans plusieurs cas proches obtient davantage de poids qu'une action isolée. Le regroupement reste toutefois textuel : deux formulations synonymes mais différentes ne sont pas automatiquement fusionnées.

## 5. Pipeline de construction de l'index

Le fichier `build_index.py` réalise le pipeline complet :

1. chargement de la configuration;
2. lecture de tous les RETEX dans PostgreSQL;
3. arrêt si la table est vide;
4. construction d'un texte cohérent pour chaque cas;
5. génération de tous les embeddings;
6. création et sauvegarde de l'index FAISS et des métadonnées;
7. affichage du nombre de cas et de la dimension des vecteurs.

La commande `generate_embeddings.py` a un rôle différent : elle vérifie que la lecture des données et la génération des embeddings fonctionnent, mais elle ne sauvegarde pas les vecteurs. Pour créer ou mettre à jour l'index persistant, il faut utiliser `build_index.py`.

L'index doit être reconstruit après l'ajout, la suppression ou la modification d'un RETEX. Il faut également reconstruire l'index si le modèle d'embedding change.

## 6. Façade applicative

`main.py` simplifie l'utilisation du système :

- `find_similar_retex(crisis_text, top_k)` renvoie les cas similaires;
- `recommend_for_crisis(crisis_text, top_k, limit)` effectue la recherche puis l'agrégation des propositions;
- `get_search_engine()` met en cache le modèle et l'index pour éviter de les recharger à chaque requête.

Exemple Python :

```python
from main import recommend_for_crisis

result = recommend_for_crisis(
    "Cyberattaque par ransomware touchant les systèmes d'un hôpital",
    top_k=5,
    limit=10,
)

for action in result["actions"]:
    print(action["text"], action["weighted_score"])
```

## 7. API FastAPI

`app.py` expose trois routes principales.

### `GET /`

Sert le fichier `templates/index.html`, qui constitue la page principale.

### `GET /health`

Retourne :

```json
{"status": "ok"}
```

Cette route vérifie seulement que le serveur HTTP répond. Elle ne vérifie pas la disponibilité de PostgreSQL, du modèle ou de l'index FAISS.

### `POST /api/recommendations`

Reçoit un objet JSON de la forme :

```json
{
  "description": "Inondation importante nécessitant une évacuation et une coordination des secours",
  "top_k": 5,
  "limit": 10
}
```

Contraintes appliquées par Pydantic :

- `description` : entre 10 et 10 000 caractères;
- `top_k` : entre 1 et 20, avec cinq par défaut;
- `limit` : entre 1 et 30, avec dix par défaut.

La réponse contient :

- `similar_cases` : les RETEX proches et leurs scores;
- `actions` : les actions pondérées;
- `recommendations` : les recommandations pondérées.

Les erreurs de configuration, de fichiers d'index ou de traitement connues sont converties en réponse HTTP d'erreur.

## 8. Interface web

L'interface est composée de :

- `templates/index.html` pour la structure HTML;
- `static/styles.css` pour la présentation responsive;
- `static/app.js` pour les appels asynchrones et l'affichage des résultats.

L'utilisateur saisit une description, sélectionne le nombre de RETEX à analyser, puis lance l'analyse. JavaScript envoie la requête à l'API avec `fetch`. La page affiche ensuite :

1. les actions prioritaires;
2. les recommandations;
3. les cas similaires avec leur score;
4. l'organisation et les réponses apportées dans les RETEX.

Les données textuelles dynamiques sont principalement insérées avec `textContent`, ce qui réduit les risques d'injection HTML. L'interface affiche néanmoins un statut opérationnel statique : ce statut ne constitue pas une vérification réelle de la santé de tous les composants.

## 9. Interfaces en ligne de commande

### Recherche de cas similaires

```powershell
python search_engine.py "Fuite de données après une cyberattaque" --top-k 5
```

Cette commande affiche le score, le titre, l'organisation, les actions et les recommandations de chaque RETEX.

### Génération de recommandations

```powershell
python recommendation_engine.py "Fuite de données après une cyberattaque" --top-k 5 --limit 10
```

Cette commande affiche les actions et recommandations classées avec leur poids et le nombre de sources.

## 10. Installation et démarrage

Depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Créer ensuite un fichier `.env` contenant au minimum :

```text
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=nom_de_la_base
POSTGRES_USER=utilisateur
POSTGRES_PASSWORD=mot_de_passe
```

Les paramètres optionnels sont :

```text
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss/retex.index
FAISS_METADATA_PATH=data/faiss/retex_metadata.json
```

Construire l'index :

```powershell
python build_index.py
```

Démarrer l'application :

```powershell
python -m uvicorn app:app --reload
```

La page est accessible à l'adresse `http://127.0.0.1:8000`. La documentation interactive de FastAPI est disponible à `http://127.0.0.1:8000/docs`.

Le premier chargement du modèle peut être long et nécessiter un téléchargement depuis Hugging Face selon l'environnement local.

## 11. Données présentes dans le dépôt

Le pipeline actif utilise PostgreSQL et les deux artefacts FAISS.

- `data/faiss/retex.index` et `data/faiss/retex_metadata.json` sont les artefacts de recherche.
- `dataset.csv` est présent dans le projet, mais aucun module actif ne le lit actuellement.
- `output_json/` contient des documents structurés, notamment des ressources ENISA et des rapports sur les vulnérabilités, mais ces fichiers ne sont pas connectés au pipeline actuel.
- `reports/` contient des ressources de rapports, mais elles ne sont pas directement exploitées par les modules de recherche.
- `src/` ne contient pas de composant applicatif central dans l'état observé.

Cette distinction est importante : déposer un document dans `output_json/` ou `reports/` ne suffit pas pour le rendre recherchable. Il doit être transformé en RETEX, inséré dans PostgreSQL, puis l'index doit être reconstruit.

## 12. Dépendances techniques

- `fastapi` : définition de l'API;
- `uvicorn` : serveur ASGI;
- `pydantic` : validation des requêtes, installé avec FastAPI;
- `sentence-transformers` : génération des représentations sémantiques;
- `faiss-cpu` : recherche vectorielle;
- `numpy` : manipulation des matrices de vecteurs;
- `psycopg2-binary` : connexion à PostgreSQL;
- `python-dotenv` : chargement des variables du fichier `.env`.

## 13. Sécurité, qualité et limites

### Points positifs

- Le mot de passe PostgreSQL est lu dans l'environnement et n'est pas codé en dur.
- Les connexions PostgreSQL sont fermées automatiquement.
- Les entrées de l'API sont limitées par longueur et par intervalle numérique.
- Les recommandations conservent la provenance des cas sources.
- L'indexation et la recherche utilisent le même paramètre de modèle par défaut.

### Limites actuelles

- Il n'y a pas de tests automatisés ni de schéma SQL versionné dans le dépôt.
- La cohérence de l'index est contrôlée sur le nombre de lignes, mais pas sur le modèle utilisé, la dimension, la normalisation ou le contenu détaillé des métadonnées.
- Un index construit avec un ancien contenu de PostgreSQL peut devenir obsolète.
- Modifier `EMBEDDING_MODEL` sans reconstruire l'index peut provoquer une incompatibilité de dimension ou dégrader les résultats.
- L'endpoint `/health` est superficiel.
- L'API ne montre pas de mécanisme d'authentification, d'autorisation ou de limitation de débit.
- Les champs d'actions sont découpés selon quelques séparateurs connus; des formats complexes peuvent être mal séparés.
- Le regroupement ne reconnaît pas les synonymes.
- Les résultats peuvent exposer directement des informations internes présentes en base.
- Le fichier `.env.example` mentionné dans le README n'est pas présent dans l'état actuel du dépôt.
- Les performances et la mémoire du chargement du modèle sont à surveiller en production.

## 14. Améliorations recommandées

1. Ajouter des tests unitaires pour `build_retex_text`, `FaissStore`, `SemanticSearchEngine` et `RecommendationEngine`.
2. Ajouter une migration SQL et un schéma documenté pour la table `retex`.
3. Enregistrer dans les métadonnées la version du modèle, la dimension et la date de construction de l'index.
4. Faire vérifier par `/health` la présence de l'index, la compatibilité du modèle et, si nécessaire, la connexion PostgreSQL.
5. Ajouter un mécanisme d'authentification, des logs structurés et une limitation de débit avant une mise en production.
6. Normaliser le modèle de données des actions et recommandations afin d'éviter un découpage textuel fragile.
7. Ajouter une étape d'ingestion pour intégrer `dataset.csv`, `output_json/` et les rapports si ces sources doivent être recherchées.
8. Mettre en place une évaluation métier avec des requêtes représentatives et des jugements d'experts sur la pertinence du Top-K.
9. Prévoir une stratégie de reconstruction et de déploiement atomique de l'index.
10. Ajouter des filtres par type de crise, pays, organisation, date ou source.

## 15. Conclusion

CrisisLens RETEX constitue un prototype fonctionnel de recherche sémantique appliquée à la gestion de crise. Sa valeur principale est de rapprocher une situation nouvelle de cas historiques même lorsque les mots employés diffèrent, puis de transformer les cas retrouvés en propositions opérationnelles traçables.

Le parcours complet est le suivant : une description est saisie, elle est convertie en embedding, FAISS retrouve les RETEX les plus proches, leurs actions et recommandations sont agrégées, puis l'API et l'interface affichent les résultats. Pour passer d'un prototype à un service de production, les priorités sont la couverture de tests, la gestion du cycle de vie de l'index, la sécurisation de l'API et l'intégration formelle des autres sources documentaires.
