import unittest

from recommendations.engine import RecommendationEngine
from search.engine import SemanticSearchEngine


class FakeIndex:
    ntotal = 3

    def search(self, vector, limit):
        # first result: cyber-related and relevant, second: mismatched transport case,
        # third: low-similarity unrelated case
        scores = [[0.92, 0.11, 0.03]]
        positions = [[0, 1, 2]]
        return scores, positions


class FakeEmbedder:
    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class RecommendationEngineTests(unittest.TestCase):
    def test_search_filters_mismatched_cases(self):
        engine = SemanticSearchEngine(
            index=FakeIndex(),
            metadata=[
                {
                    "id": 1,
                    "title": "Ransomware attack in hospital",
                    "crisis_type": "cyber",
                    "summary": "Malware encrypted patient systems.",
                    "description": "Critical healthcare systems were locked by ransomware.",
                    "actions": "Activate crisis cell; isolate infected servers",
                    "recommendations": "Run backup tests; coordinate with legal counsel",
                },
                {
                    "id": 2,
                    "title": "Rail network disruption",
                    "crisis_type": "transport",
                    "summary": "Frequent delays after a signal failure.",
                    "description": "Rail traffic was disturbed by a signal outage.",
                    "actions": "Reroute trains; inform passengers",
                    "recommendations": "Improve infrastructure maintenance",
                },
                {
                    "id": 3,
                    "title": "Supply chain break at manufacturing plant",
                    "crisis_type": "industrial",
                    "summary": "Late deliveries blocked production.",
                    "description": "A plant lost part of its logistics capacity.",
                    "actions": "Temporarily reroute shipments",
                    "recommendations": "Re-design supplier map",
                },
            ],
            embedder=FakeEmbedder(),
        )

        results = engine.search("Ransomware attack on hospital IT systems", top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_recommendations_get_priority_and_type(self):
        engine = RecommendationEngine()
        results = engine.recommend(
            [
                {
                    "id": 1,
                    "title": "Ransomware attack in hospital",
                    "crisis_type": "cyber",
                    "similarity_score": 0.92,
                    "actions": "Activate crisis cell; isolate infected systems; communicate with public health authorities",
                    "recommendations": "Set up an incident governance board; run backup tests; maintain access control monitoring",
                }
            ],
            limit=10,
        )

        self.assertGreater(len(results["actions"]), 2)
        self.assertGreater(len(results["recommendations"]), 2)
        self.assertTrue(all(item["text"] for item in results["actions"]))
        self.assertTrue(all(item["text"] for item in results["recommendations"]))
        priorities = {item["priority"] for item in results["actions"]}
        self.assertTrue(priorities.intersection({"urgence_immediate", "court_terme", "stabilisation"}))
        classifications = {item["classification"] for item in results["recommendations"]}
        self.assertTrue(classifications.intersection({"structurelle", "bonne_pratique_generale"}))

    def test_actions_and_recommendations_are_specific(self):
        engine = RecommendationEngine()
        results = engine.recommend(
            [{
                "id": 1,
                "title": "Ransomware hospital crisis",
                "crisis_type": "cyber",
                "similarity_score": 0.92,
                "actions": "Basculement sur systèmes de secours; déploiement de systèmes papier; communication de crise 24/7",
                "recommendations": "Sauvegardes hors ligne; Plan de continuité de service; Formation cybersécurité; Assurance cyber",
            }],
            limit=10,
        )

        action_texts = {item["text"] for item in results["actions"]}
        recommendation_texts = {item["text"] for item in results["recommendations"]}
        self.assertIn("Basculement sur systèmes de secours", action_texts)
        self.assertIn("Plan de continuité de service", recommendation_texts)
        self.assertIn("Formation cybersécurité", recommendation_texts)

    def test_variant_case_types_are_not_filtered_out(self):
        engine = SemanticSearchEngine(
            index=FakeIndex(),
            metadata=[
                {
                    "id": 1,
                    "title": "Cybersecurity breach in healthcare",
                    "crisis_type": "cybersecurity",
                    "summary": "Incident de fuite de données dans un hôpital.",
                    "description": "Des données patients ont été exfiltrées.",
                    "actions": "Activation de la cellule de crise; isolement du réseau",
                    "recommendations": "Renforcement de la sécurité; test de continuité",
                }
            ],
            embedder=FakeEmbedder(),
        )

        results = engine.search("Fuite de données clients après une cyberattaque", top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_relevant_low_similarity_cases_are_not_rejected(self):
        class LowSimilarityIndex:
            ntotal = 1

            def search(self, vector, limit):
                scores = [[0.08]]
                positions = [[0]]
                return scores, positions

        engine = SemanticSearchEngine(
            index=LowSimilarityIndex(),
            metadata=[
                {
                    "id": 11,
                    "title": "Exfiltration de données interne",
                    "crisis_type": "cybersecurity",
                    "summary": "Fuite de données suite à une compromission du réseau interne.",
                    "description": "Le compte administrateur a été compromis et des données confidentielles ont été exfiltrées.",
                    "actions": "isolation du réseau; activation de la cellule de crise",
                    "recommendations": "renforcement de la segmentation; surveillance des accès",
                }
            ],
            embedder=FakeEmbedder(),
        )

        results = engine.search("Fuite de données après compromission de comptes administrateurs", top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 11)


if __name__ == "__main__":
    unittest.main()
