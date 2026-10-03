import json
import unittest

from manual_agent import (
    create_vector_index,
    parse_manual_answer,
    parse_selected_ids,
    search_vector_index,
)


class FakeEmbeddingModel:
    def embed(self, texts):
        for text in texts:
            normalized = text.casefold()
            if "smoke" in normalized or "exhaust" in normalized:
                yield [1.0, 0.0, 0.0]
            else:
                yield [0.0, 1.0, 0.0]


class ManualAgentTests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            {
                "id": "P1",
                "reference": "manual.pdf, p. 2",
                "text": "White smoke from the exhaust may indicate oil entering the chamber.",
            },
            {
                "id": "P2",
                "reference": "manual.pdf, p. 8",
                "text": "Check the brake fluid level before riding.",
            },
        ]

    def test_vector_search_returns_matching_manual_text(self):
        index = create_vector_index(self.passages, FakeEmbeddingModel())
        results = search_vector_index(index, "white smoke exhaust")
        self.assertEqual(results[0]["id"], "P1")

    def test_model_ids_are_restricted_to_candidates(self):
        model_output = json.dumps({"passage_ids": ["P999", "P1", "P1"]})
        selected = parse_selected_ids(model_output, ["P1", "P2"])
        self.assertEqual(selected, ["P1"])

    def test_answer_requires_citation_to_a_supplied_passage(self):
        model_output = json.dumps(
            {"answer": "The manual notes white smoke from the exhaust.", "passage_ids": ["P1"]}
        )
        result = parse_manual_answer(model_output, self.passages)
        self.assertEqual(result["passage_ids"], ["P1"])

    def test_answer_without_valid_citation_is_rejected(self):
        model_output = json.dumps(
            {"answer": "The manual says to replace the piston.", "passage_ids": ["P999"]}
        )
        self.assertIsNone(parse_manual_answer(model_output, self.passages))


if __name__ == "__main__":
    unittest.main()