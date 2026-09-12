#!/usr/bin/env python3
from __future__ import annotations

import unittest

from knowledge_base import semantic_response
from reasoner import normalize_reason_size, ollama_model_for
from tensor_engine import TensorEngine


class TensorEngineTests(unittest.TestCase):
    def test_ranks_matching_preprint(self) -> None:
        engine = TensorEngine(device="cpu")
        engine.add_paper(
            {
                "doi": "10.14293/example.immuno",
                "title": "Checkpoint inhibitors in melanoma",
                "abstract": "PD-1 blockade can produce durable responses in melanoma immunotherapy trials.",
                "url": "https://www.scienceopen.com/hosted-document?doi=10.14293/example.immuno",
                "source": "scienceopen",
            }
        )
        engine.add_paper(
            {
                "doi": "10.14293/example.soil",
                "title": "Agricultural soil microbiome",
                "abstract": "Crop yield depends on soil nitrogen cycling in wheat fields far from oncology clinics.",
                "url": "https://www.scienceopen.com/hosted-document?doi=10.14293/example.soil",
                "source": "scienceopen",
            }
        )
        hits = engine.search("cancer immunotherapy checkpoint PD-1 melanoma", k=2)
        self.assertTrue(hits)
        self.assertIn("checkpoint", hits[0].title.lower())
        status = engine.status()
        self.assertIn("multihead-scaled-dot-product", status["strategies"])
        self.assertEqual(status["device"], "cpu")


class ReasonerTests(unittest.TestCase):
    def test_selectable_sizes(self) -> None:
        self.assertEqual(normalize_reason_size("3b"), "3b")
        self.assertEqual(normalize_reason_size("1.5b"), "1.5b")
        self.assertNotEqual(ollama_model_for("1.5b"), ollama_model_for("3b"))


class KnowledgeTests(unittest.TestCase):
    def test_medical_identity_and_no_game_fallback(self) -> None:
        text, conf = semantic_response("Who are you?")
        self.assertIn("ScienceOpen", text)
        self.assertGreater(conf, 0.9)
        trial, _ = semantic_response("What is randomization?")
        self.assertIn("chance", trial.lower())
        game = semantic_response("What is a game engine swapchain?")
        self.assertTrue(game is None or "game engine" not in game[0].lower() or "not generate" in game[0].lower())


if __name__ == "__main__":
    unittest.main()
