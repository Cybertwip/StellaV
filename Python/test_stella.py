#!/usr/bin/env python3
from __future__ import annotations

import unittest

import ast

from artifact import (
    extract_fenced_code,
    fallback_research_script,
    find_write_tool,
    infer_artifact_path,
    looks_like_artifact_request,
    research_query_from,
    synthesize_write_call,
    write_tool_arg_keys,
)
from knowledge_base import semantic_response
from reasoner import normalize_reason_size, ollama_model_for
from tensor_engine import RetrievedChunk, TensorEngine


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


class ArtifactTests(unittest.TestCase):
    def test_detects_script_requests(self) -> None:
        self.assertTrue(looks_like_artifact_request("Write a python chempy script to begin a cancer vaccine."))
        self.assertTrue(looks_like_artifact_request("Write the python script now, do that based on the research"))
        self.assertFalse(looks_like_artifact_request("summarize this preprint about cancer vaccines"))
        self.assertFalse(looks_like_artifact_request("what is a randomized trial?"))

    def test_research_query_strips_script_words(self) -> None:
        q = research_query_from("Write a python chempy script to begin a cancer vaccine.", "")
        self.assertIn("cancer", q)
        self.assertIn("vaccine", q)
        self.assertNotIn("python", q)
        self.assertNotIn("chempy", q)

    def test_follow_up_keeps_doi(self) -> None:
        q = research_query_from(
            "Write the python script now, do that based on the research",
            "ScienceOpen preprint: 10.14293/s2199-1006.1.sor-.ppsvhlh.v1",
        )
        self.assertIn("10.14293/s2199-1006.1.sor-.ppsvhlh.v1", q)

    def test_write_tool_schema_and_fallback(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["filePath", "content"],
                },
            },
        }
        self.assertEqual(find_write_tool([tool])["function"]["name"], "write")
        self.assertEqual(write_tool_arg_keys(tool), ("filePath", "content"))
        path = infer_artifact_path("Write a python chempy script to begin a cancer vaccine.")
        self.assertTrue(path.endswith(".py"))
        hits = [
            RetrievedChunk(
                text="RNA vaccines have potential as Novel therapeutic options for major disease such as cancer for development of personalized medicine.",
                score=0.9,
                title="mRNA Vaccine: The Next Generation Vaccine Revolution",
                doi="10.14293/s2199-1006.1.sor-.ppsvhlh.v1",
            )
        ]
        script = fallback_research_script("Write a python chempy script to begin a cancer vaccine.", hits)
        self.assertIn("chempy", script)
        self.assertIn("10.14293/s2199-1006.1.sor-.ppsvhlh.v1", script)
        self.assertIn("NOT a vaccine", script)
        ast.parse(script)
        call = synthesize_write_call(tool, path, script)
        self.assertEqual(call["function"]["name"], "write")
        self.assertIn("filePath", call["function"]["arguments"])
        self.assertEqual(extract_fenced_code("```python\nprint(1)\n```"), "print(1)")


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
