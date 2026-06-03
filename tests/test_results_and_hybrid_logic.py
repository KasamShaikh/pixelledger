import unittest

from src.config import AzureConfig
from src.pipelines.base import DocumentInput, PipelineResult
from src.pipelines.hybrid import HybridDIPipeline
from src.ui import results_view


class SummaryLogicTests(unittest.TestCase):
    def test_allow_cost_guidance_only_for_multi_gpt5(self):
        rows = [
            {"pipeline": "DI + GPT-5.4 mini", "error": ""},
            {"pipeline": "DI + GPT-5.1", "error": ""},
        ]
        self.assertTrue(results_view._allow_cost_guidance(rows))

    def test_disallow_cost_guidance_for_mixed_gpt4_gpt5(self):
        rows = [
            {"pipeline": "DI + GPT-5.4 mini", "error": ""},
            {"pipeline": "DI + GPT-4.0 Mini", "error": ""},
        ]
        self.assertFalse(results_view._allow_cost_guidance(rows))

    def test_single_model_assessment_has_strength_and_improvement(self):
        row = {
            "pipeline": "DI + GPT-5.1",
            "cer": 0.22,
            "wer": 0.30,
            "judge_avg": None,
            "judge_accuracy": None,
            "judge_structure": None,
        }
        went_well, improve = results_view._single_model_assessment(row, gt_text="x")
        self.assertGreaterEqual(len(went_well), 1)
        self.assertGreaterEqual(len(improve), 1)


class HybridConfidenceRetentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_keeps_di_confidence_when_llm_fails(self):
        cfg = AzureConfig(
            di_endpoint="https://example-di",
            di_key="",
            aoai_endpoint="https://example-aoai",
            aoai_key="",
            aoai_api_version="2024-12-01-preview",
            dep_gpt5="gpt-5",
            dep_gpt51="gpt-5.1",
            dep_gpt5_mini="gpt-5-mini",
            dep_gpt54_mini="gpt-5.4-mini",
            dep_gpt4o="gpt-4o",
            dep_gpt4o_mini="gpt-4o-mini",
            dep_judge="gpt-5",
        )

        class FakeDIPipeline:
            def __init__(self, *_args, **_kwargs):
                pass

            async def run(self, _doc):
                return PipelineResult(
                    pipeline_id="di-prebuilt-layout",
                    display_name="Document Intelligence",
                    raw_text="di markdown",
                    pages=3,
                    confidence_avg=0.87,
                    per_line_confidence=[0.9, 0.84],
                    cost_usd=0.03,
                    raw_response={"source": "di"},
                )

        class FailingCompletions:
            async def create(self, **_kwargs):
                raise RuntimeError("forced llm failure")

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.chat = type("Chat", (), {"completions": FailingCompletions()})()

            async def close(self):
                return None

        import src.pipelines.hybrid as hybrid_module

        original_di = hybrid_module.DocIntelligencePipeline
        original_client = hybrid_module.AsyncAzureOpenAI
        hybrid_module.DocIntelligencePipeline = FakeDIPipeline
        hybrid_module.AsyncAzureOpenAI = FakeClient
        try:
            pipeline = HybridDIPipeline(cfg, "dep", "gpt-5.1", "prebuilt-layout")
            doc = DocumentInput(
                filename="x.pdf",
                content=b"%PDF-test",
                mime_type="application/pdf",
                images=[b"x"],
            )
            result = await pipeline.run(doc, structuring_prompt="test")
        finally:
            hybrid_module.DocIntelligencePipeline = original_di
            hybrid_module.AsyncAzureOpenAI = original_client

        self.assertIsNotNone(result.error)
        self.assertEqual(result.pages, 3)
        self.assertAlmostEqual(result.confidence_avg or 0, 0.87, places=2)
        self.assertEqual(result.per_line_confidence, [0.9, 0.84])
        self.assertGreaterEqual(result.cost_usd, 0.03)


if __name__ == "__main__":
    unittest.main()
