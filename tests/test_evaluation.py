import unittest

from r2t2_core.evaluation import aggregate, quality_gate, score


class EvaluationTests(unittest.TestCase):
    def test_weighted_errors_and_frozen_normalization(self):
        self.assertEqual(score("你好，世界！", "你好世界", "cer")["errors"], 0)
        self.assertEqual(score("The train leaves at seven.",
                               "the train leaves at 7", "wer")["errors"], 1)
        cases = [
            {"q8_score": {"errors": 0, "reference_units": 100}},
            {"q8_score": {"errors": 1, "reference_units": 1}},
        ]
        self.assertAlmostEqual(aggregate(cases)["rate"], 1 / 101)

    def test_small_or_incomplete_corpus_cannot_pass(self):
        base = {"language": "Chinese", "tags": [], "status": "complete",
                "q8_score": {"errors": 0, "reference_units": 5}}
        self.assertEqual(quality_gate([base], mode="offline", audio_seconds=1800)["status"],
                         "insufficient_corpus")
        incomplete = {**base, "status": "requires_review"}
        self.assertEqual(quality_gate([incomplete], mode="offline", audio_seconds=1800)["status"],
                         "incomplete_results")

    def test_full_gate_requires_paired_bf16_evidence(self):
        def case(language, tags=()):
            return {"language": language, "tags": list(tags), "status": "complete",
                    "q8_score": {"errors": 1, "reference_units": 10},
                    "bf16_score": {"errors": 0, "reference_units": 10}}

        cases = ([case("Chinese") for _ in range(50)] +
                 [case("English") for _ in range(50)] +
                 [case("Mixed", (tag,)) for tag in ("mixed", "quiet", "noise", "short")
                  for _ in range(5)])
        for index, item in enumerate(cases):
            item["id"] = f"case_{index}"
        self.assertEqual(quality_gate(cases, mode="offline", audio_seconds=1800)["status"], "fail")
        del cases[0]["bf16_score"]
        self.assertEqual(quality_gate(cases, mode="offline", audio_seconds=1800)["status"],
                         "missing_bf16_baseline")

    def test_catastrophic_challenge_blocks_good_paired_average(self):
        def case(language, tag="", errors=0):
            return {"id": f"{language}_{tag}_{errors}", "language": language,
                    "tags": [tag] if tag else [], "status": "complete",
                    "q8_score": {"errors": errors, "reference_units": 10},
                    "bf16_score": {"errors": 0, "reference_units": 10}}
        cases = ([case("Chinese") for _ in range(50)] +
                 [case("English") for _ in range(50)] +
                 [case("Mixed", tag) for tag in ("mixed", "quiet", "noise", "short")
                  for _ in range(5)])
        cases[-1] = case("Mixed", "short", 5)
        gate = quality_gate(cases, mode="offline", audio_seconds=1800)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["severe_challenge_cases"], ["Mixed_short_5"])


if __name__ == "__main__":
    unittest.main()
