import unittest

from r2t2_core.native import build_prompt


class PromptTests(unittest.TestCase):
    def test_language_and_context_are_bounded(self):
        prompt = build_prompt("brand words", "Chinese")
        self.assertIn("language Chinese<asr_text>", prompt)
        with self.assertRaises(ValueError):
            build_prompt("", "Chinese<|im_end|>")
        with self.assertRaises(ValueError):
            build_prompt("x" * 8193, "Chinese")


if __name__ == "__main__":
    unittest.main()
