import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("vllm"), "requires vLLM")
class AttentionBackendTest(unittest.TestCase):
    def test_backend_reaches_engine_kwargs_only_when_requested(self):
        import profiler
        from profiler.__main__ import build_parser
        from profiler.core.config import ProfileArgs
        from profiler.core.engine import _profile_engine_overrides

        parser = build_parser()
        ns = parser.parse_args([
            "profile", "microsoft/phi-4", "--hardware", "MI300X",
            "--attention-backend", "TRITON_ATTN",
        ])
        args = ProfileArgs(
            architecture="phi3",
            model="microsoft/phi-4",
            hardware="MI300X",
            attention_backend=ns.attention_backend,
        )

        self.assertEqual(
            _profile_engine_overrides(args)["attention_backend"],
            "TRITON_ATTN",
        )
        self.assertNotIn(
            "attention_backend",
            _profile_engine_overrides(ProfileArgs(
                architecture="phi3",
                model="microsoft/phi-4",
                hardware="MI300X",
            )),
        )


if __name__ == "__main__":
    unittest.main()
