import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import image_server_lib as image_lib
import spark_llm as app


class StateModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "models_dir": "/srv/models",
            "listen_host": "127.0.0.1",
            "consumer_context_length": 1024,
            "headroom_factor": 2,
            "max_memory_gb": 128,
            "min_free_gib_after_load": 4,
            "consumers": [
                {
                    "name": "Primary LLM API",
                    "port": 8000,
                    "alias": "local-llm",
                    "requires_tool_calling": True,
                },
                {
                    "name": "Image Generation API",
                    "port": 8001,
                    "alias": "local-image",
                    "required_engine": "diffusers",
                },
            ],
            "auto": {"weights_overhead": 1.15, "kv_margin_gib": 8},
        }
        self.text_decl = {
            "engine": "vllm-docker",
            "image": "example/vllm:1",
            "weights": "/models/text-model",
            "served_name": "text-model",
            "gpu_memory_utilization": 0.4,
            "max_model_len": 4096,
            "est_weights_gb": 20,
            "extra_flags": [
                "--tool-call-parser",
                "example",
                "--enable-auto-tool-choice",
            ],
        }
        self.image_decl = {
            "engine": "diffusers",
            "weights": "/models/image-model",
            "served_name": "image-model",
            "est_weights_gb": 30,
        }

    def test_toml_writer_quotes_dotted_declaration_names(self):
        text = app.toml_dump_flat_tables({"model.3": {"port": 8000}})
        self.assertIn('["model.3"]', text)

    def test_capability_filter_separates_text_and_image_consumers(self):
        self.assertTrue(app.consumer_compatible(self.text_decl, self.cfg["consumers"][0]))
        self.assertFalse(app.consumer_compatible(self.image_decl, self.cfg["consumers"][0]))
        self.assertTrue(app.consumer_compatible(self.image_decl, self.cfg["consumers"][1]))

    def test_missing_tool_flags_are_rejected(self):
        declaration = dict(self.text_decl, extra_flags=[])
        errors, _warnings = app.validate_load(
            "text", declaration, 8000, self.cfg, self.cfg["consumers"][0]
        )
        self.assertTrue(any("tool" in error.lower() for error in errors))

    def test_fresh_model_does_not_evict_a_healthy_consumer(self):
        loaded = {"current": {"port": 8000}}
        port, consumer = app.decide_port("candidate", self.text_decl, loaded, self.cfg)
        self.assertNotEqual(port, 8000)
        self.assertIsNone(consumer)

    def test_free_floating_model_promotes_to_primary_consumer(self):
        loaded = {"candidate": {"port": 8010}, "current": {"port": 8000}}
        port, consumer = app.decide_port("candidate", self.text_decl, loaded, self.cfg)
        self.assertEqual(8000, port)
        self.assertEqual("Primary LLM API", consumer["name"])

    def test_image_model_routes_to_image_consumer(self):
        port, consumer = app.decide_port("image", self.image_decl, {}, self.cfg)
        self.assertEqual(8001, port)
        self.assertEqual("Image Generation API", consumer["name"])

    def test_docker_ports_bind_to_loopback_by_default(self):
        argv = app.compose_docker_argv(
            "text", self.text_decl, 8000, self.cfg, "local-llm", 128
        )
        self.assertEqual("127.0.0.1:8000:8000", argv[argv.index("-p") + 1])

    def test_llamacpp_uses_the_configured_listener(self):
        cfg = dict(self.cfg, engines={"llamacpp_bin": "/usr/bin/llama-server"})
        declaration = {
            "engine": "llamacpp",
            "weights": "/srv/models/example.gguf",
            "served_name": "example",
            "max_model_len": 4096,
        }
        argv = app.compose_llamacpp_argv(declaration, 8002, cfg, None)
        self.assertEqual("127.0.0.1", argv[argv.index("--host") + 1])

    def test_reserve_estimate_uses_configured_memory_ceiling(self):
        reserve = app.reserve_estimate_gb(self.text_decl, 130.6, self.cfg)
        self.assertLessEqual(reserve, 128)
        self.assertGreater(reserve, 0)


class ImageProtocolTests(unittest.TestCase):
    def test_size_parser_has_safe_fallback(self):
        self.assertEqual((1024, 1024), image_lib.parse_size("1024x1024", "1328x1328"))
        self.assertEqual((1328, 1328), image_lib.parse_size("invalid", "1328x1328"))

    def test_stream_chunks_round_trip_and_stay_bounded(self):
        content = "x" * 250_000
        chunks = image_lib.chunk_text(content, 100_000)
        self.assertEqual(content, "".join(chunks))
        self.assertLessEqual(max(map(len, chunks)), 100_000)

    def test_open_webui_auxiliary_jobs_are_detected(self):
        self.assertTrue(image_lib.is_auxiliary_task("### Task:\nGenerate a title"))
        self.assertFalse(image_lib.is_auxiliary_task("Generate a lighthouse at sunset"))


if __name__ == "__main__":
    unittest.main()
