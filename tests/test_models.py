from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from src.models.base import CachedModel, ModelResponse, build_model
from src.models.openai_compatible import ModelAPIError, OpenAICompatibleBackend


class CountingBackend:
    name = "counting"

    def __init__(self):
        self.calls = []

    def endpoint(self, role):
        return "mock://counter"

    def generate(self, messages, **kwargs):
        self.calls.append(kwargs)
        return ModelResponse(str(len(self.calls)), {"completion_tokens": 1}, {"output": "raw"})


class ModelTests(unittest.TestCase):
    def test_cache_replays_but_distinct_draws_do_not_collapse(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = CountingBackend()
            model = CachedModel(backend, Path(temp), {"judge": "counter"}, seed=17)
            prompt = [{"role": "user", "content": "test"}]
            self.assertEqual(model.complete(prompt, role="judge", call_id="member/0"), "1")
            self.assertEqual(model.complete(prompt, role="judge", call_id="member/0"), "1")
            self.assertEqual(model.complete(prompt, role="judge", call_id="member/1"), "2")
            self.assertEqual(len(backend.calls), 2)
            logs = [
                json.loads(line) for line in (Path(temp) / "calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["cache_hit"] for row in logs], [False, True, False])
            self.assertNotEqual(logs[0]["seed"], logs[2]["seed"])
            self.assertTrue(all(row["prompt_hash"] and row["timestamp"] for row in logs))
            cached = json.loads(next(Path(temp).glob("*.json")).read_text())
            self.assertEqual(cached["raw_response"], {"output": "raw"})

    def test_failed_call_is_audited_without_exception_text(self):
        class FailedBackend(CountingBackend):
            def generate(self, *args, **kwargs):
                raise RuntimeError("SENSITIVE_EXCEPTION_DETAIL")

        with tempfile.TemporaryDirectory() as temp:
            model = CachedModel(FailedBackend(), Path(temp), {"judge": "counter"})
            with self.assertRaises(RuntimeError):
                model.complete([], role="judge", call_id="failure")
            text = (Path(temp) / "calls.jsonl").read_text()
            row = json.loads(text)
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["error_type"], "RuntimeError")
            self.assertNotIn("SENSITIVE_EXCEPTION_DETAIL", text)

    def test_mock_ignores_real_model_environment_and_keeps_metadata_secret_free(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, {"JUDGE_MODEL": "chosen-judge", "LLM_API_KEY": "SECRET_TOKEN"}),
        ):
            model = build_model({"backend": "mock"}, Path(temp), seed=42)
            metadata = model.metadata()
            self.assertEqual(metadata["models"]["judge"], "mock-judge-v1")
            selected = build_model(
                {"backend": "mock", "judge_model": "explicit-fixture"}, Path(temp)
            )
            self.assertEqual(selected.models["judge"], "explicit-fixture")
            self.assertNotIn("SECRET_TOKEN", json.dumps(metadata))

    def test_api_request_preserves_parameters_and_extracts_usage(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self, limit):
                return json.dumps(
                    {"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 7}}
                ).encode()

        with patch("src.models.openai_compatible.urlopen", return_value=Response()) as request:
            backend = OpenAICompatibleBackend("https://example.test/v1", send_seed=True)
            output = backend.generate(
                [{"role": "user", "content": "hi"}],
                model="model",
                role="judge",
                temperature=0.3,
                seed=12,
                call_id="test",
            )
            self.assertEqual(output.content, "hello")
            self.assertEqual(output.usage["total_tokens"], 7)
            sent = request.call_args.args[0]
            self.assertEqual(sent.full_url, "https://example.test/v1/chat/completions")
            self.assertEqual(json.loads(sent.data)["seed"], 12)

    def test_real_http_wire_protocol_with_local_stub(self):
        observed = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                observed["path"] = self.path
                observed["authorization"] = self.headers.get("Authorization")
                observed["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                response = json.dumps(
                    {"choices": [{"message": {"content": "wire-ok"}}], "usage": {"total_tokens": 9}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        try:
            server = HTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("Local socket binding is blocked by the host sandbox")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            with patch.dict(
                os.environ,
                {"JUDGE_API_BASE": endpoint, "JUDGE_API_KEY": "stub-key", "no_proxy": "127.0.0.1"},
            ):
                backend = OpenAICompatibleBackend(endpoint, max_tokens=123, send_seed=True)
                response = backend.generate(
                    [{"role": "user", "content": "ping"}],
                    model="stub-model",
                    role="judge",
                    temperature=0.25,
                    seed=17,
                    call_id="wire",
                )
            self.assertEqual(response.content, "wire-ok")
            self.assertEqual(response.usage, {"total_tokens": 9})
            self.assertEqual(observed["path"], "/v1/chat/completions")
            self.assertEqual(observed["authorization"], "Bearer stub-key")
            self.assertEqual(
                observed["body"],
                {
                    "model": "stub-model",
                    "temperature": 0.25,
                    "max_tokens": 123,
                    "seed": 17,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_generation_settings_change_cache_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            for token_limit in (10, 20):
                backend = OpenAICompatibleBackend("https://example.test/v1", max_tokens=token_limit)
                with patch.object(backend, "generate", return_value=ModelResponse("ok")) as call:
                    model = CachedModel(backend, Path(temp), {"judge": "stub"})
                    model.complete([], role="judge", call_id="same")
                    self.assertEqual(call.call_count, 1)
                    self.assertEqual(
                        model.metadata()["backend_settings"]["max_tokens"], token_limit
                    )
            self.assertEqual(len(list(Path(temp).glob("*.json"))), 2)

    def test_malformed_api_response_is_not_silently_scored(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self, limit):
                return b'{"choices": []}'

        with patch("src.models.openai_compatible.urlopen", return_value=Response()):
            with self.assertRaises(ModelAPIError):
                OpenAICompatibleBackend("http://localhost:1234/v1").generate(
                    [], model="x", role="judge", temperature=0, seed=1, call_id="x"
                )

    def test_endpoint_rejects_embedded_secrets(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleBackend("https://user:secret@example.test/v1").endpoint("judge")


if __name__ == "__main__":
    unittest.main()
