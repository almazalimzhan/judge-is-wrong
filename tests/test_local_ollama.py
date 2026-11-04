import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.base import build_model
from src.models.openai_compatible import ModelAPIError, NoLocalRedirect, OpenAICompatibleBackend


class LocalOllamaTests(unittest.TestCase):
    def test_local_endpoint_and_model_cannot_use_external_environment(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"JUDGE_API_BASE": "https://external.invalid/v1", "JUDGE_MODEL": "external-model"},
            ),
        ):
            model = build_model(
                {
                    "backend": "openai_compatible",
                    "local_only": True,
                    "api_base": "http://localhost:11434/v1",
                    "proposer_model": "llama3.2:3b",
                    "judge_model": "llama3.2:3b",
                    "critic_model": "llama3.2:3b",
                },
                Path(directory),
            )
            self.assertEqual(model.models["judge"], "llama3.2:3b")
            self.assertEqual(
                model.backend.endpoint("judge"), "http://localhost:11434/v1/chat/completions"
            )
        for address in ("https://localhost/v1", "http://external.invalid/v1"):
            with self.assertRaises(ValueError):
                OpenAICompatibleBackend(address, local_only=True).endpoint("judge")

    def test_local_wire_uses_nonstreaming_json_and_placeholder_auth(self):
        backend = OpenAICompatibleBackend(
            "http://localhost:11434/v1", local_only=True, json_roles=True
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"choices":[{"message":{"content":"{}","reasoning":"private"}}]}'
        )
        opener = MagicMock()
        opener.open.return_value = response
        with patch("src.models.openai_compatible.build_opener", return_value=opener):
            result = backend.generate(
                [], model="llama3.2:3b", role="judge", temperature=0, seed=42, call_id="test"
            )
        request = opener.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertIn('"type": "boolean"', body["messages"][0]["content"])
        self.assertEqual(request.headers["Authorization"], "Bearer ollama")
        self.assertEqual(result.content, "{}")

    def test_reasoning_is_never_used_as_final_content_and_redirects_fail(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"choices":[{"message":{"content":"","reasoning":"{}"}}]}'
        )
        with (
            patch("src.models.openai_compatible.urlopen", return_value=response),
            self.assertRaises(ModelAPIError),
        ):
            OpenAICompatibleBackend("http://localhost/v1").generate(
                [], model="x", role="judge", temperature=0, seed=0, call_id="test"
            )
        with self.assertRaises(ModelAPIError):
            NoLocalRedirect().redirect_request(
                None, None, 302, "redirect", {}, "https://external.invalid"
            )
