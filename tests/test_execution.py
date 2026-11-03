import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

from src.data.base import TestCase
from src.execution import Executor, SandboxUnavailable


class ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = Executor(timeout_seconds=0.4)
        try:
            cls.executor.check_available()
        except SandboxUnavailable as error:
            raise unittest.SkipTest(str(error)) from error

    def test_fraction_and_binary_correctness(self):
        result = self.executor.run(
            "def f(x): return x+1", [TestCase("assert f(1)==2"), TestCase("assert f(2)==9")]
        )
        self.assertEqual(result.score, 0.5)
        self.assertFalse(result.all_passed)
        self.assertEqual(result.status_counts, {"passed": 1, "assertion_failure": 1})

    def test_timeout(self):
        result = self.executor.run("while True: pass", [TestCase("assert True")])
        self.assertEqual(result.passed, 0)
        self.assertEqual(result.status_counts, {"timeout": 1})

    def test_no_stdout_or_test_text_in_result(self):
        result = self.executor.run(
            "print('FAKE_PASSED')", [TestCase("assert False, 'HIDDEN_SENTINEL'")]
        )
        serialized = json.dumps(result.to_dict())
        self.assertNotIn("HIDDEN_SENTINEL", serialized)
        self.assertNotIn("FAKE_PASSED", serialized)
        self.assertEqual(result.passed, 0)

    def test_successful_exit_is_not_a_pass(self):
        result = self.executor.run("import os; os._exit(0)", [TestCase("assert True")])
        self.assertEqual(result.passed, 0)

    def test_builtin_exec_monkeypatch_cannot_disable_tests(self):
        result = self.executor.run(
            "import builtins; builtins.exec = lambda *a, **k: None", [TestCase("assert False")]
        )
        self.assertEqual(result.passed, 0)

    def test_user_files_are_not_readable(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as stream:
            stream.write("PRIVATE_HOST_SENTINEL")
            protected = Path(stream.name)
        try:
            code = f"def f():\n    try:\n        return open({str(protected)!r}).read()\n    except (PermissionError, FileNotFoundError):\n        return 'blocked'"
            result = self.executor.run(code, [TestCase("assert f() == 'blocked'")])
            self.assertTrue(result.all_passed)
        finally:
            protected.unlink()

    def test_network_access_is_denied(self):
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        code = f"""import socket
def f():
    try:
        s = socket.socket()
        s.settimeout(0.1)
        s.connect(('127.0.0.1', {port}))
        return 'connected'
    except OSError:
        return 'blocked'
"""
        result = self.executor.run(code, [TestCase("assert f() == 'blocked'")])
        self.assertTrue(result.all_passed)

    def test_secrets_not_in_environment(self):
        os.environ["RESEARCH_PRIVATE_SENTINEL"] = "secret"
        try:
            result = self.executor.run(
                "import os", [TestCase("assert 'RESEARCH_PRIVATE_SENTINEL' not in os.environ")]
            )
            self.assertTrue(result.all_passed)
        finally:
            del os.environ["RESEARCH_PRIVATE_SENTINEL"]


class ExecutionConfigurationTests(unittest.TestCase):
    def test_no_unsafe_fallback(self):
        with self.assertRaises(ValueError):
            Executor(backend="unsafe")


if __name__ == "__main__":
    unittest.main()
