import json
import os
import socket
import subprocess
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
            self.assertEqual(self.raw_os_probe(code + "\nprint(f())"), b"blocked")
            result = self.executor.run(code, [TestCase("assert f() == 'blocked'")])
            self.assertEqual(result.status_counts, {"integrity_violation": 1})
            self.assertFalse(result.oracle_valid)
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
        self.assertEqual(self.raw_os_probe(code + "\nprint(f())"), b"blocked")
        result = self.executor.run(code, [TestCase("assert f() == 'blocked'")])
        self.assertFalse(result.oracle_valid)

    def test_secrets_not_in_environment(self):
        os.environ["RESEARCH_PRIVATE_SENTINEL"] = "secret"
        try:
            result = self.executor.run(
                "def f(): return True",
                [
                    TestCase(
                        "assert 'RESEARCH_PRIVATE_SENTINEL' not in os.environ and f()",
                        setup="import os",
                    )
                ],
            )
            self.assertTrue(result.all_passed)
        finally:
            del os.environ["RESEARCH_PRIVATE_SENTINEL"]

    def raw_os_probe(self, source):
        """Exercise the OS boundary itself, independently of candidate policy."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()
            path.chmod(0o777)
            (path / "candidate.py").write_text(source)
            result = subprocess.run(
                self.executor._command(path),
                cwd=path,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                capture_output=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            return result.stdout.strip()

    def test_runtime_harness_tampering_is_rejected_before_execution(self):
        attempts = [
            "import sys\nsys.modules['__main__'].encode = lambda value, **kwargs: ['int', '2']\ndef f(x): return 999",
            "import sys\nframe=sys._getframe(1)\nframe.f_locals['verdicts']['assertion_failure']=frame.f_locals['verdicts']['passed']",
            "import inspect\ndef f(x): return inspect.currentframe()",
            "import signal\nsignal.setitimer(signal.ITIMER_REAL, 0)\ndef f(x): return 999",
            "def f(x):\n    return globals()",
            "def f(x):\n    return f.__globals__",
        ]
        for source in attempts:
            with self.subTest(source=source.splitlines()[0]):
                result = self.executor.run(source, [TestCase("assert f(1) == 2")])
                self.assertEqual(result.passed, 0)
                self.assertFalse(result.oracle_valid)
                self.assertEqual(result.status_counts, {"integrity_violation": 1})

    def test_candidate_cannot_replace_trusted_check_or_exception_namespace(self):
        result = self.executor.run(
            "def check(candidate): return True\nAssertionError=ValueError\ndef f(x): return 999",
            [TestCase("def check(candidate):\n    assert candidate(1) == 2\ncheck(f)")],
        )
        self.assertEqual(result.status_counts, {"assertion_failure": 1})
        self.assertTrue(result.oracle_valid)

    def test_unsupported_mutation_remains_invalid_when_test_catches_exception(self):
        result = self.executor.run(
            "def f(values):\n    values.sort()\n    return values",
            [TestCase("try:\n    f([2, 1])\nexcept Exception:\n    pass")],
        )
        self.assertEqual(result.status_counts, {"unsupported_protocol": 1})
        self.assertFalse(result.oracle_valid)

    def test_alias_between_positional_and_keyword_arguments_is_unsupported(self):
        result = self.executor.run(
            "def f(x, other): return x is other",
            [TestCase("a=[]\ntry:\n    f(a, other=a)\nexcept Exception:\n    pass")],
        )
        self.assertEqual(result.status_counts, {"unsupported_protocol": 1})
        self.assertFalse(result.oracle_valid)

    def test_typed_results_and_candidate_prints_preserve_function_semantics(self):
        result = self.executor.run(
            "def f(x):\n    print('debug')\n    return {'values': (x, {1, 2}), 'data': b'ok'}",
            [TestCase("assert f(3) == {'values': (3, {1, 2}), 'data': b'ok'}")],
        )
        self.assertTrue(result.all_passed)

    def test_broken_trusted_setup_is_an_oracle_error(self):
        result = self.executor.run(
            "def f(): return 1", [TestCase("assert f() == 1", setup="raise RuntimeError('setup')")]
        )
        self.assertEqual(result.status_counts, {"controller_error": 1})
        self.assertFalse(result.oracle_valid)


class ExecutionConfigurationTests(unittest.TestCase):
    def test_no_unsafe_fallback(self):
        with self.assertRaises(ValueError):
            Executor(backend="unsafe")


if __name__ == "__main__":
    unittest.main()
