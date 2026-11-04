"""Conservative candidate capabilities; this is not a general Python security proof."""

import ast
import builtins
import importlib
import types


class PolicyViolation(Exception):
    pass


SAFE_MODULES = {
    "math",
    "cmath",
    "re",
    "itertools",
    "functools",
    "collections",
    "heapq",
    "bisect",
    "statistics",
    "random",
    "hashlib",
    "decimal",
    "fractions",
    "typing",
}
FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "open",
    "input",
    "help",
    "breakpoint",
    "exit",
    "quit",
    "__import__",
    "__builtins__",
}
FORBIDDEN_ATTRIBUTES = {
    "sys",
    "os",
    "builtins",
    "modules",
    "get_field",
    "get_value",
    "Formatter",
    "attrgetter",
    "methodcaller",
    "settrace",
    "setprofile",
    "f_locals",
    "f_globals",
    "tb_frame",
    "gi_frame",
    "cr_frame",
    "ag_frame",
    "format_map",
}


def check_source(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)
        ):
            raise PolicyViolation("Classes, generators and asynchronous code are unsupported")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise PolicyViolation("Reflection or dynamic execution is unsupported")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES
        ):
            raise PolicyViolation("Private or reflective attributes are unsupported")
        if isinstance(node, ast.Import):
            if any(alias.name not in SAFE_MODULES for alias in node.names):
                raise PolicyViolation("Import outside supported standard-library modules")
        if isinstance(node, ast.ImportFrom):
            if (
                node.level
                or node.module not in SAFE_MODULES
                or any(
                    alias.name.startswith("_")
                    or alias.name in FORBIDDEN_ATTRIBUTES
                    or alias.name == "*"
                    for alias in node.names
                )
            ):
                raise PolicyViolation("Unsupported import")
        if isinstance(node, (ast.FunctionDef, ast.arg)):
            name = node.name if isinstance(node, ast.FunctionDef) else node.arg
            if name.startswith("__") or name in FORBIDDEN_NAMES:
                raise PolicyViolation("Reserved binding")
    return tree


def candidate_builtins():
    # Module facades prevent public aliases such as typing.sys from exposing
    # runtime machinery. Only explicitly imported safe modules are available.
    facades = {}
    for name in SAFE_MODULES:
        module = importlib.import_module(name)
        facade = types.ModuleType(name)
        for attribute, value in vars(module).items():
            if (
                not attribute.startswith("_")
                and attribute not in FORBIDDEN_ATTRIBUTES
                and not isinstance(value, types.ModuleType)
            ):
                setattr(facade, attribute, value)
        facades[name] = facade

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in facades:
            raise PolicyViolation("Unsupported import")
        return facades[name]

    result = {
        name: value
        for name, value in vars(builtins).items()
        if name not in FORBIDDEN_NAMES and not name.startswith("__")
    }
    result["__import__"] = safe_import
    return result
