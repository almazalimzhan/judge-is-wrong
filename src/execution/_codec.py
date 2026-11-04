"""Bounded, data-only RPC codec. No Python objects or exception objects cross."""

import base64
import json
import sys

MAX_BYTES = 1_048_576
MAX_DEPTH = 40
MAX_NODES = 100_000


class UnsupportedValue(Exception):
    pass


class InvalidMessage(Exception):
    pass


def encode(value, *, trusted=False):
    seen, count = set(), [0]

    def visit(item, depth):
        count[0] += 1
        if depth > MAX_DEPTH or count[0] > MAX_NODES:
            raise UnsupportedValue("RPC size limit")
        kind = type(item)
        if item is None:
            return ["none"]
        if kind is bool:
            return ["bool", item]
        if kind is int:
            if item.bit_length() > 13000:
                raise UnsupportedValue("RPC integer limit")
            return ["int", str(item)]
        if kind is float:
            return ["float", repr(item)]
        if kind is complex:
            return ["complex", repr(item.real), repr(item.imag)]
        if kind is str:
            return ["str", item]
        if kind is bytes:
            return ["bytes", base64.b64encode(item).decode("ascii")]
        numpy = sys.modules.get("numpy") if trusted else None
        if numpy is not None and isinstance(item, numpy.generic):
            return visit(item.item(), depth + 1)
        if kind not in {list, tuple, dict, set, frozenset}:
            raise UnsupportedValue("Only exact builtin data types are supported")
        if id(item) in seen:
            raise UnsupportedValue("Container alias or cycle is unsupported")
        seen.add(id(item))
        if kind is dict:
            return ["dict", [[visit(k, depth + 1), visit(v, depth + 1)] for k, v in item.items()]]
        children = [visit(child, depth + 1) for child in item]
        if kind in {set, frozenset}:
            children.sort(key=json.dumps)
        return [{list: "list", tuple: "tuple", set: "set", frozenset: "frozenset"}[kind], children]

    result = visit(value, 0)
    dumps(result)
    return result


def decode(value):
    count = [0]

    def visit(item, depth):
        count[0] += 1
        if depth > MAX_DEPTH or count[0] > MAX_NODES or type(item) is not list or not item:
            raise InvalidMessage("Invalid RPC value")
        tag = item[0]
        if tag == "none" and len(item) == 1:
            return None
        if len(item) != (3 if tag == "complex" else 2):
            raise InvalidMessage("Invalid RPC shape")
        value = item[1]
        if tag == "bool" and type(value) is bool:
            return value
        if tag == "int" and type(value) is str and len(value) <= 4096:
            return int(value)
        if tag == "float" and type(value) is str and len(value) <= 64:
            return float(value)
        if tag == "complex" and all(type(v) is str and len(v) <= 64 for v in item[1:]):
            return complex(float(item[1]), float(item[2]))
        if tag == "str" and type(value) is str:
            return value
        if tag == "bytes" and type(value) is str:
            return base64.b64decode(value, validate=True)
        if tag in {"list", "tuple", "set", "frozenset"} and type(value) is list:
            return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[tag](
                [visit(v, depth + 1) for v in value]
            )
        if tag == "dict" and type(value) is list:
            result = {}
            for pair in value:
                if type(pair) is not list or len(pair) != 2:
                    raise InvalidMessage("Invalid dictionary pair")
                key = visit(pair[0], depth + 1)
                if key in result:
                    raise InvalidMessage("Duplicate dictionary key")
                result[key] = visit(pair[1], depth + 1)
            return result
        raise InvalidMessage("Invalid RPC tag")

    try:
        return visit(value, 0)
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise InvalidMessage("Malformed RPC data") from error


def loads(raw):
    if len(raw) > MAX_BYTES:
        raise InvalidMessage("Message too large")
    depth, quoted, escaped = 0, False, False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > MAX_DEPTH + 10:
                raise InvalidMessage("Message too deep")
        elif byte in (93, 125):
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InvalidMessage("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise InvalidMessage("Nonfinite JSON primitive")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise InvalidMessage("Malformed JSON") from error


def dumps(value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    if len(raw) > MAX_BYTES:
        raise UnsupportedValue("Message too large")
    return raw
