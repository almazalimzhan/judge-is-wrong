import unittest

from src.execution._codec import InvalidMessage, UnsupportedValue, decode, encode, loads


class CodecTests(unittest.TestCase):
    def test_plain_typed_values_round_trip(self):
        value = {("a", 1): [None, True, b"bytes", {2, 3}], "complex": 2 + 3j}
        self.assertEqual(decode(encode(value)), value)

    def test_custom_subclass_cannot_supply_comparison_behavior(self):
        class Deceptive(int):
            def __eq__(self, other):
                return True

        with self.assertRaises(UnsupportedValue):
            encode(Deceptive(999))

    def test_object_tags_and_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(InvalidMessage):
            decode(["object", "anything"])
        with self.assertRaises(InvalidMessage):
            loads(b'{"status":"returned","status":"passed"}')

    def test_shared_container_graph_is_not_silently_copied(self):
        item = []
        with self.assertRaises(UnsupportedValue):
            encode(((item,), {"other": item}))

    def test_excessive_depth_is_rejected_before_json_decoding(self):
        with self.assertRaises(InvalidMessage):
            loads(b"[" * 1000 + b"0" + b"]" * 1000)
