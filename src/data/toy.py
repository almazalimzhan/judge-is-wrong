"""Original, intentionally simple fixtures; these are not benchmark findings."""

from .base import Task, TestCase


def toy_tasks() -> list[Task]:
    specs = [
        (
            "add",
            "Implement add(a, b): return the sum of two integers.",
            ["assert add(2, 3) == 5"],
            [
                "assert add(0, 0) == 0",
                "assert add(-2, 3) == 1",
                "assert add(-4, -5) == -9",
                "assert add(10**40, 1) == 10**40 + 1",
            ],
        ),
        (
            "is_even",
            "Implement is_even(n): return a bool indicating whether integer n is even.",
            ["assert is_even(2) is True"],
            [
                "assert is_even(0) is True",
                "assert is_even(-2) is True",
                "assert is_even(3) is False",
                "assert is_even(-1) is False",
            ],
        ),
        (
            "reverse_text",
            "Implement reverse_text(text): reverse a Unicode string, preserving all characters.",
            ["assert reverse_text('abc') == 'cba'"],
            [
                "assert reverse_text('') == ''",
                "assert reverse_text('a') == 'a'",
                "assert reverse_text('а🙂b') == 'b🙂а'",
                "assert reverse_text(' a ') == ' a '",
            ],
        ),
        (
            "factorial",
            "Implement factorial(n) for a nonnegative integer n. In particular, factorial(0) is 1.",
            ["assert factorial(3) == 6"],
            [
                "assert factorial(0) == 1",
                "assert factorial(1) == 1",
                "assert factorial(5) == 120",
                "assert factorial(10) == 3628800",
            ],
        ),
        (
            "max_or_none",
            "Implement max_or_none(values): maximum of a list of integers, or None for an empty list.",
            ["assert max_or_none([1, 4, 2]) == 4"],
            [
                "assert max_or_none([]) is None",
                "assert max_or_none([-8, -2]) == -2",
                "assert max_or_none([0]) == 0",
                "assert max_or_none([7, 7]) == 7",
            ],
        ),
        (
            "deduplicate",
            "Implement deduplicate(values): unique integers from a list, preserving first occurrence order.",
            ["assert deduplicate([1, 2, 1]) == [1, 2]"],
            [
                "assert deduplicate([]) == []",
                "assert deduplicate([3, 1, 3, 2]) == [3, 1, 2]",
                "assert deduplicate([-1, -1, 0]) == [-1, 0]",
                "assert deduplicate([5]) == [5]",
            ],
        ),
        (
            "clamp",
            "Implement clamp(value, low, high): clamp an integer inclusively to [low, high], assuming low <= high.",
            ["assert clamp(5, 1, 8) == 5"],
            [
                "assert clamp(-1, 0, 2) == 0",
                "assert clamp(8, 0, 2) == 2",
                "assert clamp(3, 3, 3) == 3",
                "assert clamp(-3, -5, -1) == -3",
            ],
        ),
        (
            "count_vowels",
            "Implement count_vowels(text): count the ASCII vowels a,e,i,o,u, case-insensitively in a string.",
            ["assert count_vowels('hello') == 2"],
            [
                "assert count_vowels('AEIOU') == 5",
                "assert count_vowels('') == 0",
                "assert count_vowels('rhythm') == 0",
                "assert count_vowels('aA🙂e') == 3",
            ],
        ),
        (
            "is_palindrome",
            "Implement is_palindrome(text): exact case-sensitive Unicode palindrome test; the empty string is a palindrome.",
            ["assert is_palindrome('aba') is True"],
            [
                "assert is_palindrome('') is True",
                "assert is_palindrome('Aa') is False",
                "assert is_palindrome('🙂a🙂') is True",
                "assert is_palindrome('ab') is False",
            ],
        ),
        (
            "binary_search",
            "Implement binary_search(values, target): index of target in a sorted list of unique integers, or -1 if absent.",
            ["assert binary_search([1, 3, 5], 3) == 1"],
            [
                "assert binary_search([], 1) == -1",
                "assert binary_search([2], 2) == 0",
                "assert binary_search([1, 3, 5], 5) == 2",
                "assert binary_search([1, 3, 5], 2) == -1",
            ],
        ),
    ]
    return [
        Task(
            f"toy/{name}",
            problem,
            name,
            tuple(TestCase(test) for test in public),
            tuple(TestCase(test) for test in hidden),
            {"source": "original synthetic fixture", "license": "MIT", "synthetic": True},
        )
        for name, problem, public, hidden in specs
    ]
