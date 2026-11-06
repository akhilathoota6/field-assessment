from __future__ import annotations

import importlib.util
import random
import sys
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("redactor", Path(__file__).with_name("redactor.py"))
redactor = importlib.util.module_from_spec(spec)
sys.modules["redactor"] = redactor
spec.loader.exec_module(redactor)

StreamRedactor = redactor.StreamRedactor
redact_text = redactor.redact_text
R = redactor.REPLACEMENT

SAMPLES = [
    "Email john.doe+tag@sub.example.co.uk now.",
    "Mail a@b.io, then c.d@e-f.org; done",
    "SSN 123-45-6789 and also 987 65 4321.",
    "Cards 4111 1111 1111 1111, 5500-0000-0000-0004 and 378282246310005!",
    "Mixed x@y.com 123-45-6789 4111111111111111 z@w.net",
    "(4111111111111111)",
    "no secrets here, just 42 words and 2024-01-15 dates",
    "phone 415-555-2671 stays",
    "ends with email a@b.com",
    "ends with ssn 123-45-6789",
    "ends with card 4111 1111 1111 1111",
    "unicode café 日本語 🙂 a@b.com 🙂",
    "a@b.com.",
    "trailing dash 123-45-6789- and more",
]


def stream(text, sizes):
    r = StreamRedactor()
    out, position = [], 0
    for size in sizes:
        out.append(r.feed(text[position : position + size]))
        position += size
    out.append(r.feed(text[position:]))
    out.append(r.flush())
    return "".join(out)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail john@example.com now", f"mail {R} now"),
        ("ssn 123-45-6789 ok", f"ssn {R} ok"),
        ("ssn 123 45 6789 ok", f"ssn {R} ok"),
        ("card 4111 1111 1111 1111 ok", f"card {R} ok"),
        ("card 4111-1111-1111-1111 ok", f"card {R} ok"),
        ("card 4111111111111111 ok", f"card {R} ok"),
        ("amex 378282246310005 ok", f"amex {R} ok"),
        ("card 1234 5678 9012 3456 ok", f"card {R} ok"),  # fails Luhn, still masked
        ("a@b.com and c@d.org", f"{R} and {R}"),
        ("the end a@b.com.", f"the end {R}."),
    ],
)
def test_patterns(text, expected):
    assert redact_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "version 1.2.3 is out",
        "dated 2024-01-15 today",
        "call 415-555-2671",
        "ids 123-456-7890",
        "1234567890123456789012",  # too long to be a card
        "12-345-6789",  # not SSN grouping
        "123-45 6789",  # mixed separators
        "not an email: foo@bar",
        "@handle and user@",
    ],
)
def test_lookalikes_survive(text):
    assert redact_text(text) == text


def test_luhn_mode_keeps_invalid_numbers_and_masks_real_cards():
    assert redact_text("n 1234 5678 9012 3456", require_luhn=True) == "n 1234 5678 9012 3456"
    assert redact_text("n 4111 1111 1111 1111", require_luhn=True) == f"n {R}"


@pytest.mark.parametrize("text", SAMPLES)
def test_every_two_way_split_matches_one_shot(text):
    expected = redact_text(text)
    for cut in range(len(text) + 1):
        assert stream(text, [cut]) == expected, f"split at {cut}"


@pytest.mark.parametrize("text", SAMPLES)
def test_character_by_character_matches_one_shot(text):
    assert stream(text, [1] * len(text)) == redact_text(text)


def test_random_chunking_fuzz():
    rng = random.Random(1234)
    pieces = [
        "john@example.com", "123-45-6789", "4111 1111 1111 1111", "hello", " ", " ", ", ",
        ".", "-", "42", "foo@", "@bar", "2024", "world", "\n", "a.b", "1 2 3",
    ]  # fmt: skip
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 25)))
        sizes = [rng.randint(0, 9) for _ in range(rng.randint(0, 30))]
        assert stream(text, sizes) == redact_text(text), repr(text)


@pytest.mark.parametrize("secret", ["john.doe@example.com", "123-45-6789", "4111 1111 1111 1111"])
def test_no_fragment_of_a_secret_is_ever_released(secret):
    """Inspect every emitted piece, not just the final text."""
    r = StreamRedactor()
    emitted = ""
    for char in f"value: {secret} end":
        emitted += r.feed(char)
        assert not any(emitted.endswith(secret[:n]) for n in range(4, len(secret) + 1))
    emitted += r.flush()
    assert emitted == f"value: {R} end"


def test_clean_prose_is_released_within_one_word():
    r = StreamRedactor()
    assert r.feed("Hello there") == "Hello "
    assert r.held == len("there")
    assert r.feed(" friend") == "there "


def test_trailing_space_does_not_stall_the_stream():
    r = StreamRedactor()
    assert r.feed("Hello world ") == "Hello world "
    assert r.held == 0


def test_flush_releases_everything():
    r = StreamRedactor()
    assert r.feed("tail") == ""
    assert r.flush() == "tail"
    assert r.flush() == ""


@pytest.mark.parametrize("unit", ["word ", "a", "7", "1-", "user@", "a.", " "])
def test_memory_stays_bounded_on_a_long_stream(unit):
    r = StreamRedactor()
    peak = 0
    for _ in range(10_000):
        r.feed(unit)
        peak = max(peak, r.held)
    assert peak <= redactor.MAX_EMAIL_HOLD + len(unit)


def test_a_card_straddling_the_forced_cut_is_not_leaked():
    # Long separated digit run: the hold cap has to cut somewhere in the middle.
    text = "4111 1111 1111 1111 " * 40
    out = stream(text, [7] * (len(text) // 7))
    assert "4111 1111 1111 1111" not in out


def test_redaction_counts_never_store_values():
    r = StreamRedactor()
    r.feed("a@b.com 123-45-6789 4111111111111111 x@y.org ")
    r.flush()
    assert r.redactions == {"email": 2, "ssn": 1, "card": 1}


def test_adversarial_input_stays_linear():
    # Would take far longer if the regex backtracked quadratically.
    start = time.perf_counter()
    redact_text("a" * 500_000)
    redact_text("a." * 250_000 + "@")
    redact_text("1 " * 250_000)
    assert time.perf_counter() - start < 5


def test_throughput_is_far_above_any_model_output_rate():
    r = StreamRedactor()
    tokens = ["Hello", " there", ",", " this", " is", " ordinary", " text", "."] * 12_500
    start = time.perf_counter()
    for token in tokens:
        r.feed(token)
    elapsed = time.perf_counter() - start
    assert len(tokens) / elapsed > 50_000  # tokens per second
