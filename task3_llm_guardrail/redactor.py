"""Streaming PII redaction with a bounded hold-back buffer.

The problem: a secret can be split across deltas ("4111 11", "11 1111 1111"), so
a chunk cannot be released just because it looks clean on its own. Buffering the
whole response would fix that but ruins latency and memory.

The approach rests on one observation. Every pattern here is a contiguous run of
either email characters or digits-and-separators. So only the run touching the
end of the buffer could still grow into a match:

    "Contact me at john.doe@exa"
                   ^^^^^^^^^^^^  held: may still become an email
     ^^^^^^^^^^^^^^              released now

Text before that run can never join a future match, so it is scanned and released
straight away. The hold is normally one word, and it is capped, so memory is O(1)
in response length. Finding the run walks backward from the end of the buffer, so
the cost per delta depends on the held text and not on the response so far.
"""

from __future__ import annotations

import re
from typing import Final

REPLACEMENT: Final = "[REDACTED]"

# Hold limits. An address is at most 254 characters (RFC 5321). A card is at most
# 19 digits plus separators. A longer run cannot be one match, so releasing the
# excess is safe.
MAX_EMAIL_HOLD: Final = 320
MAX_NUMBER_HOLD: Final = 64

_EMAIL_CHARS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-@"
)
_NUMBER_CHARS: Final = frozenset("0123456789 -")
_DIGITS: Final = frozenset("0123456789")

# One alternation, so each delta costs a single pass. The local part is capped at
# 64 (the RFC limit), which also rules out quadratic backtracking on a long run of
# email characters with no "@".
_PII: Final = re.compile(
    r"""
    (?P<email>[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,})
    | (?P<ssn>(?<![0-9])[0-9]{3}(?P<sep>[ \-])[0-9]{2}(?P=sep)[0-9]{4}(?![0-9]))
    | (?P<card>(?<![0-9])[0-9](?:[ \-]?[0-9]){12,18}(?![0-9]))
    """,
    re.VERBOSE,
)


def luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if position % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


class StreamRedactor:
    """Feed text deltas in, get safe text out. One instance per text stream.

    ``require_luhn`` trades recall for precision. It defaults to False because a
    guardrail should fail closed: a mistyped or partial card is still sensitive,
    and redacting an order number by accident costs far less than leaking a card.
    """

    __slots__ = ("_buffer", "_require_luhn", "redactions")

    def __init__(self, *, require_luhn: bool = False) -> None:
        self._buffer = ""
        self._require_luhn = require_luhn
        self.redactions: dict[str, int] = {}

    @property
    def held(self) -> int:
        return len(self._buffer)

    def feed(self, delta: str) -> str:
        """Add a delta and return whatever is now provably safe to release."""
        self._buffer += delta
        return self._release(self._hold_start())

    def flush(self) -> str:
        """End of stream: nothing can grow any further, so release the rest."""
        return self._release(len(self._buffer))

    def _hold_start(self) -> int:
        buffer = self._buffer
        end = len(buffer)

        email_start = end
        floor = max(0, end - MAX_EMAIL_HOLD)
        while email_start > floor and buffer[email_start - 1] in _EMAIL_CHARS:
            email_start -= 1

        number_start = end
        floor = max(0, end - MAX_NUMBER_HOLD)
        while number_start > floor and buffer[number_start - 1] in _NUMBER_CHARS:
            number_start -= 1
        # Spaces and dashes before the first digit cannot begin a number, and
        # without this every space at the end of a delta would stall the stream.
        while number_start < end and buffer[number_start] not in _DIGITS:
            number_start += 1

        return min(email_start, number_start)

    def _release(self, hold_start: int) -> str:
        buffer = self._buffer
        pieces: list[str] = []
        position = 0
        for match in _PII.finditer(buffer):
            if match.start() >= hold_start:
                break
            kind = match.lastgroup or "unknown"
            if kind == "card" and self._require_luhn:
                digits = match.group().replace(" ", "").replace("-", "")
                if not luhn_valid(digits):
                    continue
            pieces.append(buffer[position : match.start()])
            pieces.append(REPLACEMENT)
            self.redactions[kind] = self.redactions.get(kind, 0) + 1
            position = match.end()

        # A match that started before hold_start may reach past it, when a capped
        # run was cut mid-match. It was redacted whole, so resume after it.
        cut = max(position, hold_start)
        pieces.append(buffer[position:cut])
        self._buffer = buffer[cut:]
        return "".join(pieces)


def redact_text(text: str, *, require_luhn: bool = False) -> str:
    """One-shot redaction for non-streaming responses."""
    redactor = StreamRedactor(require_luhn=require_luhn)
    return redactor.feed(text) + redactor.flush()
