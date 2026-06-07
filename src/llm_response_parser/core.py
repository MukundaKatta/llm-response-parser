"""Extract structured data from LLM text responses.

LLMs often return prose mixed with structured data — JSON in fenced code
blocks, bulleted lists, numbered lists, key-value pairs, or markdown sections.
This module provides five zero-dependency helpers that pull out the structured
parts without forcing the caller to write ad-hoc regex each time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------


@dataclass
class ParseResult:
    """Outcome of a parse operation.

    Attributes:
        value: The extracted data (type depends on the parser used).
        raw: The portion of the input string that was matched, or ``None``
            if the parser operates on the whole string.
        ok: ``True`` when the parse succeeded.
    """

    value: Any
    raw: str | None
    ok: bool

    @classmethod
    def success(cls, value: Any, raw: str | None = None) -> ParseResult:
        """Create a successful result."""
        return cls(value=value, raw=raw, ok=True)

    @classmethod
    def failure(cls, value: Any = None) -> ParseResult:
        """Create a failed result with an optional default value."""
        return cls(value=value, raw=None, ok=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Fenced code block: ```[lang]\n...\n```  (lang is optional)
_FENCED_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Inline code block: `...`
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def _find_json_object_or_array(text: str) -> str | None:
    """Return the first JSON object or array found in *text*, or ``None``.

    "First" means the candidate whose opening delimiter (``{`` or ``[``)
    appears earliest in *text*, so an array that precedes an object is
    returned ahead of that object.
    """
    candidates = []
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start != -1:
            candidates.append((start, start_char, end_char))
    # Try delimiters in the order they appear in the text.
    candidates.sort()

    for start, start_char, end_char in candidates:
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# parse_json
# ---------------------------------------------------------------------------


def parse_json(
    text: str,
    *,
    default: Any = None,
) -> ParseResult:
    """Extract and parse the first JSON value from *text*.

    Search order:
    1. Fenced code block (```json ... ``` or plain ``` ... ```)
    2. Inline code block (`` `...` ``)
    3. Raw text — scan for the first ``{`` or ``[`` and balance braces

    Args:
        text: The LLM response string.
        default: Value to use as ``ParseResult.value`` on failure.

    Returns:
        :class:`ParseResult` with ``value`` as the parsed Python object and
        ``raw`` as the matched JSON string.  ``ok=False`` if nothing parses.
    """
    # 1. Fenced blocks
    for match in _FENCED_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            return ParseResult.success(json.loads(raw), raw)
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Inline code
    for match in _INLINE_CODE_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            return ParseResult.success(json.loads(raw), raw)
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Raw scan
    raw = _find_json_object_or_array(text)
    if raw is not None:
        try:
            return ParseResult.success(json.loads(raw), raw)
        except (json.JSONDecodeError, ValueError):
            pass

    return ParseResult.failure(default)


# ---------------------------------------------------------------------------
# parse_list
# ---------------------------------------------------------------------------

# Bullet: "- item" or "* item" or "+ item"
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+(.+)", re.MULTILINE)
# Numbered: "1. item" or "1) item"
_NUMBERED_RE = re.compile(r"^[ \t]*\d+[.)]\s+(.+)", re.MULTILINE)


def parse_list(
    text: str,
    *,
    strip: bool = True,
) -> ParseResult:
    """Extract list items from *text*.

    Detects either bullet lists (``-``, ``*``, ``+``) or numbered lists
    (``1.``, ``2)``, etc.).  Bullet lists are preferred when both appear.

    Args:
        text: The LLM response string.
        strip: Strip leading/trailing whitespace from each item.

    Returns:
        :class:`ParseResult` where ``value`` is a ``list[str]``.
        ``ok=False`` if no items were found (``value`` is ``[]``).
    """
    bullets = _BULLET_RE.findall(text)
    if bullets:
        items = [item.strip() if strip else item for item in bullets]
        return ParseResult.success(items)

    numbered = _NUMBERED_RE.findall(text)
    if numbered:
        items = [item.strip() if strip else item for item in numbered]
        return ParseResult.success(items)

    return ParseResult.failure([])


# ---------------------------------------------------------------------------
# parse_key_value
# ---------------------------------------------------------------------------

# "Key: value" or "Key = value" (optional leading whitespace)
_KV_RE = re.compile(r"^[ \t]*([A-Za-z_][\w\s\-]*)[:=][ \t]*(.+)", re.MULTILINE)


def parse_key_value(
    text: str,
    *,
    lowercase_keys: bool = False,
    strip: bool = True,
) -> ParseResult:
    """Extract ``Key: value`` or ``Key = value`` pairs from *text*.

    Args:
        text: The LLM response string.
        lowercase_keys: If ``True``, normalise keys to lower-case.
        strip: Strip surrounding whitespace from keys and values.

    Returns:
        :class:`ParseResult` where ``value`` is a ``dict[str, str]``.
        ``ok=False`` if no pairs were found (``value`` is ``{}``).
    """
    pairs: dict[str, str] = {}
    for match in _KV_RE.finditer(text):
        key = match.group(1)
        val = match.group(2)
        if strip:
            key = key.strip()
            val = val.strip()
        if lowercase_keys:
            key = key.lower()
        if key:
            pairs[key] = val

    if pairs:
        return ParseResult.success(pairs)
    return ParseResult.failure({})


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------


def parse_sections(
    text: str,
    *,
    heading: str = "##",
    strip: bool = True,
) -> ParseResult:
    """Split *text* into named sections delimited by markdown headings.

    Args:
        text: The LLM response string.
        heading: The heading prefix to split on (default ``"##"``).
            Supports any prefix such as ``"#"``, ``"##"``, ``"###"``.
        strip: Strip surrounding whitespace from section titles and bodies.

    Returns:
        :class:`ParseResult` where ``value`` is an ``OrderedDict``-like
        ``dict[str, str]`` mapping section titles to section bodies.
        Preamble text before the first heading is stored under the key
        ``""`` (empty string) only if it is non-empty after stripping.
        ``ok=False`` when no headings match (``value`` is ``{}``).
    """
    pattern = re.compile(
        r"^" + re.escape(heading) + r"[^#\n][^\n]*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return ParseResult.failure({})

    sections: dict[str, str] = {}

    # Preamble before the first heading
    preamble = text[: matches[0].start()]
    if strip:
        preamble = preamble.strip()
    if preamble:
        sections[""] = preamble

    for i, match in enumerate(matches):
        raw_title = match.group(0)[len(heading) :]
        title = raw_title.strip() if strip else raw_title
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        if strip:
            body = body.strip()
        sections[title] = body

    return ParseResult.success(sections)
