"""Tests for llm-response-parser."""

from __future__ import annotations

from llm_response_parser import (
    ParseResult,
    parse_json,
    parse_key_value,
    parse_list,
    parse_sections,
)

# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------


def test_parse_result_success():
    r = ParseResult.success({"key": "val"}, '{"key":"val"}')
    assert r.ok is True
    assert r.value == {"key": "val"}
    assert r.raw == '{"key":"val"}'


def test_parse_result_failure():
    r = ParseResult.failure()
    assert r.ok is False
    assert r.value is None


def test_parse_result_failure_default():
    r = ParseResult.failure([])
    assert r.value == []


# ---------------------------------------------------------------------------
# parse_json — fenced blocks
# ---------------------------------------------------------------------------


def test_parse_json_fenced_object():
    text = 'Here is the result:\n```json\n{"name": "Alice", "age": 30}\n```'
    r = parse_json(text)
    assert r.ok
    assert r.value == {"name": "Alice", "age": 30}


def test_parse_json_fenced_no_lang():
    text = "```\n[1, 2, 3]\n```"
    r = parse_json(text)
    assert r.ok
    assert r.value == [1, 2, 3]


def test_parse_json_fenced_array():
    text = "```json\n[true, false, null]\n```"
    r = parse_json(text)
    assert r.ok
    assert r.value == [True, False, None]


def test_parse_json_multiple_blocks_first_wins():
    text = '```json\n{"a": 1}\n```\n\n```json\n{"b": 2}\n```'
    r = parse_json(text)
    assert r.ok
    assert r.value == {"a": 1}


# ---------------------------------------------------------------------------
# parse_json — inline code
# ---------------------------------------------------------------------------


def test_parse_json_inline_object():
    text = 'The answer is `{"x": 42}`.'
    r = parse_json(text)
    assert r.ok
    assert r.value == {"x": 42}


# ---------------------------------------------------------------------------
# parse_json — raw scan
# ---------------------------------------------------------------------------


def test_parse_json_raw_object():
    text = 'Use this config: {"host": "localhost", "port": 8080}'
    r = parse_json(text)
    assert r.ok
    assert r.value["port"] == 8080


def test_parse_json_raw_array():
    text = "Items: [1, 2, 3] — done."
    r = parse_json(text)
    assert r.ok
    assert r.value == [1, 2, 3]


def test_parse_json_raw_array_before_object():
    # The array appears first in the text, so it must be returned, not the
    # later object.
    text = 'The list is [1, 2, 3] and the config is {"a": 1}'
    r = parse_json(text)
    assert r.ok
    assert r.value == [1, 2, 3]


def test_parse_json_raw_object_before_array():
    text = 'config {"a": 1} then list [4, 5, 6]'
    r = parse_json(text)
    assert r.ok
    assert r.value == {"a": 1}


def test_parse_json_raw_unbalanced_object_falls_back_to_array():
    # A stray "{" precedes a valid array; the scanner should skip the
    # unbalanced object candidate and return the array.
    text = "broken { not json here, but [1, 2, 3] is valid"
    r = parse_json(text)
    assert r.ok
    assert r.value == [1, 2, 3]


def test_parse_json_no_json():
    r = parse_json("This is plain text with no JSON.")
    assert not r.ok
    assert r.value is None


def test_parse_json_default_on_failure():
    r = parse_json("no json here", default={})
    assert not r.ok
    assert r.value == {}


def test_parse_json_nested_object():
    text = '```json\n{"outer": {"inner": [1, 2]}}\n```'
    r = parse_json(text)
    assert r.ok
    assert r.value["outer"]["inner"] == [1, 2]


def test_parse_json_raw_field():
    text = '```json\n{"k": "v"}\n```'
    r = parse_json(text)
    assert r.raw is not None
    assert '"k"' in r.raw


# ---------------------------------------------------------------------------
# parse_list — bullets
# ---------------------------------------------------------------------------


def test_parse_list_dash_bullets():
    text = "Things to do:\n- Buy milk\n- Write tests\n- Ship it"
    r = parse_list(text)
    assert r.ok
    assert r.value == ["Buy milk", "Write tests", "Ship it"]


def test_parse_list_star_bullets():
    text = "* Alpha\n* Beta\n* Gamma"
    r = parse_list(text)
    assert r.ok
    assert r.value == ["Alpha", "Beta", "Gamma"]


def test_parse_list_plus_bullets():
    text = "+ one\n+ two"
    r = parse_list(text)
    assert r.ok
    assert r.value == ["one", "two"]


# ---------------------------------------------------------------------------
# parse_list — numbered
# ---------------------------------------------------------------------------


def test_parse_list_numbered_dot():
    text = "1. First\n2. Second\n3. Third"
    r = parse_list(text)
    assert r.ok
    assert r.value == ["First", "Second", "Third"]


def test_parse_list_numbered_paren():
    text = "1) Alpha\n2) Beta"
    r = parse_list(text)
    assert r.ok
    assert r.value == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# parse_list — strip
# ---------------------------------------------------------------------------


def test_parse_list_strips_whitespace():
    text = "-  lots of space  "
    r = parse_list(text)
    assert r.value == ["lots of space"]


def test_parse_list_no_strip():
    text = "- trailing space   "
    r = parse_list(text, strip=False)
    assert r.value[0].endswith("   ")


# ---------------------------------------------------------------------------
# parse_list — no list
# ---------------------------------------------------------------------------


def test_parse_list_no_items():
    r = parse_list("This is just a paragraph.")
    assert not r.ok
    assert r.value == []


# ---------------------------------------------------------------------------
# parse_key_value
# ---------------------------------------------------------------------------


def test_parse_kv_colon():
    text = "Name: Alice\nAge: 30\nCity: NYC"
    r = parse_key_value(text)
    assert r.ok
    assert r.value == {"Name": "Alice", "Age": "30", "City": "NYC"}


def test_parse_kv_equals():
    text = "host = localhost\nport = 8080"
    r = parse_key_value(text)
    assert r.ok
    assert r.value["port"] == "8080"


def test_parse_kv_lowercase_keys():
    text = "Name: Alice\nAGE: 30"
    r = parse_key_value(text, lowercase_keys=True)
    assert r.ok
    assert "name" in r.value
    assert "age" in r.value


def test_parse_kv_no_pairs():
    r = parse_key_value("Just some text here.")
    assert not r.ok
    assert r.value == {}


def test_parse_kv_mixed_prose():
    text = "The result is:\nStatus: success\nCode: 200\nSome trailing text."
    r = parse_key_value(text)
    assert r.ok
    assert r.value["Status"] == "success"
    assert r.value["Code"] == "200"


def test_parse_kv_with_spaces_in_key():
    text = "First Name: Bob"
    r = parse_key_value(text)
    assert r.ok
    assert "First Name" in r.value


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------


def test_parse_sections_basic():
    text = "## Introduction\nHello world.\n## Body\nMore text."
    r = parse_sections(text)
    assert r.ok
    assert "Introduction" in r.value
    assert r.value["Introduction"] == "Hello world."


def test_parse_sections_multiple():
    text = "## A\nContent A.\n## B\nContent B.\n## C\nContent C."
    r = parse_sections(text)
    assert r.ok
    assert list(r.value.keys()) == ["A", "B", "C"]


def test_parse_sections_preamble():
    text = "Intro text.\n## Section\nBody."
    r = parse_sections(text)
    assert r.ok
    assert "" in r.value
    assert r.value[""] == "Intro text."


def test_parse_sections_no_preamble_when_empty():
    text = "## Section\nBody."
    r = parse_sections(text)
    assert "" not in r.value


def test_parse_sections_custom_heading():
    text = "### A\nContent A.\n### B\nContent B."
    r = parse_sections(text, heading="###")
    assert r.ok
    assert "A" in r.value
    assert "B" in r.value


def test_parse_sections_h1():
    text = "# Title\nIntro.\n# Second\nMore."
    r = parse_sections(text, heading="#")
    assert r.ok
    assert "Title" in r.value


def test_parse_sections_no_headings():
    r = parse_sections("Just plain text with no headings.")
    assert not r.ok
    assert r.value == {}


def test_parse_sections_strips_body():
    text = "## Section\n\n  Body text.  \n\n"
    r = parse_sections(text)
    assert r.value["Section"] == "Body text."


def test_parse_sections_no_strip():
    text = "## Section\n  Body.  "
    r = parse_sections(text, strip=False)
    # With strip=False the heading prefix is not stripped, so title has leading space
    # and body retains surrounding newlines/spaces
    title = " Section"  # "## Section"[len("##"):] without strip
    assert title in r.value
    assert r.value[title].startswith("\n")
