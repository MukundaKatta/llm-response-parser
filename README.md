# llm-response-parser

Extract structured data from LLM text responses.

Five zero-dependency helpers that pull structured content out of prose: JSON from code blocks, bullet/numbered lists, key-value pairs, and markdown sections.

## Install

```bash
pip install llm-response-parser
```

## Quick start

```python
from llm_response_parser import parse_json, parse_list, parse_key_value, parse_sections

# JSON in a fenced code block
r = parse_json('Here is the data:\n```json\n{"name": "Alice", "age": 30}\n```')
r.ok    # True
r.value # {"name": "Alice", "age": 30}

# Bullet list
r = parse_list("Things to do:\n- Buy milk\n- Write tests\n- Ship it")
r.value # ["Buy milk", "Write tests", "Ship it"]

# Key-value pairs
r = parse_key_value("Status: success\nCode: 200")
r.value # {"Status": "success", "Code": "200"}

# Markdown sections
r = parse_sections("## Summary\nShort. ## Details\nLong.")
r.value # {"Summary": "Short.", "Details": "Long."}
```

## API

All functions return a `ParseResult`:

```python
result.ok     # bool — True on success
result.value  # extracted data (type depends on parser)
result.raw    # matched string, or None
```

### `parse_json(text, *, default=None)`

Finds the first JSON value. Search order: fenced code block → inline code → raw scan.

### `parse_list(text, *, strip=True)`

Extracts bullet (`-`, `*`, `+`) or numbered (`1.`, `1)`) list items.

### `parse_key_value(text, *, lowercase_keys=False, strip=True)`

Extracts `Key: value` or `Key = value` pairs into a `dict[str, str]`.

### `parse_sections(text, *, heading="##", strip=True)`

Splits text on markdown headings. Returns `dict[str, str]` mapping title → body.
Pre-heading preamble is stored under key `""` if non-empty.

## License

MIT
