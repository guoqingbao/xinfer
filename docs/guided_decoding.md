# Guided Decoding

This document describes the current guided decoding implementation in `xinfer`.

It focuses on:
- the live request-to-sampling workflow
- the supported request surfaces
- how reasoning effort is applied
- practical usage and validation commands

## Current Model

Guided decoding is request-scoped.

The core engine does not invent grammars on its own. A request either:
- supplies a constraint grammar
- gets that constraint grammar composed with a reasoning prefix
- or runs unconstrained when neither exists

The final grammar is stored in `SamplingParams.grammar` and consumed by the runner.

## End-to-End Workflow

### 1. Request parsing

OpenAI-compatible requests can provide a grammar through:
- `structured_outputs`
- `response_format`
- legacy `constraint` plus `constraint_type`

Relevant code:
- `src/server/mod.rs`
- `src/server/server.rs`

The server converts those request fields into `TopLevelGrammar`.

Only one constraint source may be set at a time.

### 2. Grammar composition

The server composes:
- request constraint grammar
- optional reasoning grammar prefix

The result is a single `TopLevelGrammar` assigned to `SamplingParams.grammar`.

If no client-supplied constraint grammar exists, `params.grammar` stays `None`.

### 3. Sampling in runner

The runner uses the standard llguidance loop:

1. build or reuse `GuidanceState` from `SamplingParams.grammar`
2. call `compute_mask_or_eos()`
3. mask logits
4. apply penalties on masked logits
5. sample once
6. `commit_token()` into matcher state

Failures are fail-soft per sequence:
- on matcher creation failure, mask failure, or commit failure, guidance is disabled for that sequence
- normal generation continues for the affected sequence

Relevant code:
- `src/core/runner.rs`
- `src/utils/guidance.rs`

## Supported Request Surfaces

### OpenAI-compatible request fields

`structured_outputs`
- `choice`
- `regex`
- `json`
- `grammar`
- `structural_tag`

`response_format`
- `json_schema`
- `json_object`

Legacy fields
- `constraint`
- `constraint_type = regex | lark | json_schema | json`

If a request provides none of the above, guided decoding is not enabled.

### Claude server

Current state:
- Claude does not expose the same client-supplied grammar request surface as the OpenAI endpoint
- Claude reasoning is still driven by explicit thinking behavior, not by `reasoning_effort` grammar composition
- Claude requests therefore do not currently enable guided decoding

## Reasoning Effort

Reasoning effort is separate from ordinary structured outputs.

### Current state

The OpenAI path maps `reasoning_effort` into grammar composition when a request constraint exists.

Accepted values come from `ReasoningEffort::from_str`:
- `none`
- `low`
- `medium`
- `normal`
- `high`
- `chain_of_thought`
- `cot`
- `cove`

Non-Python builds also support:
- `custom:<template>`

Relevant code:
- `src/utils/reasoning.rs`
- `src/server/server.rs`
- `src/utils/guidance.rs`

### Important behavior

Reasoning effort:
- does not enable chat-template thinking by itself
- only affects grammar composition
- is ignored when no request constraint grammar is present
- only works when reasoning start/end tokens are available

If the tokenizer does not expose reasoning markers, the system logs a warning and falls back to the base grammar.

### Why this is separate

Keep these concerns separate:
- `thinking`: prompt/template behavior
- `reasoning_effort`: guided-decoding grammar behavior

Server default:
- when a request omits `thinking` / `enable_thinking`, the Rust CLI defaults to thinking enabled
- start the Rust server with `--disable-reasoning` to change only that default fallback
- explicit request values still win over the CLI default
- `reasoning_effort` may still set `thinking=true` when guided decoding normalization is active

That separation avoids changing prompt rendering for request types that do not support OpenAI-style reasoning controls.

## Use Cases

Present guided decoding from the simplest use case to the most constrained one.

### Running the server
```shell
xinfer --m Qwen/Qwen3.5-35B-A3B-FP8/ --ui-server --d 0
```

### Quick map

| Goal | Request surface | Typical type |
|------|-----------------|--------------|
| Pick one label | `structured_outputs` | `choice` |
| Enforce text pattern | `structured_outputs` or `constraint` | `regex` |
| Enforce full object schema | `structured_outputs` or `response_format` | `json` / `json_schema` |
| Enforce custom grammar | `structured_outputs` or `constraint` | `grammar` / `lark` |
| Constrain tagged structured output payload | `structured_outputs` | `structural_tag` |
| Add a reasoning prefix | `reasoning_effort` | `low`, `medium`, `high`, etc. |

### 1. Constrain the answer to a fixed set

Use this when the model must pick exactly one label.

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Classify this sentiment"}],
    "structured_outputs": {
      "choice": ["positive", "negative", "neutral"]
    },
    "max_tokens": 50
  }'
```

Expected behavior:
- output is one of the listed strings
- server log shows `source=structured_outputs type=choice`

### 2. Constrain text with regex

Use this when the output is text, but must follow a narrow textual pattern.

`structured_outputs.regex` example:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Return a 3-digit code"}],
    "structured_outputs": {
      "regex": "^[0-9]{3}$"
    },
    "max_tokens": 10
  }'
```

Expected behavior:
- output matches the regex
- server log shows `source=structured_outputs type=regex`

### 3. Constrain to a JSON object with a schema

Use this when the model must produce machine-parseable JSON.

Exact example:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Generate a user profile"}],
    "structured_outputs": {
      "json": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer", "minimum": 0, "maximum": 150},
          "email": {"type": "string", "pattern": "^[a-z]+@[a-z]+\\.[a-z]+$"}
        },
        "required": ["name", "age", "email"],
        "additionalProperties": false
      }
    },
    "max_tokens": 500
  }'
```

Expected behavior:
- output is valid JSON
- required fields are present
- extra properties are rejected by grammar

### 4. Use OpenAI-style `response_format`

Use this when compatibility with clients that already emit `response_format` matters.

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Return a JSON object"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "schema": {
          "type": "object",
          "properties": {
            "answer": {"type": "string"}
          },
          "required": ["answer"],
          "additionalProperties": false
        }
      }
    }
  }'
```

Expected behavior:
- output follows the supplied schema
- server log shows `source=response_format type=json_schema`

`json_object` example:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Return any JSON object"}],
    "response_format": {
      "type": "json_object"
    },
    "max_tokens": 200
  }'
```

Expected behavior:
- output is a valid JSON object
- no custom property schema is enforced beyond object shape

### 5. Constrain with custom grammar

Use this when the output is not simple JSON schema and needs a formal grammar.

Regex example:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Return a 3-digit code"}],
    "constraint": "^[0-9]{3}$",
    "constraint_type": "regex",
    "max_tokens": 10
  }'
```

Lark example:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Return yes or no"}],
    "constraint": "start: \"yes\" | \"no\"",
    "constraint_type": "lark",
    "max_tokens": 10
  }'
```

Expected behavior:
- output satisfies the supplied grammar exactly
- server log shows `source=constraint type=regex` or `type=lark`

`structured_outputs.grammar` example:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Return yes or no"}],
    "structured_outputs": {
      "grammar": "start: \"yes\" | \"no\""
    },
    "max_tokens": 10
  }'
```

Expected behavior:
- output follows the supplied Lark grammar
- server log shows `source=structured_outputs type=grammar`

### 6. Constrain structural-tag output

Use this when a request wants tagged structured content instead of plain JSON.

Example:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Emit a search tool call"}],
    "structured_outputs": {
      "structural_tag": {
        "tag": "tool_call",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string", "enum": ["search"]},
            "arguments": {
              "type": "object",
              "properties": {
                "query": {"type": "string"}
              },
              "required": ["query"],
              "additionalProperties": false
            }
          },
          "required": ["name", "arguments"],
          "additionalProperties": false
        }
      }
    },
    "max_tokens": 200
  }'
```

Expected behavior:
- output follows the tagged structural envelope
- schema rules still constrain the payload

### 7. Add reasoning effort on top of a constraint

Use this when you want the output constrained, but still want the model to emit a reasoning block first when the tokenizer supports reasoning markers.

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Classify this sentiment and think first: I am happy!"}],
    "structured_outputs": {
      "choice": ["positive", "negative", "neutral"]
    },
    "reasoning_effort": "low",
    "max_tokens": 1000
  }'
```

Expected behavior:
- the base answer remains grammar-constrained
- reasoning grammar is prefixed only when reasoning tokens are available
- if reasoning tokens are missing, the server logs a warning and falls back


### Conflict rejection

Run:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"hi"}],
    "structured_outputs": {"choice": ["a", "b"]},
    "response_format": {
      "type": "json_schema",
      "json_schema": {"schema": {"type": "object"}}
    }
  }'
```

Check:
- request is rejected
- error says only one of `structured_outputs`, `response_format`, or `constraint` may be set

### Unsupported `response_format` rejection

Run:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"hi"}],
    "response_format": {
      "type": "xml"
    }
  }'
```

Check:
- request is rejected
- error mentions only `json_schema` and `json_object` are supported

### Empty `structured_outputs` rejection

Run:

```bash
curl -sXPOST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"hi"}],
    "structured_outputs": {}
  }'
```

Check:
- request is rejected
- error says one of `choice`, `regex`, `json`, `grammar`, or `structural_tag` must be set

### Reasoning effort fallback

Run the reasoning-effort example above on a model without reasoning tokens.

Check:
- request still completes
- base constraint still applies
- warning indicates reasoning tokens were not found


## Known Boundaries

- Guided decoding is only active when `SamplingParams.grammar` is present.
- OpenAI currently has the richest client-facing grammar surface.
- Claude does not currently expose request-level guided decoding.
- No request-level grammar means no guided decoding.
