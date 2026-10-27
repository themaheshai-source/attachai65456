# Pipeline forensic debugging guide

This project writes structured JSON events through the existing Fastify/Pino logger to standard output. It does not persist an independent diagnostic log file. Apply the retention/access-control policy of the process supervisor, container runtime, or deployment log collector; do not route full-text diagnostics to a third party unless that destination is explicitly approved.

## Configure a controlled local trace

The defaults keep metadata-only diagnostic events enabled and full text disabled:

```dotenv
PIPELINE_DEBUG_LOGS_ENABLED=true
PIPELINE_DEBUG_LOG_LEVEL=info
PIPELINE_DEBUG_LOG_FULL_TEXT=false
PIPELINE_DEBUG_LOG_MAX_TEXT_CHARS=12000
PIPELINE_DIAGNOSTICS=basic
```

For a short, controlled local investigation, enable bounded full-text events and detailed stage diagnostics before starting the server:

```dotenv
PIPELINE_DEBUG_LOGS_ENABLED=true
PIPELINE_DEBUG_LOG_LEVEL=debug
PIPELINE_DEBUG_LOG_FULL_TEXT=true
PIPELINE_DEBUG_LOG_MAX_TEXT_CHARS=200000
PIPELINE_DIAGNOSTICS=detailed
```

Then run the project with its configured development command and capture stdout locally, for example:

```sh
npm run dev 2>&1 | tee pipeline-debug.jsonl
```

`PIPELINE_DEBUG_LOG_FULL_TEXT=true` may write private questions, prompt text, retrieved evidence, model outputs, and validator inputs to local logs. Use only synthetic or specifically approved data, limit access to the file and terminal output, and delete diagnostic logs after the investigation. Turn it off immediately after the trace. Redaction of obvious credential patterns remains active. If any `captureText` field has `complete: false`, inspect `truncated`, `redactionsApplied`, `originalCharacterCount`, and `loggedCharacterCount`; do not treat the value as a complete snapshot. The stable SHA-256 is calculated over the original value to help detect mutation, but it is not a replacement for text.

The maximum is applied to each captured string independently. Even with the maximum configured, a single value longer than that limit or containing a redacted credential is not recorded in full. The application does not provide durable retention controls itself; retention is managed by the stdout/container logging backend.

## Find one request

Fastify request logs and pipeline events are newline-delimited JSON. Request IDs and trace IDs are generated/validated at request ingress. Copy the `traceId` from `pipeline.request.started` or the response headers if available, then filter the captured file:

```sh
jq -c 'select(.traceId == "<trace-id>")' pipeline-debug.jsonl
```

Inspect the service-level question, the exact assembled LLM payload and raw model output, validator boundary, per-rule decisions, and final selection:

```sh
jq -c 'select(.traceId == "<trace-id>" and (.event == "pipeline.input.received" or .event == "pipeline.llm.request.input" or .event == "pipeline.llm.response.raw" or .event == "pipeline.grounding.validator.input" or .event == "pipeline.grounding.validator.rule.completed" or .event == "pipeline.grounding.validator.output" or .event == "pipeline.answer.selection.forensic" or .event == "pipeline.fallback.text.selected" or .event == "pipeline.trace.service.summary" or .event == "pipeline.trace.summary" or .event == "pipeline.response.delivery"))' pipeline-debug.jsonl
```

Each validator invocation has a different `snapshotId` and an `attemptLabel` (`initial`, `corrected`, `stream_initial`, or `stream_corrected`). The `pipeline.grounding.validator.input` event is emitted from inside `validateGrounding` and contains the actual `question`, `answer`, and `evidence` arguments it received. Each text item has a `value` plus original/logged lengths, SHA-256, truncation, redaction, and completeness metadata. `value` is `null` while full-text logging is disabled. The `pipeline.grounding.validator.rule.completed` event identifies each rule's status (`passed`, `failed`, `skipped`, or `error`), its reason, and its specific examined inputs. The final validator event includes the acceptance decision, classification, and rejection reasons.

## Compare evidence and answer stages

- Compare `pipeline.retrieval.completed` / detailed retrieval events and `pipeline.answer.generation` context diagnostics with the evidence array in `pipeline.grounding.validator.input`.
- That validator event records the evidence supplied by the caller, the exact evidence received at validation entry, and whether their order/content matched. It also separately compares the validator evidence with the selected retrieval evidence (`prepared.found`). History-derived evidence can legitimately make those sets differ; inspect the accompanying refs and captured evidence instead of assuming such a difference is a defect.
- Compare `pipeline.llm.request.input` with `pipeline.llm.response.raw`, then `pipeline.answer.cleaned.forensic`. The LLM request event reflects the final payload assembled by the provider before HTTP transmission; authorization headers are not included.
- Use the initial validator `snapshotId`, `pipeline.grounding.correction.prompt`, corrected raw/cleaned answer events, and a new corrected validator snapshot to check correction behavior. Original and corrected decisions are never overwritten.
- `pipeline.fallback.text.selected` records why a fallback was chosen and whether it was validated. `pipeline.answer.selection.forensic` and `pipeline.response.delivery` record the selected/delivered response boundary. These delivery events mean the handler returned a value or the server attempted a stream write; they do **not** prove the browser received it. Client acknowledgement is not implemented.

## Distinguish likely failure sources

- **Retrieval issue:** selected chunks are irrelevant or incorrect before prompt construction.
- **Context construction issue:** selected evidence is correct, but evidence in the provider payload differs or is omitted.
- **Generation issue:** the actual LLM payload is correct but the raw answer is unsupported or incomplete.
- **Input-to-validator issue:** the cleaned answer or evidence differs between upstream and the exact validator input event.
- **Validator issue:** the exact arguments are adequate but a rule event identifies an unexpected comparison or reason.
- **Correction/fallback issue:** correction input or output introduces a new problem, or the correction remains rejected and selects the configured fallback.
- **Delivery issue:** selected final output differs from the route/stream delivery record, or a disconnect/failure occurs before the completion frame. A successful write is not an end-user receipt acknowledgement.

## Disable and clean up

Set `PIPELINE_DEBUG_LOG_FULL_TEXT=false` (recommended normal setting), optionally set `PIPELINE_DEBUG_LOGS_ENABLED=false` for all forensic events, and use `PIPELINE_DIAGNOSTICS=off` to disable pipeline events. Restart the process after changing environment values. Delete local capture files using the approved handling process and ensure any container/provider log retention does not preserve the sensitive trace longer than authorized.

## Known limits

- Full-text capture is deliberately bounded per field and still redacts recognizable credentials; incomplete captures are marked, not silently presented as complete.
- The validator is deterministic. Per-rule logs describe the code's actual comparisons, not an independent assessment of answer correctness.
- For a streamed provider, raw answer text means the assembled text from parseable token deltas, not the original raw network byte frames.
- `pipeline.trace.service.summary` captures the answer-generation/validation outcome inside the service. `pipeline.trace.summary` is the terminal HTTP/SSE boundary summary and includes the service summary plus delivery outcome. Neither proves that a browser received or rendered the response; client logs provide separate evidence when enabled.
- Percentiles/counters are process-local in-memory metrics, not durable telemetry.
