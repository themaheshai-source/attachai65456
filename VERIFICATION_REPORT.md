# AttachAI Pipeline Repair — Verification Report

## Root cause verified

The supplied `dashboard-prompt-preview` trace was consistent with the source:

- `src/services/advanced-rag.ts` did not normalize `tum koun ho` as an identity question. Its lexical terms therefore had no overlap with the identity/service chunks.
- The evidence evaluator derives coverage from normalized lexical terms and requires at least two matched terms plus 65% coverage for support. It returned `NOT_SUPPORTED` with zero coverage despite a relevant identity chunk appearing in the dense candidate set.
- `src/services/chat.ts` preview used `retrieval.decision.sufficient` as a pre-generation gate. The result skipped answer generation and selected the tenant fallback, exactly as shown in the provided log.
- Dense scores were used for candidate recall and rank fusion; their low, provider-specific values are not evidence support. The logged pricing-first ordering did not itself prove support, and the later identity chunk was never able to pass the lexical gate for the unnormalized phrase.

## Repair design

Identity is a narrow, deterministic tenant-configuration route, not an ungrounded LLM exception. `src/services/identity.ts` recognizes equivalent English and common Roman-Hindi forms and returns only the active tenant's configured assistant/company identity. Chat, preview, and streaming share this route; provider calls are skipped and the selection is observable. Retrieval normalization also maps these forms to stable identity/service terms, preserving a grounded retrieval path for related knowledge questions. Unsupported non-identity questions still follow the existing evidence gate and fallback behavior.

## Verification rounds

### Round 1 — implementation and focused regression

Command: `npx vitest run tests/advanced-rag.test.ts tests/chat-rag-regression.test.ts --reporter=verbose -t "identity"`

Result: PASS — 2 files, 3 tests passed, 18 tests skipped by the focused filter. It covered the reported spelling, preview, streaming token delivery, no provider call, and tenant isolation.

### Round 2 — broader regression attempt

Command: `npx vitest run tests/advanced-rag.test.ts tests/chat-rag-regression.test.ts --reporter=verbose`

Result: the new identity tests passed. Six existing baseline assertions failed: five expect mocked `embedding.embed` to have no explicitly passed `undefined` debug argument or expect evidence that the current baseline gate rejects; one existing correction test received zero provider calls for the same baseline evidence-gate reason. These failures were not changed, skipped, or weakened.

Command: `npm test -- --run tests/advanced-rag.test.ts tests/chat-rag-regression.test.ts`

Result: this project script always invokes `vitest run tests`, so it ran the full suite rather than only the supplied paths. 101 tests passed and 14 failed. In addition to the baseline evidence-gate/mock assertions, failures included an existing observability metric expectation and a Windows SQLite `EBUSY` cleanup failure. This command was not represented as a passing focused run.

### Round 3 — final static and build verification

Commands:

- `npm run typecheck` — PASS.
- `npm run lint` — PASS.
- `npm --prefix dashboard ci` — completed to install the dashboard's lockfile dependencies.
- `npm run build` — PASS. Vite reported only its pre-existing large-chunk warning (687 kB minified JavaScript).
- Focused identity command from Round 1 rerun after the final streaming-cleanup change — PASS (3/3).

The first build attempt was blocked because dashboard dependencies had not yet been installed; after `npm --prefix dashboard ci`, the same build passed. `npm ci` reported 14 root dependency audit findings (1 moderate, 11 high, 2 critical), and dashboard `npm ci` reported 6 high findings; no dependency upgrades were made because they are outside this scoped source repair.

## Limitations and scope

No live external provider or production deployment was contacted. The direct identity route is verified with local unit/service doubles and uses tenant configuration, so it does not require provider credentials. The broader baseline test failures and audit findings remain follow-up work; they are not claimed resolved by this repair.

## Delivery verification

The delivery archive was created from the six manifest paths only and opened/listed after creation to confirm that it contains no generated assets, dependencies, or unchanged project files.
