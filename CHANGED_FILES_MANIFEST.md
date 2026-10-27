# Changed Files Manifest

- `src/services/identity.ts` — new. Defines the narrow English/Roman-Hindi identity-intent detector and produces answers only from the active tenant configuration.
- `src/services/advanced-rag.ts` — modified. Normalizes supported identity wording for retrieval and lexical evidence evaluation while preserving the original user question for answer generation.
- `src/services/chat.ts` — modified. Routes configured identity requests consistently through chat, dashboard preview, and streaming; records the deterministic selection and avoids unnecessary provider calls.
- `tests/advanced-rag.test.ts` — modified. Adds a Roman-Hindi identity-normalization/evidence regression.
- `tests/chat-rag-regression.test.ts` — modified. Adds chat, preview, streaming, and two-tenant identity regressions.
- `VERIFICATION_REPORT.md` — new. Records the forensic finding, implementation rationale, and actual verification results.

No dependency or generated-dashboard asset is required by this repair. The archive intentionally excludes build output, `node_modules`, credentials, and unchanged files.
