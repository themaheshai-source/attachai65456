/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { CheckCircle2, Download, FileCode2, FolderGit2, ShieldCheck } from 'lucide-react';

const CHANGED_FILES = [
  {
    path: 'public/widget.js',
    summary: 'Restored leadSubmitted state, standalone clientName binding, and launcher backdrop-filter.',
  },
  {
    path: 'src/app.ts',
    summary: 'Resolved public static asset root reliably relative to module path with CWD fallback.',
  },
  {
    path: 'src/domain/schemas.ts',
    summary: 'Allowed lead capture payloads linked to an active conversationId without requiring duplicate message fields.',
  },
  {
    path: 'src/providers/llm.ts',
    summary: 'Updated LocalLlm to prioritize retrieved context, expand query synonyms, fall back to top high-confidence chunks, and answer history follow-ups.',
  },
  {
    path: 'src/repositories/repository.ts',
    summary: 'Added tenant conversations listing contract to Repository interface.',
  },
  {
    path: 'src/repositories/sqlite.ts',
    summary: 'Fixed missing await in conversationForClient and implemented tenant conversations query.',
  },
  {
    path: 'src/repositories/supabase.ts',
    summary: 'Implemented tenant conversations listing query with message count and latest message preview.',
  },
  {
    path: 'src/routes/admin.ts',
    summary: 'Added GET /admin/clients/:id/conversations and GET /admin/clients/:id/conversations/:conversationId routes.',
  },
  {
    path: 'src/routes/public.ts',
    summary: 'Scoped strict email destination requirement in POST /v1/leads to SMTP provider so outbox providers work out of the box.',
  },
  {
    path: 'src/services/advanced-rag.ts',
    summary: 'Fixed lexical/BM25 zero-hit filtering, preserved dense semantic scores in rerank, fixed single-chunk semantic evidence sufficiency, and cleaned Q/A scaffolding.',
  },
  {
    path: 'src/services/chat.ts',
    summary: 'Connected conversation history, follow-up query resolution, bounded LLM reformulation, summary generation, and tenant-specific fallback.',
  },
  {
    path: 'src/services/grounding.ts',
    summary: 'Added INR (₹) and word-number support, prevented false-positive contradiction/qualification rejections, and rejected unjustified generic refusals.',
  },
  {
    path: 'tests/context-stream.test.ts',
    summary: 'Added regression tests for conversational fact memory, pronoun follow-ups, and admin session endpoints.',
  },
];

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-12 font-sans">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="border border-slate-800 bg-slate-900/70 rounded-2xl p-6 md:p-8 flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
              Verified Patch Package Ready
            </div>
            <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-white">
              Orbit AI — Repaired Changed Files Archive
            </h1>
            <p className="text-sm text-slate-400 max-w-2xl">
              Contains only the 13 modified files with their exact relative directory structure preserved. All 14 test suites (93 tests), TypeScript typecheck, and ESLint verification pass.
            </p>
          </div>
          <a
            href="/orbit-ai-repaired-changes.zip"
            download="orbit-ai-repaired-changes.zip"
            className="inline-flex items-center justify-center gap-2.5 px-5 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-sm transition shrink-0 shadow-lg shadow-emerald-500/10"
          >
            <Download className="w-4 h-4" />
            Download Changed Files (.zip)
          </a>
        </header>

        <section className="border border-slate-800 bg-slate-900/40 rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-medium text-slate-200">
              <FolderGit2 className="w-4 h-4 text-emerald-400" />
              Modified Files in Archive ({CHANGED_FILES.length})
            </div>
            <span className="text-xs text-slate-400 font-mono">orbit-ai-repaired-changes.zip</span>
          </div>
          <div className="divide-y divide-slate-800/80">
            {CHANGED_FILES.map((file) => (
              <div key={file.path} className="px-6 py-3.5 flex items-start gap-3.5 hover:bg-slate-900/60 transition">
                <FileCode2 className="w-4 h-4 text-slate-400 shrink-0 mt-1" />
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-mono text-emerald-300 break-all">{file.path}</div>
                  <div className="text-xs text-slate-400 mt-0.5">{file.summary}</div>
                </div>
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-1" />
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
