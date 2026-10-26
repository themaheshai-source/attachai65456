/* eslint-disable @typescript-eslint/no-explicit-any */
import fs from 'node:fs';
import path from 'node:path';
import type { ProviderHealth, RetrievedChunk, VectorRecord } from '../domain/types.js';
import type { ProviderDiagnostic } from '../lib/observability.js';
export interface VectorProvider {
  readonly name: string;
  upsert(clientId: string, rows: VectorRecord[]): Promise<void>;
  replaceSource(clientId: string, source: string, rows: VectorRecord[]): Promise<void>;
  search(clientId: string, v: number[], topK: number, min: number, diagnostic?: ProviderDiagnostic): Promise<RetrievedChunk[]>;
  /** Bounded tenant-only text corpus used by exact, keyword, and BM25 recall. */
  listTenant(clientId: string, limit?: number): Promise<RetrievedChunk[]>;
  deleteTenant(clientId: string): Promise<void>;
  health(): Promise<ProviderHealth>;
}
const cosine = (a: number[], b: number[]) =>
  a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0) /
  ((Math.hypot(...a) || 1) * (Math.hypot(...b) || 1));
export class LocalVector implements VectorProvider {
  name = 'local-json-cosine';
  private rows: VectorRecord[] = [];
  constructor(private file: string) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (fs.existsSync(file))
      this.rows = JSON.parse(fs.readFileSync(file, 'utf8')) as VectorRecord[];
  }
  private save() {
    const t = `${this.file}.tmp`;
    fs.writeFileSync(t, JSON.stringify(this.rows));
    fs.renameSync(t, this.file);
  }
  async upsert(c: string, r: VectorRecord[]) {
    const incoming = new Set(r.map((x) => x.id));
    this.rows = this.rows.filter((x) => x.clientId !== c || !incoming.has(x.id)).concat(r);
    this.save();
  }
  async replaceSource(c: string, source: string, r: VectorRecord[]) {
    this.rows = this.rows.filter((x) => x.clientId !== c || x.source !== source).concat(r);
    this.save();
  }
  async search(c: string, v: number[], k: number, min: number) {
    return this.rows
      .filter((x) => x.clientId === c)
      .map((x) => ({ id: x.id, text: x.text, source: x.source, score: cosine(x.embedding, v) }))
      .filter((x) => x.score >= min)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
  async listTenant(c: string, limit = 2000) {
    return this.rows.filter((x) => x.clientId === c).slice(0, Math.max(0, limit)).map((x) => ({ id: x.id, text: x.text, source: x.source, score: 0 }));
  }
  async deleteTenant(c: string) {
    this.rows = this.rows.filter((x) => x.clientId !== c);
    this.save();
  }
  async health() {
    return { provider: this.name, connected: true, detail: `${this.rows.length} vectors` };
  }
}
export class QdrantVector implements VectorProvider {
  name = 'qdrant';
  constructor(
    private url: string,
    private key?: string,
  ) {}
  private headers() {
    return { 'content-type': 'application/json', ...(this.key ? { 'api-key': this.key } : {}) };
  }
  async ensure(dim: number) {
    await fetch(`${this.url}/collections/tenant_knowledge`, {
      method: 'PUT',
      headers: this.headers(),
      body: JSON.stringify({ vectors: { size: dim, distance: 'Cosine' } }),
    });
  }
  async upsert(c: string, r: VectorRecord[]) {
    if (r[0]) await this.ensure(r[0].embedding.length);
    const res = await fetch(`${this.url}/collections/tenant_knowledge/points?wait=true`, {
      method: 'PUT',
      headers: this.headers(),
      body: JSON.stringify({
        points: r.map((x) => ({
          id: x.id,
          vector: x.embedding,
          payload: { clientId: c, text: x.text, source: x.source },
        })),
      }),
    });
    if (!res.ok) throw new Error(`Qdrant upsert failed (${res.status})`);
  }
  async replaceSource(c: string, source: string, r: VectorRecord[]) {
    const deleted = await fetch(
      `${this.url}/collections/tenant_knowledge/points/delete?wait=true`,
      {
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify({
          filter: {
            must: [
              { key: 'clientId', match: { value: c } },
              { key: 'source', match: { value: source } },
            ],
          },
        }),
      },
    );
    if (!deleted.ok && deleted.status !== 404)
      throw new Error(`Qdrant source replacement failed (${deleted.status})`);
    await this.upsert(c, r);
  }
  async search(c: string, v: number[], k: number, min: number, diagnostic?: ProviderDiagnostic) {
    const started = process.hrtime.bigint();
    diagnostic?.('VECTOR_SEARCH_STARTED', { provider: this.name, topK: k, minSimilarity: min, vectorDimensions: v.length, tenantScoped: true });
    const r = await fetch(`${this.url}/collections/tenant_knowledge/points/search`, {
      method: 'POST', headers: this.headers(),
      body: JSON.stringify({ vector: v, limit: k, score_threshold: min, filter: { must: [{ key: 'clientId', match: { value: c } }] }, with_payload: true }),
    });
    diagnostic?.('VECTOR_SEARCH_HTTP_RESPONSE', { provider: this.name, httpStatus: r.status, responseOk: r.ok, durationMs: Number(process.hrtime.bigint() - started) / 1_000_000 });
    if (!r.ok) throw new Error(`Qdrant search failed (${r.status})`);
    let j: { result: { id: string | number; score: number; payload: { text: string; source: string } }[] };
    try { j = await r.json() as typeof j; }
    catch (error) { diagnostic?.('VECTOR_SEARCH_RESPONSE_PARSE_FAILED', { provider: this.name, errorCategory: error instanceof Error ? error.name : 'unknown' }); throw error; }
    let rows: RetrievedChunk[];
    try { rows = j.result.map((x) => ({ id: String(x.id), ...x.payload, score: x.score })); }
    catch (error) { diagnostic?.('VECTOR_SEARCH_RESPONSE_PARSE_FAILED', { provider: this.name, errorCategory: 'invalid_response_shape' }); throw error; }
    const scores = rows.map((x) => x.score).filter(Number.isFinite).sort((a, b) => a - b);
    diagnostic?.('VECTOR_SEARCH_PARSED', { provider: this.name, candidateCount: rows.length, scoreMin: scores[0] ?? null, scoreMedian: scores.length ? scores[Math.floor((scores.length - 1) / 2)]! : null, scoreMax: scores.at(-1) ?? null, tenantScoped: true });
    return rows;
  }
  async listTenant(c: string, limit = 2000) {
    const r = await fetch(`${this.url}/collections/tenant_knowledge/points/scroll`, {
      method: 'POST', headers: this.headers(),
      body: JSON.stringify({ limit: Math.min(Math.max(0, limit), 2000), with_payload: true, filter: { must: [{ key: 'clientId', match: { value: c } }] } }),
    });
    if (!r.ok) throw new Error(`Qdrant tenant scan failed (${r.status})`);
    const j = await r.json() as { result?: { points?: { id: string | number; payload?: { text?: string; source?: string } }[] } };
    return (j.result?.points ?? []).map((x) => ({ id: String(x.id), text: String(x.payload?.text ?? ''), source: String(x.payload?.source ?? ''), score: 0 }));
  }
  async deleteTenant(c: string) {
    await fetch(`${this.url}/collections/tenant_knowledge/points/delete?wait=true`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ filter: { must: [{ key: 'clientId', match: { value: c } }] } }),
    });
  }
  async health() {
    try {
      const r = await fetch(`${this.url}/healthz`);
      return { provider: this.name, connected: r.ok };
    } catch {
      return { provider: this.name, connected: false };
    }
  }
}

import { createClient, type SupabaseClient } from '@supabase/supabase-js';
/** pgvector adapter. Every operation supplies client_id; SQL RPC also enforces it. */
export class SupabaseVector implements VectorProvider {
  readonly name = 'supabase-pgvector';
  private readonly client: SupabaseClient<any>;
  constructor(urlOrClient: string | SupabaseClient<any>, key?: string) {
    this.client =
      typeof urlOrClient === 'string'
        ? createClient(urlOrClient, key ?? '', {
            auth: { persistSession: false, autoRefreshToken: false },
          })
        : urlOrClient;
  }
  // LocalEmbedding and the Supabase schema both use exactly 128 dimensions.
  // Never coerce vectors: accepting a different dimension would fail later in
  // pgvector with a less actionable error and could corrupt provider switching.
  private validate(v: number[]) {
    if (v.length !== 128)
      throw new Error(`pgvector embedding must have exactly 128 dimensions; received ${v.length}`);
    return v;
  }
  async upsert(clientId: string, rows: VectorRecord[]) {
    if (!rows.length) return;
    const { error } = await this.client.from('knowledge_vectors').upsert(
      rows.map((x) => ({
        id: x.id,
        client_id: clientId,
        source: x.source,
        content: x.text,
        embedding: this.validate(x.embedding),
      })),
      { onConflict: 'id' },
    );
    if (error) throw new Error(`pgvector upsert failed: ${error.message}`);
  }
  async replaceSource(clientId: string, source: string, rows: VectorRecord[]) {
    const { error: de } = await this.client
      .from('knowledge_vectors')
      .delete()
      .eq('client_id', clientId)
      .eq('source', source);
    if (de) throw new Error(`pgvector source delete failed: ${de.message}`);
    await this.upsert(clientId, rows);
  }
  async search(clientId: string, v: number[], topK: number, min: number, diagnostic?: ProviderDiagnostic) {
    const started = process.hrtime.bigint();
    diagnostic?.('VECTOR_SEARCH_STARTED', { provider: this.name, topK, minSimilarity: min, vectorDimensions: v.length, tenantScoped: true });
    const { data, error } = await this.client.rpc('match_knowledge_vectors', {
      query_embedding: this.validate(v), match_client_id: clientId, match_threshold: min, match_count: topK,
    });
    const durationMs = Number(process.hrtime.bigint() - started) / 1_000_000;
    if (error) {
      diagnostic?.('VECTOR_SEARCH_FAILED', { provider: this.name, durationMs, errorCategory: 'database_rpc_error' });
      throw new Error(`pgvector search failed: ${error.message}`);
    }
    const rows = (data ?? []).map((x: any) => ({ id: String(x.id), text: String(x.text ?? x.content), source: String(x.source), score: Number(x.similarity ?? x.score) }));
    const scores = rows.map((x: RetrievedChunk) => x.score).filter(Number.isFinite).sort((a: number, b: number) => a - b);
    diagnostic?.('VECTOR_SEARCH_PARSED', { provider: this.name, durationMs, candidateCount: rows.length, scoreMin: scores[0] ?? null, scoreMedian: scores.length ? scores[Math.floor((scores.length - 1) / 2)]! : null, scoreMax: scores.at(-1) ?? null, tenantScoped: true });
    return rows;
  }
  async listTenant(clientId: string, limit = 2000) {
    const { data, error } = await this.client
      .from('knowledge_vectors')
      .select('id,content,source')
      .eq('client_id', clientId)
      .order('created_at', { ascending: true })
      .limit(Math.min(Math.max(0, limit), 2000));
    if (error) throw new Error(`pgvector tenant scan failed: ${error.message}`);
    return (data ?? []).map((x: any) => ({ id: String(x.id), text: String(x.content), source: String(x.source), score: 0 }));
  }
  async deleteTenant(clientId: string) {
    const { error } = await this.client
      .from('knowledge_vectors')
      .delete()
      .eq('client_id', clientId);
    if (error) throw new Error(`pgvector tenant delete failed: ${error.message}`);
  }
  async health(): Promise<ProviderHealth> {
    const { error } = await this.client.from('knowledge_vectors').select('id').limit(1);
    return { provider: this.name, connected: !error, detail: error?.message };
  }
}
