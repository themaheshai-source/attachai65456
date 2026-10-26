import crypto from 'node:crypto';
import type { ProviderHealth } from '../domain/types.js';
import type { ProviderDiagnostic } from '../lib/observability.js';
export interface EmbeddingProvider {
  readonly name: string;
  embed(texts: string[], diagnostic?: ProviderDiagnostic): Promise<number[][]>;
  health(): Promise<ProviderHealth>;
}
export class LocalEmbedding implements EmbeddingProvider {
  name = 'local-deterministic';
  constructor(private dimensions = 128) {}
  async embed(texts: string[]) {
    return texts.map((t) => {
      const v = Array(this.dimensions).fill(0) as number[];
      for (const token of t.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
        const h = crypto.createHash('sha256').update(token).digest();
        const i = h.readUInt32BE(0) % this.dimensions;
        v[i] = (v[i] ?? 0) + (h[4]! % 2 ? 1 : -1);
      }
      const n = Math.hypot(...v) || 1;
      return v.map((x) => x / n);
    });
  }
  async health() {
    return { provider: this.name, connected: true };
  }
}
export class OpenAIEmbedding implements EmbeddingProvider {
  name = 'openai-compatible';
  constructor(
    private base: string,
    private key: string | undefined,
    private model: string,
  ) {}
  async embed(texts: string[], diagnostic?: ProviderDiagnostic) {
    if (!this.key) { diagnostic?.('EMBEDDING_CREDENTIALS_MISSING', { provider: this.name, model: this.model }); throw new Error('OPENAI_API_KEY is required'); }
    const started = process.hrtime.bigint();
    diagnostic?.('EMBEDDING_REQUEST_STARTED', { provider: this.name, model: this.model, inputCount: texts.length, inputChars: texts.reduce((sum, text) => sum + text.length, 0) });
    let r: Response;
    try {
      r = await fetch(`${this.base}/embeddings`, { method: 'POST', headers: { authorization: `Bearer ${this.key}`, 'content-type': 'application/json' }, body: JSON.stringify({ model: this.model, input: texts }) });
    } catch (error) {
      diagnostic?.('EMBEDDING_REQUEST_FAILED', { provider: this.name, model: this.model, durationMs: Number(process.hrtime.bigint() - started) / 1_000_000, errorCategory: error instanceof Error ? error.name : 'unknown' });
      throw error;
    }
    diagnostic?.('EMBEDDING_HTTP_RESPONSE', { provider: this.name, model: this.model, httpStatus: r.status, responseOk: r.ok, durationMs: Number(process.hrtime.bigint() - started) / 1_000_000 });
    if (!r.ok) throw new Error(`Embedding provider failed (${r.status})`);
    let j: { data: { embedding: number[] }[]; usage?: { prompt_tokens?: number; total_tokens?: number } };
    try { j = await r.json() as typeof j; }
    catch (error) { diagnostic?.('EMBEDDING_RESPONSE_PARSE_FAILED', { provider: this.name, model: this.model, errorCategory: error instanceof Error ? error.name : 'unknown' }); throw error; }
    let vectors: number[][];
    try { vectors = j.data.map((x) => x.embedding); }
    catch (error) { diagnostic?.('EMBEDDING_RESPONSE_PARSE_FAILED', { provider: this.name, model: this.model, errorCategory: 'invalid_response_shape' }); throw error; }
    diagnostic?.('EMBEDDING_RESPONSE_PARSED', { provider: this.name, model: this.model, vectorCount: vectors.length, dimensions: vectors[0]?.length ?? 0, finiteValues: vectors.every((v) => v.every(Number.isFinite)), usage: j.usage ? { promptTokens: j.usage.prompt_tokens ?? null, totalTokens: j.usage.total_tokens ?? null, source: 'provider_reported' } : { source: 'unavailable' } });
    return vectors;
  }
  async health() {
    return {
      provider: this.name,
      connected: Boolean(this.key),
      detail: this.key ? 'configured' : 'missing credentials',
    };
  }
}
