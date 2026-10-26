import crypto from 'node:crypto';
import type { EmbeddingProvider } from '../providers/embedding.js';
import type { VectorProvider } from '../providers/vector.js';
import type { ObjectStorage } from '../providers/storage.js';
import { chunks, parseDocument } from './parser.js';
import type { Repository } from '../repositories/repository.js';
import { PipelineObserver, safeRef, type PipelineContext } from '../lib/observability.js';

export class KnowledgeService {
  constructor(
    private storage: ObjectStorage,
    private embeddings: EmbeddingProvider,
    private vectors: VectorProvider,
    private repo?: Repository,
    private pipeline: PipelineObserver = new PipelineObserver({}, 'off'),
  ) {}

  private telemetry(clientId: string, context?: PipelineContext) {
    return context ?? this.pipeline.createContext({ requestId: crypto.randomUUID(), entryPoint: 'knowledge-service', tenantId: clientId });
  }

  private providerDebug(context: PipelineContext, provider: string) {
    if (this.pipeline.mode === 'off') return undefined;
    const basicOperations = new Set([
      'EMBEDDING_CREDENTIALS_MISSING', 'EMBEDDING_REQUEST_STARTED', 'EMBEDDING_REQUEST_FAILED', 'EMBEDDING_HTTP_RESPONSE', 'EMBEDDING_RESPONSE_PARSE_FAILED', 'EMBEDDING_RESPONSE_PARSED',
      'VECTOR_SEARCH_STARTED', 'VECTOR_SEARCH_FAILED', 'VECTOR_SEARCH_HTTP_RESPONSE', 'VECTOR_SEARCH_RESPONSE_PARSE_FAILED', 'VECTOR_SEARCH_PARSED',
    ]);
    return (operation: string, data: Record<string, unknown>) => {
      const detailed = this.pipeline.mode === 'detailed';
      if (!detailed && !basicOperations.has(operation)) return;
      const httpFailure = data.responseOk === false || (typeof data.httpStatus === 'number' && data.httpStatus >= 400);
      const retrying = /retry|scheduled/i.test(operation) || (httpFailure && data.retryableStatus === true && typeof data.attemptNumber === 'number' && typeof data.maxAttempts === 'number' && data.attemptNumber < data.maxAttempts);
      const failed = /fail|error|parse_failed|credentials_missing/i.test(operation) || (httpFailure && !retrying);
      const status = retrying ? 'retrying' : failed ? 'failed' : /started/i.test(operation) ? 'started' : 'success';
      this.pipeline.emit(context, 'pipeline.provider.diagnostic', { stage: `${provider}.provider`, operation: operation.toLowerCase().replace(/[^a-z0-9_.-]/g, '_').slice(0, 90), status, outputMetrics: data, detailed: detailed, errorCode: failed ? `${provider.toUpperCase()}_PROVIDER_OPERATION_FAILED` : undefined });
    };
  }

  async upload(clientId: string, name: string, data: Buffer, traceContext?: PipelineContext) {
    const telemetry = this.telemetry(clientId, traceContext);
    const sourceRef = safeRef(name);
    const text = await this.pipeline.measure(telemetry, 'knowledge.parsing', 'parse_uploaded_document', () => parseDocument(name, data), {
      inputMetrics: { sourceRef, extension: name.includes('.') ? name.slice(name.lastIndexOf('.')).toLowerCase().slice(0, 12) : 'unknown', bytes: data.byteLength },
    });
    if (!text.trim()) {
      this.pipeline.emit(telemetry, 'pipeline.knowledge_ingestion.rejected', { stage: 'knowledge.parsing', operation: 'validate_extracted_text', status: 'rejected', reason: 'document_contains_no_text', errorCode: 'EMPTY_DOCUMENT', inputMetrics: { sourceRef, bytes: data.byteLength } });
      throw new Error('Document contains no text');
    }
    await this.pipeline.measure(telemetry, 'knowledge.storage', 'store_original_document', () => this.storage.put(clientId, name, data), {
      inputMetrics: { sourceRef, bytes: data.byteLength, provider: this.storage.name },
    });
    const parts = await this.pipeline.measure(telemetry, 'knowledge.chunking', 'chunk_extracted_text', async () => chunks(text), {
      inputMetrics: { extractedChars: text.length },
    });
    this.pipeline.emit(telemetry, 'pipeline.knowledge_chunks.created', { stage: 'knowledge.chunking', operation: 'chunk_extracted_text', status: 'success', inputMetrics: { extractedChars: text.length }, outputMetrics: { chunkCount: parts.length, totalChunkChars: parts.reduce((sum, x) => sum + x.length, 0), averageChunkChars: parts.length ? Math.round(parts.reduce((sum, x) => sum + x.length, 0) / parts.length) : 0 } });
    const embedded = await this.pipeline.measure(telemetry, 'knowledge.embedding', 'embed_document_chunks', () => this.embeddings.embed(parts, this.providerDebug(telemetry, 'embedding')), {
      inputMetrics: { provider: this.embeddings.name, inputCount: parts.length, inputChars: parts.reduce((sum, x) => sum + x.length, 0) },
    });
    this.pipeline.emit(telemetry, 'pipeline.knowledge_embedding.completed', { stage: 'knowledge.embedding', operation: 'embed_document_chunks', status: 'success', outputMetrics: { vectorCount: embedded.length, dimensions: embedded[0]?.length ?? 0, vectorsValid: embedded.length === parts.length && embedded.every((v) => v.length > 0 && v.every(Number.isFinite)) } });
    await this.pipeline.measure(telemetry, 'knowledge.indexing', 'replace_source_vectors', () => this.vectors.replaceSource(
      clientId,
      name,
      parts.map((x, i) => ({
        id: crypto.createHash('sha256').update(`${clientId}\0${name}\0${i}\0${x}`).digest('hex'),
        clientId,
        source: name,
        text: x,
        embedding: embedded[i]!,
      })),
    ), { inputMetrics: { provider: this.vectors.name, sourceRef, vectorCount: parts.length, tenantScoped: true } });
    if (this.repo) await this.pipeline.measure(telemetry, 'knowledge.persistence', 'save_knowledge_metadata', () => this.repo!.saveKnowledge(clientId, name, data.byteLength, parts.length, 'ready'), {
      inputMetrics: { sourceRef, bytes: data.byteLength, chunkCount: parts.length },
    });
    this.pipeline.emit(telemetry, 'pipeline.knowledge_ingestion.completed', { stage: 'knowledge.ingestion', operation: 'upload_document', status: 'success', outputMetrics: { sourceRef, bytes: data.byteLength, extractedChars: text.length, chunkCount: parts.length, embeddingCount: embedded.length, storageProvider: this.storage.name, vectorProvider: this.vectors.name } });
    return { filename: name, size: data.byteLength, chunks: parts.length, embeddingStatus: 'ready' };
  }

  async list(clientId: string) {
    const metadata = (await this.repo?.knowledge(clientId)) as
      | Array<{ filename: string; size: number; chunks: number; status: string; updatedAt: string }>
      | undefined;
    if (metadata?.length)
      return metadata.map((x) => ({ ...x, bytes: x.size, embeddingStatus: x.status }));
    const names = await this.storage.list(clientId);
    return Promise.all(names.map(async (filename) => ({
      filename,
      size: (await this.storage.read(clientId, filename)).byteLength,
      bytes: (await this.storage.read(clientId, filename)).byteLength,
      chunks: 0,
      embeddingStatus: 'unknown',
    })));
  }

  async remove(clientId: string, filename: string, traceContext?: PipelineContext) {
    const telemetry = this.telemetry(clientId, traceContext);
    const sourceRef = safeRef(filename);
    await this.pipeline.measure(telemetry, 'knowledge.storage', 'remove_original_document', () => this.storage.remove(clientId, filename), { inputMetrics: { sourceRef, provider: this.storage.name } });
    if (this.repo) await this.pipeline.measure(telemetry, 'knowledge.persistence', 'remove_knowledge_metadata', () => this.repo!.removeKnowledge(clientId, filename), { inputMetrics: { sourceRef } });
    this.pipeline.emit(telemetry, 'pipeline.knowledge_ingestion.rebuild_started', { stage: 'knowledge.rebuild', operation: 'rebuild_after_document_removal', status: 'started', reason: 'document_removed_from_source_storage' });
    return this.rebuild(clientId, telemetry);
  }

  async rebuild(clientId: string, traceContext?: PipelineContext) {
    const telemetry = this.telemetry(clientId, traceContext);
    const names = await this.pipeline.measure(telemetry, 'knowledge.storage', 'list_documents_for_rebuild', () => this.storage.list(clientId), { inputMetrics: { provider: this.storage.name, tenantScoped: true } });
    const all: { source: string; text: string }[] = [];
    for (const name of names) {
      const data = await this.pipeline.measure(telemetry, 'knowledge.storage', 'read_document_for_rebuild', () => this.storage.read(clientId, name), { inputMetrics: { sourceRef: safeRef(name) } });
      const text = await this.pipeline.measure(telemetry, 'knowledge.parsing', 'parse_document_for_rebuild', () => parseDocument(name, data), { inputMetrics: { sourceRef: safeRef(name), bytes: data.byteLength } });
      const parts = await this.pipeline.measure(telemetry, 'knowledge.chunking', 'chunk_document_for_rebuild', async () => chunks(text), { inputMetrics: { sourceRef: safeRef(name), extractedChars: text.length } });
      all.push(...parts.map((x) => ({ source: name, text: x })));
    }
    this.pipeline.emit(telemetry, 'pipeline.knowledge_rebuild.corpus_ready', { stage: 'knowledge.rebuild', operation: 'prepare_rebuild_corpus', status: 'success', outputMetrics: { fileCount: names.length, chunkCount: all.length, textChars: all.reduce((sum, x) => sum + x.text.length, 0) } });
    const e = await this.pipeline.measure(telemetry, 'knowledge.embedding', 'embed_rebuild_corpus', () => this.embeddings.embed(all.map((x) => x.text), this.providerDebug(telemetry, 'embedding')), { inputMetrics: { provider: this.embeddings.name, inputCount: all.length, inputChars: all.reduce((sum, x) => sum + x.text.length, 0) } });
    await this.pipeline.measure(telemetry, 'knowledge.indexing', 'delete_existing_tenant_vectors', () => this.vectors.deleteTenant(clientId), { inputMetrics: { provider: this.vectors.name, tenantScoped: true } });
    await this.pipeline.measure(telemetry, 'knowledge.indexing', 'write_rebuilt_tenant_vectors', () => this.vectors.upsert(clientId, all.map((x, i) => ({
      id: crypto.createHash('sha256').update(`${clientId}\0${x.source}\0${i}\0${x.text}`).digest('hex'),
      clientId, ...x, embedding: e[i]!,
    }))), { inputMetrics: { provider: this.vectors.name, vectorCount: all.length, tenantScoped: true } });
    this.pipeline.emit(telemetry, 'pipeline.knowledge_rebuild.completed', { stage: 'knowledge.rebuild', operation: 'rebuild_tenant_index', status: 'success', outputMetrics: { fileCount: names.length, chunkCount: all.length, embeddingCount: e.length, vectorProvider: this.vectors.name } });
    return { files: names.length, chunks: all.length };
  }
}
