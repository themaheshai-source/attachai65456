import crypto from 'node:crypto';
import type { Repository } from '../repositories/repository.js';
import type { EmbeddingProvider } from '../providers/embedding.js';
import type { VectorProvider } from '../providers/vector.js';
import type { LlmInput, LlmProvider } from '../providers/llm.js';
import type { Client, ChatMessage, RetrievedChunk } from '../domain/types.js';
import { AppError } from '../lib/errors.js';
import { PipelineObserver, safeErrorCode, safeRef, type PipelineContext } from '../lib/observability.js';
import {
  cleanAnswerScaffolding,
  retrieveAdvancedEvidence,
  tokens,
  type EvidenceDecision,
  type RetrievalDiagnostics,
} from './advanced-rag.js';
import { groundingCorrection, validateGrounding } from './grounding.js';

type ChatResult = {
  sessionId: string;
  conversationId: string;
  answer: string;
  sources: { chunkId: string; source: string; score: number }[];
  leadCollection?: { enabled: true; fields: string[] };
};

type Prepared = {
  requestId: string;
  telemetry: PipelineContext;
  client: Client;
  sid: string;
  conversationId: string;
  history: ChatMessage[];
  found: RetrievedChunk[];
  decision: EvidenceDecision;
  diagnostics: RetrievalDiagnostics;
  started: number;
  summary: boolean;
};

const summaryRequest = (message: string) =>
  /\b(summarize|summary|recap|what did we discuss|what did i|what you just said|what you just told)\b/i.test(message);
const clearlyGeneral = (message: string) =>
  /^(hi|hello|hey|thanks|thank you|okay|ok|good morning|good afternoon|good evening|how are you|tell me a joke|what is the meaning of life)[!.? ]*$/i.test(
    message.trim(),
  );
const followUpReference = (message: string) =>
  /\b(it|its|they|them|their|that|this|those|these|there|he|she|his|her)\b/i.test(message);
const answerTokens = (answer: string) => answer.match(/\S+\s*/g) ?? [];
const scoreSummary = (chunks: RetrievedChunk[]) => {
  const scores = chunks.map((x) => x.score).filter(Number.isFinite).sort((a, b) => a - b);
  return {
    count: scores.length,
    min: scores[0] ?? null,
    median: scores.length ? scores[Math.floor((scores.length - 1) / 2)]! : null,
    max: scores.at(-1) ?? null,
  };
};

function boundHistory(prior: ChatMessage[], maxRecent = 8, maxChars = 4000): ChatMessage[] {
  if (!prior.length) return [];
  const recent = prior.slice(-maxRecent);
  const older = prior.slice(0, -maxRecent);
  const result: ChatMessage[] = [];
  if (older.length > 0) {
    const facts = older
      .map((m) => m.content.trim())
      .filter((c) => c.length > 0 && !c.endsWith('?'))
      .slice(-6)
      .join(' ');
    if (facts) {
      result.push({
        role: 'assistant',
        content: `Earlier conversation context: ${facts.slice(0, 800)}`,
      });
    }
  }
  let used = result.reduce((sum, m) => sum + m.content.length, 0);
  const selectedRecent: ChatMessage[] = [];
  for (let i = recent.length - 1; i >= 0; i--) {
    const msg = recent[i]!;
    if (used + msg.content.length > maxChars && selectedRecent.length >= 2) break;
    selectedRecent.unshift(msg);
    used += msg.content.length;
  }
  return [...result, ...selectedRecent];
}

function historyContextualQuery(message: string, history: ChatMessage[]): string | null {
  if (!history.length || !followUpReference(message)) return null;
  const recentText = history
    .filter((m) => !/provided by|don['’]?t have|no information|not found in/i.test(m.content))
    .slice(-4)
    .map((m) => m.content)
    .join(' ');
  const contextTerms = [...new Set(tokens(recentText))].slice(0, 8);
  return contextTerms.length ? `${message} ${contextTerms.join(' ')}` : null;
}

function historySupportsFollowUp(message: string, history: ChatMessage[]): boolean {
  if (!history.length) return false;
  const qTerms = new Set(tokens(message));
  const declarativeHistory = history.filter(
    (m) =>
      m.content.trim().length > 0 &&
      !m.content.trim().endsWith('?') &&
      !/provided by|don['’]?t have|no information|not found in/i.test(m.content),
  );
  if (!declarativeHistory.length) return false;
  if (followUpReference(message)) {
    return declarativeHistory.some((m) => {
      const hTerms = new Set(tokens(m.content));
      return [...qTerms].some((t) => hTerms.has(t)) || /\d/.test(m.content);
    });
  }
  return declarativeHistory.some((m) => {
    const hTerms = new Set(tokens(m.content));
    const hits = [...qTerms].filter((t) => hTerms.has(t)).length;
    return hits >= Math.min(2, Math.max(1, qTerms.size));
  });
}

function tenantFallback(client: Client): string {
  const configured = client.config.fallbackMessage;
  if (configured === 'UNKNOWN') return configured;
  if (configured.includes(`provided by ${client.name}`)) return configured;
  const generic =
    /sorry,?\s+i\s+(?:don['’]?t\s+have|have\s+no)\s+(?:that\s+)?information/i.test(configured);
  return generic
    ? `I can only answer using the verified information provided by ${client.name}.`
    : `${configured} (based on information provided by ${client.name})`;
}

/** Tenant-scoped orchestration. Optional telemetry preserves compatibility for direct service consumers. */
export class ChatService {
  constructor(
    private repo: Repository,
    private embedding: EmbeddingProvider,
    private vector: VectorProvider,
    private llm: LlmProvider,
    private pipeline: PipelineObserver = new PipelineObserver({}, 'off'),
  ) {}

  private context(clientId: string, entryPoint = 'service-chat', context?: PipelineContext) {
    return context ?? this.pipeline.createContext({ requestId: crypto.randomUUID(), entryPoint, tenantId: clientId });
  }

  private emit(
    context: PipelineContext,
    event: string,
    stage: string,
    operation: string,
    status: 'started' | 'success' | 'failed' | 'skipped' | 'retrying' | 'fallback' | 'rejected' | 'cancelled',
    data: { durationMs?: number; reason?: string | null; errorCode?: string | null; inputMetrics?: Record<string, unknown>; outputMetrics?: Record<string, unknown>; detailed?: boolean } = {},
  ) {
    this.pipeline.emit(context, event, { stage, operation, status, ...data });
  }

  private providerDebug(context: PipelineContext, provider = 'llm') {
    if (this.pipeline.mode === 'off') return undefined;
    const basicOperations = new Set([
      'LLM_HTTP_ATTEMPT_STARTED', 'LLM_HTTP_ATTEMPT_COMPLETED', 'LLM_HTTP_ATTEMPT_FAILED', 'LLM_RETRY_SCHEDULED',
      'LLM_RESPONSE_HTTP_ERROR', 'LLM_RESPONSE_PARSE_FAILED', 'LLM_RESPONSE_PARSED', 'LLM_STREAM_BODY_MISSING', 'LLM_STREAM_PARSED',
      'LLM_REFORMULATION_HTTP_ERROR', 'LLM_REFORMULATION_PARSE_FAILED', 'LLM_REFORMULATION_RESPONSE', 'LLM_STREAM_RECOVERY_STARTED',
      'EMBEDDING_CREDENTIALS_MISSING', 'EMBEDDING_REQUEST_STARTED', 'EMBEDDING_REQUEST_FAILED', 'EMBEDDING_HTTP_RESPONSE', 'EMBEDDING_RESPONSE_PARSE_FAILED', 'EMBEDDING_RESPONSE_PARSED',
      'VECTOR_SEARCH_STARTED', 'VECTOR_SEARCH_FAILED', 'VECTOR_SEARCH_HTTP_RESPONSE', 'VECTOR_SEARCH_RESPONSE_PARSE_FAILED', 'VECTOR_SEARCH_PARSED',
    ]);
    return (operation: string, data: Record<string, unknown>) => {
      const detailed = this.pipeline.mode === 'detailed';
      if (!detailed && !basicOperations.has(operation)) return;
      const httpFailure = data.responseOk === false || (typeof data.httpStatus === 'number' && data.httpStatus >= 400);
      const retrying = /retry|scheduled/i.test(operation) || (httpFailure && data.retryableStatus === true && typeof data.attemptNumber === 'number' && typeof data.maxAttempts === 'number' && data.attemptNumber < data.maxAttempts) || (operation === 'LLM_HTTP_ATTEMPT_FAILED' && typeof data.attemptNumber === 'number' && typeof data.maxAttempts === 'number' && data.attemptNumber < data.maxAttempts);
      const failed = /fail|error|parse_failed|credentials_missing/i.test(operation) || (httpFailure && !retrying);
      const status = retrying ? 'retrying' : failed ? 'failed' : /started/i.test(operation) ? 'started' : 'success';
      const fallback = operation === 'LLM_RESPONSE_PARSED' && data.fallbackSubstituted === true;
      this.pipeline.emit(context, fallback ? 'pipeline.fallback.selected' : 'pipeline.provider.diagnostic', {
        stage: fallback ? 'answer.fallback' : `${provider}.provider`,
        operation: operation.toLowerCase().replace(/[^a-z0-9_.-]/g, '_').slice(0, 90),
        status: fallback ? 'fallback' : status,
        outputMetrics: data,
        detailed: detailed && !fallback,
        reason: fallback ? 'llm_response_content_missing' : undefined,
        errorCode: failed ? `${provider.toUpperCase()}_PROVIDER_OPERATION_FAILED` : undefined,
      });
    };
  }

  private retrievalDiagnostic(context: PipelineContext, prefix: string, stage: string, metrics: Record<string, unknown>) {
    if (this.pipeline.mode !== 'detailed') return;
    const secured: Record<string, unknown> = { ...metrics };
    for (const key of ['selected', 'ranked']) {
      const entries = secured[key];
      if (Array.isArray(entries)) secured[key] = entries.slice(0, 25).map((value) => {
        const item = value as Record<string, unknown>;
        return {
          chunkRef: typeof item.id === 'string' ? safeRef(item.id) : undefined,
          score: item.score,
          textChars: item.textChars,
          ...(typeof item.source === 'string' ? { sourceRef: safeRef(item.source) } : {}),
        };
      });
    }
    const durationMs = typeof secured.durationMs === 'number' ? secured.durationMs : undefined;
    delete secured.durationMs;
    const skipped = secured.executed === false;
    this.pipeline.emit(context, `${prefix}${stage}`, {
      stage: `${prefix.replace(/^pipeline\./, '').replace(/\.$/, '')}.${stage}`.slice(0, 100),
      operation: stage,
      status: skipped ? 'skipped' : 'success',
      ...(durationMs === undefined ? {} : { durationMs }),
      reason: typeof secured.reason === 'string' ? secured.reason : skipped ? 'operation_not_executed' : undefined,
      outputMetrics: secured,
      detailed: true,
    });
  }

  async retrieve(clientId: string, message: string, history: ChatMessage[] = [], traceContext?: PipelineContext) {
    const telemetry = this.context(clientId, 'chat-retrieval', traceContext);
    const client = await this.pipeline.measure(telemetry, 'tenant', 'load_client_configuration', () => this.repo.getClient(clientId), { inputMetrics: { tenantScoped: true } });
    if (!client?.enabled) {
      this.emit(telemetry, 'pipeline.stage.failed', 'tenant', 'check_client_enabled', 'failed', { errorCode: 'TENANT_DISABLED', reason: 'tenant_missing_or_disabled' });
      throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    }
    this.pipeline.setTenant(telemetry, client.id);
    const topK = Math.min(client.config.topK ?? 20, 40);
    const minSim = client.config.minSimilarity ?? 0.05;
    const queryEmbedding = await this.pipeline.measure(telemetry, 'retrieval.embedding', 'embed_user_query', () => this.embedding.embed([message], this.providerDebug(telemetry, 'embedding')), { inputMetrics: { provider: this.embedding.name, inputCount: 1, inputChars: message.length } });
    const queryVector = queryEmbedding[0];
    this.emit(telemetry, 'pipeline.embedding.completed', 'retrieval.embedding', 'embed_user_query', queryVector && queryVector.length ? 'success' : 'failed', {
      inputMetrics: { provider: this.embedding.name, inputCount: 1, inputChars: message.length },
      outputMetrics: { vectorCount: queryEmbedding.length, dimensions: queryVector?.length ?? 0, finiteValues: Boolean(queryVector?.every(Number.isFinite)) },
      errorCode: queryVector && queryVector.length ? undefined : 'EMPTY_EMBEDDING',
    });
    const dense = await this.pipeline.measure(telemetry, 'retrieval.dense_search', 'tenant_vector_search', () => this.vector.search(clientId, queryVector!, topK, minSim, this.providerDebug(telemetry, 'vector')), {
      inputMetrics: { provider: this.vector.name, topK, minSimilarity: minSim, tenantScoped: true, queryVectorDimensions: queryVector?.length ?? 0 },
    });
    this.emit(telemetry, 'pipeline.retrieval.dense.completed', 'retrieval.dense_search', 'tenant_vector_search', 'success', {
      outputMetrics: { candidateCount: dense.length, scores: scoreSummary(dense), candidateRefs: dense.slice(0, 20).map((x) => ({ chunkRef: safeRef(x.id), sourceRef: safeRef(x.source), score: x.score })) },
    });
    const corpus = await this.pipeline.measure(telemetry, 'retrieval.corpus', 'load_tenant_corpus', async () =>
      typeof this.vector.listTenant === 'function' ? await this.vector.listTenant(clientId, 2000) : dense,
    { inputMetrics: { provider: this.vector.name, tenantScoped: true, maxChunks: 2000, fallbackToDense: typeof this.vector.listTenant !== 'function' } });
    this.emit(telemetry, 'pipeline.retrieval.corpus.completed', 'retrieval.corpus', 'load_tenant_corpus', 'success', {
      outputMetrics: { candidateCount: corpus.length, textChars: corpus.reduce((sum, x) => sum + x.text.length, 0), sourceCount: new Set(corpus.map((x) => x.source)).size },
    });
    const result = await this.pipeline.measure(telemetry, 'retrieval.advanced_rag', 'retrieve_and_rank_evidence', () => retrieveAdvancedEvidence({
      question: message,
      corpus,
      dense,
      onDiagnostic: this.pipeline.mode === 'detailed' ? (stage, metrics) => this.retrievalDiagnostic(telemetry, 'pipeline.retrieval.', stage, metrics) : undefined,
    }), { inputMetrics: { denseCandidates: dense.length, corpusChunks: corpus.length, queryChars: message.length } });

    const followUpQuery = historyContextualQuery(message, history);
    if (!result.decision.sufficient && followUpQuery) {
      this.emit(telemetry, 'pipeline.retrieval.followup.started', 'retrieval.followup', 'history_contextual_query', 'started', {
        inputMetrics: { queryChars: message.length, contextualQueryChars: followUpQuery.length, historyMessages: history.length },
      });
      const followUpResult = await this.pipeline.measure(telemetry, 'retrieval.followup', 'rerun_advanced_retrieval_with_history_terms', () => retrieveAdvancedEvidence({
        question: followUpQuery,
        corpus,
        dense,
        onDiagnostic: this.pipeline.mode === 'detailed' ? (stage, metrics) => this.retrievalDiagnostic(telemetry, 'pipeline.retrieval.followup.', stage, metrics) : undefined,
      }));
      const promoted = followUpResult.decision.sufficient && followUpResult.evidence.length > 0;
      this.emit(telemetry, 'pipeline.retrieval.followup.completed', 'retrieval.followup', 'rerun_advanced_retrieval_with_history_terms', 'success', {
        outputMetrics: { promoted, candidateCount: followUpResult.evidence.length, status: followUpResult.decision.status, coverage: followUpResult.decision.coverage },
        reason: promoted ? 'history_contextual_evidence_sufficient' : 'history_contextual_evidence_not_sufficient',
      });
      if (promoted) Object.assign(result, followUpResult);
    } else {
      this.emit(telemetry, 'pipeline.retrieval.followup.skipped', 'retrieval.followup', 'history_contextual_query', 'skipped', {
        reason: result.decision.sufficient ? 'initial_evidence_sufficient' : 'no_history_contextual_query',
        outputMetrics: { historyMessages: history.length, followUpQueryAvailable: Boolean(followUpQuery) },
      });
    }

    const maxDenseScore = Math.max(0, ...dense.map((x) => x.score));
    const shouldReformulate = result.decision.status !== 'SUPPORTED' && maxDenseScore < 0.5;
    if (shouldReformulate && typeof this.llm.reformulate === 'function') {
      this.emit(telemetry, 'pipeline.retrieval.reformulation.started', 'retrieval.query_reformulation', 'reformulate_query', 'started', {
        inputMetrics: { evidenceStatus: result.decision.status, maxDenseScore, evidenceCount: result.evidence.length, historyMessages: history.length, provider: this.llm.name },
      });
      const reformulationInput: LlmInput = { question: message, prompt: client.prompt, context: result.evidence, config: client.config, history, debug: this.providerDebug(telemetry), debugDetailed: this.pipeline.mode === 'detailed' };
      const reformulated = await this.pipeline.measure(telemetry, 'llm.query_reformulation', 'reformulate_retrieval_query', () => this.llm.reformulate!(reformulationInput), {
        inputMetrics: { provider: this.llm.name, questionChars: message.length, evidenceCount: result.evidence.length, historyMessages: history.length },
      });
      if (reformulated) {
        const lines = reformulated.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
        const firstLine = lines[0] ?? reformulated;
        this.emit(telemetry, 'pipeline.retrieval.reformulation.completed', 'retrieval.query_reformulation', 'reformulate_query', 'success', {
          outputMetrics: { returned: true, alternativeQueryCount: lines.length, selectedQueryChars: firstLine.length },
        });
        const refEmbedding = await this.pipeline.measure(telemetry, 'retrieval.embedding', 'embed_reformulated_query', () => this.embedding.embed([firstLine], this.providerDebug(telemetry, 'embedding')), { inputMetrics: { provider: this.embedding.name, inputCount: 1, inputChars: firstLine.length } });
        const refVector = refEmbedding[0];
        const recovered = await this.pipeline.measure(telemetry, 'retrieval.dense_search', 'search_reformulated_query', () => this.vector.search(clientId, refVector!, topK, minSim, this.providerDebug(telemetry, 'vector')), {
          inputMetrics: { provider: this.vector.name, topK, minSimilarity: minSim, tenantScoped: true, queryVectorDimensions: refVector?.length ?? 0 },
        });
        const mergedMap = new Map<string, RetrievedChunk>();
        for (const item of [...recovered, ...dense]) {
          const existing = mergedMap.get(item.id);
          if (!existing || item.score > existing.score) mergedMap.set(item.id, item);
        }
        const mergedDense = [...mergedMap.values()].sort((a, b) => b.score - a.score);
        this.emit(telemetry, 'pipeline.retrieval.candidates_merged', 'retrieval.candidate_merge', 'merge_dense_candidates', 'success', {
          inputMetrics: { reformulatedCandidates: recovered.length, originalDenseCandidates: dense.length },
          outputMetrics: { mergedCount: mergedDense.length, duplicateCount: recovered.length + dense.length - mergedDense.length, scores: scoreSummary(mergedDense) },
        });
        const mergedCorpus = typeof this.vector.listTenant === 'function' ? corpus : mergedDense;
        const reretrieved = await this.pipeline.measure(telemetry, 'retrieval.advanced_rag', 'rerun_retrieval_after_reformulation', () => retrieveAdvancedEvidence({
          question: `${message} ${firstLine}`,
          corpus: mergedCorpus,
          dense: mergedDense,
          onDiagnostic: this.pipeline.mode === 'detailed' ? (stage, metrics) => this.retrievalDiagnostic(telemetry, 'pipeline.retrieval.reformulated.', stage, metrics) : undefined,
        }));
        Object.assign(result, reretrieved);
        this.emit(telemetry, 'pipeline.retrieval.reformulation_retrieval.completed', 'retrieval.query_reformulation', 'rerun_retrieval_after_reformulation', 'success', {
          outputMetrics: { evidenceCount: result.evidence.length, evidenceStatus: result.decision.status, sufficient: result.decision.sufficient },
        });
      } else {
        this.emit(telemetry, 'pipeline.retrieval.reformulation.empty', 'retrieval.query_reformulation', 'reformulate_query', 'skipped', {
          reason: 'provider_returned_no_usable_alternative_query', outputMetrics: { returned: false },
        });
      }
    } else {
      this.emit(telemetry, 'pipeline.retrieval.reformulation.skipped', 'retrieval.query_reformulation', 'reformulate_query', 'skipped', {
        reason: typeof this.llm.reformulate !== 'function' ? 'provider_does_not_support_reformulation' : result.decision.status === 'SUPPORTED' ? 'evidence_supported' : 'dense_score_at_or_above_threshold',
        outputMetrics: { evidenceStatus: result.decision.status, maxDenseScore, reformulationSupported: typeof this.llm.reformulate === 'function' },
      });
    }
    return { client, ...result };
  }

  private selectEvidence(candidates: RetrievedChunk[], question: string) {
    const queryTerms = new Set(question.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
    return candidates.slice().sort((a, b) => {
      const score = (chunk: RetrievedChunk) =>
        [...queryTerms].filter((term) => chunk.text.toLowerCase().includes(term)).length;
      if (Math.abs(b.score - a.score) < 0.1) return score(b) - score(a) || b.score - a.score;
      return b.score - a.score || score(b) - score(a);
    });
  }

  private groundingEvidence(prepared: Prepared, question: string): RetrievedChunk[] {
    if (historySupportsFollowUp(question, prepared.history)) {
      const historyChunks = prepared.history
        .filter((m) => m.role === 'user' || (!m.content.includes(prepared.client.config.fallbackMessage) && !/provided by|don['’]?t have|no information|not found in/i.test(m.content)))
        .map((m, idx) => ({ id: `history-${idx}`, source: 'conversation-history', text: m.content, score: 1 }));
      this.emit(prepared.telemetry, 'pipeline.grounding_evidence.completed', 'grounding.evidence', 'include_followup_history', 'success', {
        outputMetrics: { retrievedEvidenceCount: prepared.found.length, historyEvidenceCount: historyChunks.length, totalEvidenceCount: prepared.found.length + historyChunks.length },
        reason: 'history_supports_follow_up',
      });
      return [...prepared.found, ...historyChunks];
    }
    this.emit(prepared.telemetry, 'pipeline.grounding_evidence.skipped', 'grounding.evidence', 'include_followup_history', 'skipped', {
      reason: 'history_does_not_support_follow_up', outputMetrics: { retrievedEvidenceCount: prepared.found.length },
    });
    return prepared.found;
  }

  private llmInput(prepared: Prepared, question: string, prompt = prepared.client.prompt): LlmInput {
    const finalPrompt = `${prompt}\nEvidence decision: ${prepared.decision.status}. Answer only supported portions. If evidence is partial, state the supported portion and clearly qualify what is not available. If evidence is uncertain or absent for a tenant-specific fact, use the configured fallback message.`;
    this.emit(prepared.telemetry, 'pipeline.prompt.constructed', 'prompt.construction', 'assemble_llm_input', 'success', {
      inputMetrics: { questionChars: question.length, configuredPromptChars: prompt.length, contextChunks: prepared.found.length, historyMessagesAvailable: prepared.history.length },
      outputMetrics: {
        effectivePromptChars: finalPrompt.length,
        contextChars: prepared.found.reduce((sum, x) => sum + x.text.length, 0),
        contextTokenEstimate: Math.ceil(prepared.found.reduce((sum, x) => sum + x.text.length, 0) / 4),
        historyMessagesSentToProvider: Math.min(prepared.history.length, 8),
        historyChars: prepared.history.slice(-8).reduce((sum, x) => sum + x.content.length, 0),
        roles: ['system', 'system', ...prepared.history.slice(-8).map((x) => x.role), 'user'],
      },
    });
    return { question, prompt: finalPrompt, context: prepared.found, config: prepared.client.config, history: prepared.history, debug: this.providerDebug(prepared.telemetry), debugDetailed: this.pipeline.mode === 'detailed' };
  }

  private async answerWithGrounding(prepared: Prepared, question: string, prompt = prepared.client.prompt) {
    const evidence = this.groundingEvidence(prepared, question);
    const llmInput = this.llmInput(prepared, question, prompt);
    this.emit(prepared.telemetry, 'pipeline.llm.call.started', 'llm.answer_generation', 'generate_initial_answer', 'started', {
      inputMetrics: { provider: this.llm.name, purpose: 'answer_generation', streaming: false, contextChunks: llmInput.context.length, historyMessages: llmInput.history?.length ?? 0, questionChars: question.length },
    });
    const generated = await this.pipeline.measure(prepared.telemetry, 'llm.answer_generation', 'generate_initial_answer', () => this.llm.answer(llmInput), {
      inputMetrics: { provider: this.llm.name, purpose: 'answer_generation', streaming: false, contextChunks: llmInput.context.length, questionChars: question.length },
    });
    const first = cleanAnswerScaffolding(generated);
    this.emit(prepared.telemetry, 'pipeline.answer.cleaned', 'answer.cleaning', 'clean_initial_answer', 'success', {
      inputMetrics: { rawAnswerChars: generated.length }, outputMetrics: { cleanedAnswerChars: first.length, changed: generated !== first },
    });
    const firstCheck = await this.pipeline.measure(prepared.telemetry, 'grounding.validation', 'validate_initial_answer', async () => validateGrounding(question, first, evidence), {
      inputMetrics: { answerChars: first.length, evidenceCount: evidence.length, questionChars: question.length },
    });
    this.emit(prepared.telemetry, 'pipeline.grounding.validation.completed', 'grounding.validation', 'validate_initial_answer', firstCheck.ok ? 'success' : 'rejected', {
      inputMetrics: { evidenceCount: evidence.length, answerChars: first.length },
      outputMetrics: { accepted: firstCheck.ok, rejectionReasons: firstCheck.reasons, reasonCount: firstCheck.reasons.length },
      reason: firstCheck.ok ? 'answer_passed_grounding' : 'answer_failed_grounding',
    });
    if (firstCheck.ok) {
      this.emit(prepared.telemetry, 'pipeline.answer.selected', 'answer.selection', 'select_initial_answer', 'success', { outputMetrics: { answerChars: first.length, choice: 'original' } });
      return first;
    }

    this.emit(prepared.telemetry, 'pipeline.correction.started', 'grounding.correction', 'generate_corrected_answer', 'started', {
      inputMetrics: { rejectionReasons: firstCheck.reasons, evidenceCount: evidence.length }, reason: 'initial_answer_failed_grounding',
    });
    const correctionInput = this.llmInput(prepared, question, `${prompt}\n${groundingCorrection(firstCheck.reasons)}`);
    const correctedRaw = await this.pipeline.measure(prepared.telemetry, 'llm.grounding_correction', 'generate_corrected_answer', () => this.llm.answer(correctionInput), {
      inputMetrics: { provider: this.llm.name, purpose: 'grounding_correction', streaming: false, correctionReasonCount: firstCheck.reasons.length, contextChunks: correctionInput.context.length },
    });
    const corrected = cleanAnswerScaffolding(correctedRaw);
    const correctedCheck = await this.pipeline.measure(prepared.telemetry, 'grounding.validation', 'validate_corrected_answer', async () => validateGrounding(question, corrected, evidence), {
      inputMetrics: { answerChars: corrected.length, evidenceCount: evidence.length, questionChars: question.length },
    });
    const useCorrected = correctedCheck.ok;
    this.emit(prepared.telemetry, 'pipeline.grounding.correction.completed', 'grounding.correction', 'validate_corrected_answer', useCorrected ? 'success' : 'fallback', {
      inputMetrics: { initialRejectionReasons: firstCheck.reasons, correctedAnswerChars: corrected.length },
      outputMetrics: { accepted: useCorrected, correctedRejectionReasons: correctedCheck.reasons, selected: useCorrected ? 'corrected' : 'fallback' },
      reason: useCorrected ? 'corrected_answer_passed_grounding' : 'corrected_answer_failed_grounding',
    });
    if (useCorrected) {
      this.emit(prepared.telemetry, 'pipeline.answer.selected', 'answer.selection', 'select_corrected_answer', 'success', { outputMetrics: { answerChars: corrected.length, choice: 'corrected' } });
      return corrected;
    }
    this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'select_tenant_fallback', 'fallback', {
      reason: 'initial_and_corrected_answers_failed_grounding', outputMetrics: { fallbackChars: prepared.client.config.fallbackMessage.length, initialRejectionReasons: firstCheck.reasons, correctedRejectionReasons: correctedCheck.reasons },
    });
    return prepared.client.config.fallbackMessage;
  }

  async preview(clientId: string, message: string, prompt: string, traceContext?: PipelineContext) {
    const telemetry = this.context(clientId, 'dashboard-prompt-preview', traceContext);
    this.emit(telemetry, 'pipeline.input.accepted', 'input', 'prompt_preview', 'success', { inputMetrics: { questionChars: message.length, promptChars: prompt.length } });
    const retrieval = await this.retrieve(clientId, message, [], telemetry);
    const prepared = {
      requestId: telemetry.requestId,
      telemetry,
      client: retrieval.client,
      sid: '',
      conversationId: '',
      history: [],
      found: retrieval.evidence,
      decision: retrieval.decision,
      diagnostics: retrieval.diagnostics,
      started: Date.now(),
      summary: false,
    } satisfies Prepared;
    const canGenerate = retrieval.decision.sufficient || clearlyGeneral(message);
    this.emit(telemetry, canGenerate ? 'pipeline.stage.completed' : 'pipeline.stage.skipped', 'answer_generation', 'prompt_preview_generation', canGenerate ? 'success' : 'skipped', {
      reason: canGenerate ? (retrieval.decision.sufficient ? 'evidence_sufficient' : 'general_conversation') : 'evidence_insufficient',
      outputMetrics: { evidenceStatus: retrieval.decision.status, sufficient: retrieval.decision.sufficient },
    });
    const answer = canGenerate ? await this.answerWithGrounding(prepared, message, prompt) : retrieval.client.config.fallbackMessage;
    if (!canGenerate) this.emit(telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'prompt_preview_fallback', 'fallback', { reason: 'evidence_insufficient_before_llm', outputMetrics: { fallbackChars: answer.length } });
    return { answer, retrieval };
  }

  private async prepare(clientId: string, message: string, sessionId?: string, traceContext?: PipelineContext): Promise<Prepared> {
    const telemetry = this.context(clientId, 'chat', traceContext);
    const client = await this.pipeline.measure(telemetry, 'tenant', 'load_client_configuration', () => this.repo.getClient(clientId), { inputMetrics: { tenantScoped: true } });
    if (!client?.enabled) {
      this.emit(telemetry, 'pipeline.stage.failed', 'tenant', 'check_client_enabled', 'failed', { errorCode: 'TENANT_DISABLED', reason: 'tenant_missing_or_disabled' });
      throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    }
    this.pipeline.setTenant(telemetry, client.id);
    const sid = sessionId ?? crypto.randomUUID();
    const conversationId = await this.pipeline.measure(telemetry, 'conversation', 'create_or_resolve_conversation', () => this.repo.createConversation(clientId, sid), {
      inputMetrics: { sessionIdProvided: Boolean(sessionId) },
    });
    const prior = await this.pipeline.measure(telemetry, 'conversation.history', 'load_conversation_history', () => this.repo.messages(conversationId), {
      inputMetrics: { conversationRef: safeRef(conversationId) },
    });
    await this.pipeline.measure(telemetry, 'persistence.user_message', 'persist_user_message', () => this.repo.addMessage(conversationId, 'user', message), {
      inputMetrics: { messageChars: message.length, conversationRef: safeRef(conversationId) },
    });
    const history = boundHistory(prior);
    const historyCharsBefore = prior.reduce((sum, x) => sum + x.content.length, 0);
    const historyCharsAfter = history.reduce((sum, x) => sum + x.content.length, 0);
    this.emit(telemetry, 'pipeline.history.bounded', 'conversation.history', 'bound_history_for_context', 'success', {
      inputMetrics: { messageCountBefore: prior.length, charsBefore: historyCharsBefore },
      outputMetrics: { messageCountAfter: history.length, charsAfter: historyCharsAfter, messageCountExcluded: Math.max(0, prior.length - history.length), charsExcluded: Math.max(0, historyCharsBefore - historyCharsAfter), maxRecentMessages: 8, maxChars: 4000 },
    });
    const started = Date.now();
    const empty: EvidenceDecision = { status: 'NOT_SUPPORTED', coverage: 0, count: 0, sources: 0, sufficient: false };
    const emptyDiagnostics: RetrievalDiagnostics = {
      query: message,
      expanded: false,
      profile: { documents: 0, chunks: 0, vocabulary: 0, topTerms: [] },
      candidateCounts: { exact: 0, keyword: 0, bm25: 0, dense: 0, fused: 0 },
      evidence: empty,
      rejected: [],
    };
    const isSummary = summaryRequest(message);
    const isGeneral = clearlyGeneral(message);
    this.emit(telemetry, 'pipeline.input.classified', 'input.classification', 'classify_request_branch', 'success', {
      inputMetrics: { messageChars: message.length },
      outputMetrics: { summaryRequest: isSummary, clearlyGeneral: isGeneral, historyMessageCount: history.length },
      reason: isSummary ? 'summary_pattern_matched' : isGeneral ? 'general_conversation_pattern_matched' : 'knowledge_retrieval_branch',
    });
    if (isSummary) {
      this.emit(telemetry, 'pipeline.stage.skipped', 'retrieval', 'retrieve_knowledge', 'skipped', { reason: 'summary_request_uses_conversation_history', outputMetrics: { historyMessages: history.length } });
      return { requestId: telemetry.requestId, telemetry, client, sid, conversationId, history, found: [], decision: empty, diagnostics: emptyDiagnostics, started, summary: true };
    }
    if (isGeneral) {
      this.emit(telemetry, 'pipeline.stage.skipped', 'retrieval', 'retrieve_knowledge', 'skipped', { reason: 'clearly_general_conversation', outputMetrics: { historyMessages: history.length } });
      return { requestId: telemetry.requestId, telemetry, client, sid, conversationId, history, found: [], decision: empty, diagnostics: emptyDiagnostics, started, summary: false };
    }
    const retrieval = await this.retrieve(clientId, message, history, telemetry);
    this.emit(telemetry, 'pipeline.retrieval.completed', 'retrieval', 'retrieve_evidence', 'success', {
      inputMetrics: { questionChars: message.length },
      outputMetrics: { evidenceCount: retrieval.evidence.length, evidenceStatus: retrieval.decision.status, evidenceSufficient: retrieval.decision.sufficient, coverage: retrieval.decision.coverage, sourceCount: retrieval.decision.sources, candidateCounts: retrieval.diagnostics.candidateCounts, expanded: retrieval.diagnostics.expanded, selectedChunkRefs: retrieval.evidence.slice(0, 20).map((x) => ({ chunkRef: safeRef(x.id), sourceRef: safeRef(x.source), score: x.score, textChars: x.text.length })) },
    });
    return { requestId: telemetry.requestId, telemetry, client, sid, conversationId, history, found: retrieval.evidence, decision: retrieval.decision, diagnostics: retrieval.diagnostics, started, summary: false };
  }

  private result(client: Client, sid: string, conversationId: string, answer: string, found: RetrievedChunk[]): ChatResult {
    return {
      sessionId: sid,
      conversationId,
      answer,
      sources: found.map((x) => ({ chunkId: x.id, source: x.source, score: Math.round(x.score * 1000) / 1000 })),
      leadCollection: client.config.collectLead ? { enabled: true, fields: ['name', ...(client.config.leadFields ?? ['email', 'requirement']).filter((field) => field !== 'name')] } : undefined,
    };
  }

  private async finalize(prepared: Prepared, answer: string) {
    const cleanStart = process.hrtime.bigint();
    const cleaned = cleanAnswerScaffolding(answer);
    this.emit(prepared.telemetry, 'pipeline.answer.cleaned', 'answer.cleaning', 'clean_final_answer', 'success', {
      durationMs: Number(process.hrtime.bigint() - cleanStart) / 1_000_000,
      inputMetrics: { rawAnswerChars: answer.length }, outputMetrics: { cleanedAnswerChars: cleaned.length, changed: answer !== cleaned },
    });
    const conversationRef = safeRef(prepared.conversationId);
    try {
      await this.pipeline.measure(prepared.telemetry, 'persistence.assistant_message', 'persist_assistant_message', () => this.repo.addMessage(prepared.conversationId, 'assistant', cleaned), {
        inputMetrics: { conversationRef, answerChars: cleaned.length },
      });
      await this.pipeline.measure(prepared.telemetry, 'persistence.usage', 'record_usage_and_latency', () => this.repo.usage(prepared.client.id, 'chat', cleaned.length, Date.now() - prepared.started), {
        inputMetrics: { answerChars: cleaned.length, elapsedMs: Date.now() - prepared.started },
      });
      const result = this.result(prepared.client, prepared.sid, prepared.conversationId, cleaned, prepared.found);
      this.emit(prepared.telemetry, 'pipeline.answer.finalized', 'answer.finalization', 'build_chat_result', 'success', {
        outputMetrics: { answerChars: cleaned.length, sourceCount: result.sources.length, conversationPersisted: true, usageRecorded: true },
      });
      return result;
    } catch (error) {
      this.emit(prepared.telemetry, 'pipeline.persistence.failed', 'persistence', 'finalize_conversation', 'failed', {
        errorCode: safeErrorCode(error), reason: 'assistant_message_or_usage_write_failed', outputMetrics: { conversationRef, answerChars: cleaned.length },
      });
      throw error;
    }
  }

  async chat(clientId: string, message: string, sessionId?: string, traceContext?: PipelineContext): Promise<ChatResult> {
    const prepared = await this.prepare(clientId, message, sessionId, traceContext);
    if (prepared.summary) {
      if (!prepared.history.length) {
        const fallback = tenantFallback(prepared.client);
        this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'summary_without_history', 'fallback', { reason: 'summary_empty_history', outputMetrics: { fallbackChars: fallback.length } });
        return this.finalize(prepared, fallback);
      }
      const answer = await this.pipeline.measure(prepared.telemetry, 'llm.answer_generation', 'generate_conversation_summary', () => this.llm.answer(this.llmInput(prepared, message)), {
        inputMetrics: { provider: this.llm.name, purpose: 'conversation_summary', historyMessages: prepared.history.length },
      });
      return this.finalize(prepared, answer);
    }
    const canUseHistory = historySupportsFollowUp(message, prepared.history);
    this.emit(prepared.telemetry, 'pipeline.history.followup_decision', 'conversation.history', 'evaluate_history_support', 'success', {
      outputMetrics: { canUseHistory, historyMessages: prepared.history.length }, reason: canUseHistory ? 'history_supports_follow_up' : 'history_not_used_as_evidence',
    });
    if (!prepared.decision.sufficient && !clearlyGeneral(message) && !canUseHistory) {
      const fallback = tenantFallback(prepared.client);
      this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'evidence_insufficient_before_llm', 'fallback', {
        reason: 'evidence_insufficient_before_llm', outputMetrics: { evidenceStatus: prepared.decision.status, coverage: prepared.decision.coverage, evidenceCount: prepared.decision.count, fallbackChars: fallback.length, llmSkipped: true },
      });
      this.emit(prepared.telemetry, 'pipeline.stage.skipped', 'llm.answer_generation', 'generate_initial_answer', 'skipped', { reason: 'evidence_insufficient_before_llm' });
      return this.finalize(prepared, fallback);
    }
    return this.finalize(prepared, await this.answerWithGrounding(prepared, message));
  }

  async stream(
    clientId: string,
    message: string,
    sessionId: string | undefined,
    onToken: (token: string) => void,
    traceContext?: PipelineContext,
  ): Promise<ChatResult> {
    const prepared = await this.prepare(clientId, message, sessionId, traceContext);
    if (prepared.summary) {
      if (!prepared.history.length) {
        const fallback = tenantFallback(prepared.client);
        this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'summary_without_history', 'fallback', { reason: 'summary_empty_history', outputMetrics: { fallbackChars: fallback.length } });
        for (const token of answerTokens(fallback)) onToken(token);
        return this.finalize(prepared, fallback);
      }
      let streamedSummary = '';
      await this.pipeline.measure(prepared.telemetry, 'llm.answer_generation', 'stream_conversation_summary_buffered', () => this.llm.streamAnswer(this.llmInput(prepared, message), (token) => { streamedSummary += token; }), {
        inputMetrics: { provider: this.llm.name, purpose: 'conversation_summary', streaming: true, historyMessages: prepared.history.length },
      });
      const cleanedSummary = cleanAnswerScaffolding(streamedSummary);
      for (const token of answerTokens(cleanedSummary)) onToken(token);
      return this.finalize(prepared, cleanedSummary);
    }
    const canUseHistory = historySupportsFollowUp(message, prepared.history);
    this.emit(prepared.telemetry, 'pipeline.history.followup_decision', 'conversation.history', 'evaluate_history_support', 'success', {
      outputMetrics: { canUseHistory, historyMessages: prepared.history.length }, reason: canUseHistory ? 'history_supports_follow_up' : 'history_not_used_as_evidence',
    });
    if (!prepared.decision.sufficient && !clearlyGeneral(message) && !canUseHistory) {
      const fallback = tenantFallback(prepared.client);
      this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'evidence_insufficient_before_llm', 'fallback', {
        reason: 'evidence_insufficient_before_llm', outputMetrics: { evidenceStatus: prepared.decision.status, coverage: prepared.decision.coverage, evidenceCount: prepared.decision.count, fallbackChars: fallback.length, llmSkipped: true },
      });
      for (const token of answerTokens(fallback)) onToken(token);
      return this.finalize(prepared, fallback);
    }

    const evidence = this.groundingEvidence(prepared, message);
    let streamed = '';
    await this.pipeline.measure(prepared.telemetry, 'llm.answer_generation', 'stream_answer_buffered_until_grounding', () => this.llm.streamAnswer(this.llmInput(prepared, message), (token) => { streamed += token; }), {
      inputMetrics: { provider: this.llm.name, purpose: 'answer_generation', streaming: true, bufferedBeforeClientDelivery: true, contextChunks: prepared.found.length },
    });
    const answerBeforeClean = streamed;
    let answer = cleanAnswerScaffolding(streamed);
    this.emit(prepared.telemetry, 'pipeline.answer.cleaned', 'answer.cleaning', 'clean_streamed_answer', 'success', {
      inputMetrics: { rawAnswerChars: answerBeforeClean.length }, outputMetrics: { cleanedAnswerChars: answer.length, changed: answerBeforeClean !== answer },
    });
    const check = await this.pipeline.measure(prepared.telemetry, 'grounding.validation', 'validate_streamed_answer', async () => validateGrounding(message, answer, evidence), {
      inputMetrics: { evidenceCount: evidence.length, answerChars: answer.length, streaming: true },
    });
    this.emit(prepared.telemetry, 'pipeline.grounding.validation.completed', 'grounding.validation', 'validate_streamed_answer', check.ok ? 'success' : 'rejected', {
      outputMetrics: { accepted: check.ok, rejectionReasons: check.reasons, reasonCount: check.reasons.length, mode: 'stream-buffered' },
    });
    if (!check.ok) {
      const correctionPrompt = `${prepared.client.prompt}\n${groundingCorrection(check.reasons)}`;
      const correctionInput = this.llmInput(prepared, message, correctionPrompt);
      const correctedRaw = await this.pipeline.measure(prepared.telemetry, 'llm.grounding_correction', 'generate_stream_correction_nonstreaming', () => this.llm.answer(correctionInput), {
        inputMetrics: { provider: this.llm.name, purpose: 'grounding_correction', streaming: false, correctionReasonCount: check.reasons.length, originalStream: true },
      });
      const corrected = cleanAnswerScaffolding(correctedRaw);
      const correctedCheck = await this.pipeline.measure(prepared.telemetry, 'grounding.validation', 'validate_stream_correction', async () => validateGrounding(message, corrected, evidence), {
        inputMetrics: { evidenceCount: evidence.length, answerChars: corrected.length, streaming: true },
      });
      this.emit(prepared.telemetry, 'pipeline.grounding.correction.completed', 'grounding.correction', 'validate_stream_correction', correctedCheck.ok ? 'success' : 'fallback', {
        inputMetrics: { initialRejectionReasons: check.reasons }, outputMetrics: { accepted: correctedCheck.ok, correctedRejectionReasons: correctedCheck.reasons, selected: correctedCheck.ok ? 'corrected' : 'fallback' },
      });
      answer = correctedCheck.ok ? corrected : prepared.client.config.fallbackMessage;
      if (!correctedCheck.ok) this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'stream_grounding_fallback', 'fallback', { reason: 'corrected_answer_failed_grounding', outputMetrics: { fallbackChars: answer.length } });
    }
    const outputTokens = answerTokens(answer);
    this.emit(prepared.telemetry, 'pipeline.stream.delivery.started', 'response.stream', 'deliver_validated_answer_tokens', 'started', { outputMetrics: { tokenCount: outputTokens.length, answerChars: answer.length } });
    for (const token of outputTokens) onToken(token);
    this.emit(prepared.telemetry, 'pipeline.stream.delivery.completed', 'response.stream', 'deliver_validated_answer_tokens', 'success', { outputMetrics: { tokenCount: outputTokens.length, answerChars: answer.length, groundingPassedOrFallbackSelected: true } });
    return this.finalize(prepared, answer);
  }
}
