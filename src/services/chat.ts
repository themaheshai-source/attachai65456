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
import { groundingCorrection, validateGrounding, type GroundingCheck, type GroundingDiagnostics, type GroundingRuleDiagnostic } from './grounding.js';
import { isTenantIdentityQuestion, tenantIdentityAnswer } from './identity.js';

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

type ValidationState = 'not_run' | 'accepted' | 'rejected' | 'failed';
type ChatTraceSummary = {
  startedAt: number;
  terminalSummaryEmitted: boolean;
  initialAnswerGenerated: boolean;
  initialValidation: ValidationState;
  initialRejectionReasons: string[];
  correctionAttempted: boolean;
  correctedValidation: ValidationState;
  correctedRejectionReasons: string[];
  fallbackUsed: boolean;
  finalResponseSource: 'initial' | 'corrected' | 'fallback' | 'summary' | 'unknown';
  failedStages: string[];
  validatorSnapshotIds: string[];
};

const summaryRequest = (message: string) =>
  /\b(summarize|summary|recap|what did we discuss|what did i|what you just said|what you just told)\b/i.test(message);
const clearlyGeneral = (message: string) =>
  /^(?:hi|hello|hey|thanks|thank you|okay|ok|good morning|good afternoon|good evening|how are you|tell me a joke|what is the meaning of life|who are you|what are you|what(?:'s| is) your name|tell me about yourself|who created you|who developed you|what company do you belong to|which ai model are you|are you\s+\S+)[!.? ]*$/i.test(
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
  private readonly traceSummaries = new WeakMap<PipelineContext, ChatTraceSummary>();

  constructor(
    private repo: Repository,
    private embedding: EmbeddingProvider,
    private vector: VectorProvider,
    private llm: LlmProvider,
    private pipeline: PipelineObserver = new PipelineObserver({}, 'off'),
  ) {}

  private context(clientId: string, entryPoint = 'service-chat', context?: PipelineContext) {
    const telemetry = context ?? this.pipeline.createContext({ requestId: crypto.randomUUID(), entryPoint, tenantId: clientId });
    if (!this.traceSummaries.has(telemetry)) {
      this.traceSummaries.set(telemetry, {
        startedAt: Date.now(),
        terminalSummaryEmitted: false,
        initialAnswerGenerated: false,
        initialValidation: 'not_run',
        initialRejectionReasons: [],
        correctionAttempted: false,
        correctedValidation: 'not_run',
        correctedRejectionReasons: [],
        fallbackUsed: false,
        finalResponseSource: 'unknown',
        failedStages: [],
        validatorSnapshotIds: [],
      });
    }
    return telemetry;
  }

  private traceSummary(context: PipelineContext): ChatTraceSummary {
    let summary = this.traceSummaries.get(context);
    if (!summary) {
      summary = {
        startedAt: Date.now(),
        terminalSummaryEmitted: false,
        initialAnswerGenerated: false,
        initialValidation: 'not_run',
        initialRejectionReasons: [],
        correctionAttempted: false,
        correctedValidation: 'not_run',
        correctedRejectionReasons: [],
        fallbackUsed: false,
        finalResponseSource: 'unknown',
        failedStages: [],
        validatorSnapshotIds: [],
      };
      this.traceSummaries.set(context, summary);
    }
    return summary;
  }

  private captureDiagnosticValue(value: unknown): unknown {
    if (typeof value === 'string') return this.pipeline.captureText(value);
    if (Array.isArray(value)) return value.map((entry) => this.captureDiagnosticValue(entry));
    if (value && typeof value === 'object') {
      return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, this.captureDiagnosticValue(entry)]));
    }
    return value;
  }

  private captureChunks(chunks: RetrievedChunk[]) {
    return chunks.map((chunk) => ({
      idRef: safeRef(chunk.id),
      id: this.pipeline.captureText(chunk.id),
      sourceRef: safeRef(chunk.source),
      source: this.pipeline.captureText(chunk.source),
      score: chunk.score,
      text: this.pipeline.captureText(chunk.text),
    }));
  }

  private emitCandidateTraces(
    context: PipelineContext,
    event: string,
    stage: string,
    candidateSet: string,
    chunks: RetrievedChunk[],
    scoreType: string,
  ) {
    if (this.pipeline.mode !== 'detailed') return;
    const limit = 40;
    if (!chunks.length) {
      this.pipeline.emitForensic(context, `${event}.summary`, {
        stage, operation: `${candidateSet}_candidates`, status: 'success',
        reason: 'no_candidates_returned', metadata: { candidateSet, candidateCount: 0, loggedCandidateCount: 0, omittedCandidateCount: 0, scoreType },
        output: { candidates: [] },
      });
      return;
    }
    chunks.slice(0, limit).forEach((chunk, index) => {
      this.pipeline.emitForensic(context, event, {
        stage, operation: `${candidateSet}_candidate`, status: 'success',
        metadata: {
          candidateSet, rank: index + 1, candidateCount: chunks.length, scoreType,
          chunkRef: safeRef(chunk.id), sourceRef: safeRef(chunk.source), score: chunk.score,
          textChars: chunk.text.length, textFingerprint: safeRef(chunk.text),
        },
        output: { candidate: { id: this.pipeline.captureText(chunk.id), source: this.pipeline.captureText(chunk.source), text: this.pipeline.captureText(chunk.text) } },
      });
    });
    if (chunks.length > limit) {
      this.pipeline.emitForensic(context, `${event}.truncated`, {
        stage, operation: `${candidateSet}_candidate_trace_limit`, status: 'skipped',
        reason: 'candidate_trace_limit_reached', metadata: { candidateSet, candidateCount: chunks.length, loggedCandidateCount: limit, omittedCandidateCount: chunks.length - limit, scoreType },
      });
    }
  }

  private groundingDiagnostics(
    prepared: Prepared,
    attemptLabel: 'initial' | 'corrected' | 'stream_initial' | 'stream_corrected',
    upstreamAnswer: string,
    upstreamEvidence: RetrievedChunk[],
  ): GroundingDiagnostics {
    let snapshotId = crypto.randomUUID();
    const summary = this.traceSummary(prepared.telemetry);
    const contextMetadata = {
      conversationRef: prepared.conversationId ? safeRef(prepared.conversationId) : undefined,
      attemptLabel,
      tenantScoped: true,
      evidenceCountBeforeValidator: upstreamEvidence.length,
    };
    return {
      onInput: ({ question, answer, evidence }) => {
        // This callback is invoked inside validateGrounding, before its first decision.
        snapshotId = crypto.randomUUID();
        summary.validatorSnapshotIds.push(snapshotId);
        const sameEvidence = upstreamEvidence.length === evidence.length && upstreamEvidence.every((chunk, index) => {
          const received = evidence[index];
          return Boolean(received && chunk.id === received.id && chunk.source === received.source && chunk.score === received.score && chunk.text === received.text);
        });
        const retrievedEvidenceSame = prepared.found.length === evidence.length && prepared.found.every((chunk, index) => {
          const received = evidence[index];
          return Boolean(received && chunk.id === received.id && chunk.source === received.source && chunk.score === received.score && chunk.text === received.text);
        });
        this.pipeline.emitForensic(prepared.telemetry, 'pipeline.grounding.validator.input', {
          stage: 'grounding.validator', operation: 'validate_grounding', status: 'started', snapshotId,
          metadata: {
            ...contextMetadata,
            questionChars: question.length,
            answerChars: answer.length,
            evidenceCount: evidence.length,
            upstreamAnswerDiffers: upstreamAnswer !== answer,
            callerEvidenceSameOrderAndContent: sameEvidence,
            callerEvidenceCount: upstreamEvidence.length,
            retrievedEvidenceSameOrderAndContent: retrievedEvidenceSame,
            retrievedEvidenceCount: prepared.found.length,
            validatorEvidenceCount: evidence.length,
            evidenceCountDeltaFromCaller: evidence.length - upstreamEvidence.length,
          },
          input: {
            question: this.pipeline.captureText(question),
            answer: this.pipeline.captureText(answer),
            evidence: this.captureChunks(evidence),
            upstream: {
              answerSameAsValidatorInput: upstreamAnswer === answer,
              ...(upstreamAnswer === answer ? {} : { answer: this.pipeline.captureText(upstreamAnswer) }),
              evidenceSameAsValidatorInput: sameEvidence,
              ...(sameEvidence ? { evidenceCount: upstreamEvidence.length } : { evidence: this.captureChunks(upstreamEvidence) }),
              retrievedEvidenceSameAsValidatorInput: retrievedEvidenceSame,
              ...(retrievedEvidenceSame ? {} : { retrievedEvidence: this.captureChunks(prepared.found) }),
            },
            evidenceComparison: {
              callerEvidenceSameOrderAndContent: sameEvidence,
              retrievedEvidenceSameOrderAndContent: retrievedEvidenceSame,
              callerEvidenceIds: upstreamEvidence.map((chunk) => safeRef(chunk.id)),
              retrievedEvidenceIds: prepared.found.map((chunk) => safeRef(chunk.id)),
              validatorEvidenceIds: evidence.map((chunk) => safeRef(chunk.id)),
            },
          },
        });
      },
      onRule: (rule: GroundingRuleDiagnostic) => {
        this.pipeline.emitForensic(prepared.telemetry, 'pipeline.grounding.validator.rule.completed', {
          stage: 'grounding.validator.rule', operation: rule.name,
          status: rule.status === 'failed' ? 'rejected' : rule.status === 'error' ? 'failed' : rule.status === 'skipped' ? 'skipped' : 'success',
          snapshotId,
          metadata: {
            ...contextMetadata,
            ruleName: rule.name,
            ruleVersion: rule.version,
            ruleStatus: rule.status,
            ruleResult: rule.result,
            reason: rule.reason,
          },
          input: this.captureDiagnosticValue(rule.input) as Record<string, unknown>,
          output: { ruleName: rule.name, ruleVersion: rule.version, status: rule.status, result: rule.result, reason: rule.reason },
        });
      },
      onOutput: (output) => {
        const isCorrected = attemptLabel === 'corrected' || attemptLabel === 'stream_corrected';
        if (output.status === 'failed') {
          summary.failedStages.push(`grounding.validator.${attemptLabel}`);
          if (isCorrected) summary.correctedValidation = 'failed';
          else summary.initialValidation = 'failed';
        } else if (isCorrected) {
          summary.correctedValidation = output.ok ? 'accepted' : 'rejected';
          summary.correctedRejectionReasons = [...output.reasons];
        } else {
          summary.initialValidation = output.ok ? 'accepted' : 'rejected';
          summary.initialRejectionReasons = [...output.reasons];
        }
        this.pipeline.emitForensic(prepared.telemetry, 'pipeline.grounding.validator.output', {
          stage: 'grounding.validator', operation: 'validate_grounding',
          status: output.status ?? (output.ok ? 'success' : 'rejected'), snapshotId,
          metadata: {
            ...contextMetadata,
            accepted: output.ok,
            classification: output.classification,
            rejectionReasons: output.reasons,
            rejectionReasonCount: output.reasons.length,
          },
          output: {
            accepted: output.ok,
            classification: output.classification,
            rejectionReasons: output.reasons.map((reason) => this.pipeline.captureText(reason)),
          },
        });
      },
    };
  }

  private async validateWithTrace(
    prepared: Prepared,
    question: string,
    answer: string,
    evidence: RetrievedChunk[],
    attemptLabel: 'initial' | 'corrected' | 'stream_initial' | 'stream_corrected',
    upstreamAnswer: string,
    operation: string,
    streaming = false,
  ): Promise<GroundingCheck> {
    try {
      return await this.pipeline.measure(prepared.telemetry, 'grounding.validation', operation, async () => validateGrounding(
        question,
        answer,
        evidence,
        this.groundingDiagnostics(prepared, attemptLabel, upstreamAnswer, evidence),
      ), {
        inputMetrics: { answerChars: answer.length, evidenceCount: evidence.length, questionChars: question.length, streaming },
      });
    } catch (error) {
      const failedStage = `grounding.validation.${operation}`;
      const summary = this.traceSummary(prepared.telemetry);
      if (!summary.failedStages.includes(failedStage)) summary.failedStages.push(failedStage);
      this.emitTraceSummaryForContext(prepared.telemetry, 'failed', answer, safeErrorCode(error), failedStage);
      throw error;
    }
  }

  private selectAnswer(prepared: Prepared, source: ChatTraceSummary['finalResponseSource'], answer: string, reason: string) {
    const summary = this.traceSummary(prepared.telemetry);
    summary.finalResponseSource = source;
    summary.fallbackUsed = source === 'fallback';
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.selection.forensic', {
      stage: 'answer.selection', operation: 'select_final_candidate',
      status: source === 'fallback' ? 'fallback' : 'success', reason,
      metadata: { source, answerChars: answer.length, answerSha256: this.pipeline.captureText(answer).sha256 },
      output: { source, answer: this.pipeline.captureText(answer) },
    });
    return answer;
  }

  private emitTraceSummaryForContext(context: PipelineContext, status: 'success' | 'failed' | 'fallback', finalAnswer: string, reason?: string, failedStage?: string) {
    const summary = this.traceSummary(context);
    if (summary.terminalSummaryEmitted) return;
    summary.terminalSummaryEmitted = true;
    if (failedStage && !summary.failedStages.includes(failedStage)) summary.failedStages.push(failedStage);
    const answerCapture = this.pipeline.captureText(finalAnswer);
    const summaryMetadata: Record<string, unknown> = {
        serviceOutcome: status,
        serviceFailureReason: reason ?? null,
        serviceElapsedMs: Math.max(0, Date.now() - summary.startedAt),
        initialAnswerGenerated: summary.initialAnswerGenerated,
        initialValidation: summary.initialValidation,
        initialRejectionReasons: summary.initialRejectionReasons,
        correctionAttempted: summary.correctionAttempted,
        correctedValidation: summary.correctedValidation,
        correctedRejectionReasons: summary.correctedRejectionReasons,
        fallbackUsed: summary.fallbackUsed,
        finalResponseSource: summary.finalResponseSource,
        finalResponseChars: finalAnswer.length,
        finalResponseSha256: answerCapture.sha256,
        validatorSnapshotIds: summary.validatorSnapshotIds,
        failedStages: summary.failedStages,
      };
    context.traceSummary = summaryMetadata;
    this.pipeline.emitForensic(context, 'pipeline.trace.service.summary', {
      stage: 'pipeline.trace', operation: 'summarize_answer_lifecycle', status,
      reason: reason ?? null,
      durationMs: Math.max(0, Date.now() - summary.startedAt),
      metadata: summaryMetadata,
      output: { finalResponse: answerCapture, finalResponseSource: summary.finalResponseSource },
    });
  }

  private emitTraceSummary(prepared: Prepared, status: 'success' | 'failed' | 'fallback', finalAnswer: string, reason?: string) {
    this.emitTraceSummaryForContext(prepared.telemetry, status, finalAnswer, reason);
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
      if (operation === 'LLM_INPUT_BOUNDARY_FULL' || operation === 'LLM_REFORMULATION_INPUT_FULL') {
        const payload = data.payload && typeof data.payload === 'object' ? data.payload as { model?: unknown; stream?: unknown; messages?: Array<{ role?: unknown; content?: unknown }> } : undefined;
        const messages = Array.isArray(payload?.messages) ? payload.messages : [];
        this.pipeline.emitForensic(context, operation === 'LLM_INPUT_BOUNDARY_FULL' ? 'pipeline.llm.request.input' : 'pipeline.llm.reformulation.request.input', {
          stage: 'llm.request', operation: operation.toLowerCase(), status: 'success',
          metadata: { provider: data.provider, model: data.model ?? payload?.model, purpose: data.purpose, stream: data.stream ?? payload?.stream, messageCount: messages.length, serializedPayloadChars: typeof data.serializedPayload === 'string' ? data.serializedPayload.length : null },
          input: {
            serializedPayload: this.pipeline.captureText(typeof data.serializedPayload === 'string' ? data.serializedPayload : ''),
            payload: payload ? { model: payload.model, stream: payload.stream, messages: messages.map((message) => ({ role: message.role, content: this.pipeline.captureText(typeof message.content === 'string' ? message.content : '') })) } : null,
          },
        });
        return;
      }
      if (operation === 'LLM_RESPONSE_TEXT' || operation === 'LLM_REFORMULATION_RESPONSE_TEXT') {
        const responseText = typeof data.responseText === 'string' ? data.responseText : null;
        this.pipeline.emitForensic(context, operation === 'LLM_RESPONSE_TEXT' ? 'pipeline.llm.response.raw' : 'pipeline.llm.reformulation.response.raw', {
          stage: 'llm.response', operation: operation.toLowerCase(), status: 'success',
          metadata: { provider, purpose: data.purpose, responsePresent: responseText !== null, responseChars: responseText?.length ?? 0 },
          output: { responseText: responseText === null ? null : this.pipeline.captureText(responseText) },
        });
        return;
      }
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
    const candidateKeys = ['exactCandidates', 'keywordCandidates', 'bm25Candidates', 'denseCandidates', 'fusedCandidates', 'selected', 'ranked', 'selectionDecisions', 'rerankRejected', 'rankCutoffExcluded'];
    const rawLists = new Map<string, Array<Record<string, unknown>>>();
    for (const key of candidateKeys) {
      const entries = secured[key];
      if (!Array.isArray(entries)) continue;
      rawLists.set(key, entries as Array<Record<string, unknown>>);
      secured[key] = entries.slice(0, 25).map((value, index) => {
        const item = value as Record<string, unknown>;
        return {
          rank: typeof item.rank === 'number' ? item.rank : index + 1,
          chunkRef: typeof item.id === 'string' ? safeRef(item.id) : undefined,
          score: item.score,
          scoreType: typeof item.scoreType === 'string' ? item.scoreType : undefined,
          sourceChunkScore: typeof item.sourceChunkScore === 'number' ? item.sourceChunkScore : undefined,
          matchType: typeof item.matchType === 'string' ? item.matchType : undefined,
          keywordHits: typeof item.keywordHits === 'number' ? item.keywordHits : undefined,
          textChars: typeof item.textChars === 'number' ? item.textChars : typeof item.text === 'string' ? item.text.length : undefined,
          ...(typeof item.source === 'string' ? { sourceRef: safeRef(item.source) } : {}),
          ...(typeof item.status === 'string' ? { status: item.status } : {}),
          ...(typeof item.reason === 'string' ? { reason: item.reason } : {}),
        };
      });
    }
    const rejected = secured.rejected;
    if (Array.isArray(rejected)) {
      secured.rejected = rejected.slice(0, 25).map((value) => {
        const item = value as Record<string, unknown>;
        return { chunkRef: typeof item.id === 'string' ? safeRef(item.id) : undefined, reason: item.reason };
      });
    }
    const durationMs = typeof secured.durationMs === 'number' ? secured.durationMs : undefined;
    const query = typeof secured.query === 'string' ? secured.query : undefined;
    const expandedQuery = typeof secured.expandedQuery === 'string' ? secured.expandedQuery : undefined;
    delete secured.durationMs;
    delete secured.query;
    delete secured.expandedQuery;
    const skipped = secured.executed === false;
    const forensicMetadata = Object.fromEntries(Object.entries(secured).filter(([key]) => !candidateKeys.includes(key) && key !== 'rejected'));
    const outputCandidateSets: Record<string, unknown> = {};
    const scoreTypes: Record<string, string> = {
      exactCandidates: 'exact_search_rank_or_match', keywordCandidates: 'keyword_overlap_score',
      bm25Candidates: 'bm25_score', denseCandidates: 'provider_returned_score',
      fusedCandidates: 'reciprocal_rank_fusion_score', ranked: 'advanced_rag_rerank_score',
      selected: 'advanced_rag_rerank_score', selectionDecisions: 'context_selection_decision',
      rerankRejected: 'fusion_score_before_rerank_rejection', rankCutoffExcluded: 'advanced_rag_rerank_score',
    };
    for (const [key, entries] of rawLists) {
      outputCandidateSets[key] = entries.slice(0, 25).map((item, index) => ({
        scoreType: typeof item.scoreType === 'string' ? item.scoreType : scoreTypes[key] ?? 'unknown',
        rank: typeof item.rank === 'number' ? item.rank : index + 1,
        chunkRef: typeof item.id === 'string' ? safeRef(item.id) : undefined,
        id: typeof item.id === 'string' ? this.pipeline.captureText(item.id) : null,
        sourceRef: typeof item.source === 'string' ? safeRef(item.source) : undefined,
        source: typeof item.source === 'string' ? this.pipeline.captureText(item.source) : null,
        score: item.score,
        sourceChunkScore: typeof item.sourceChunkScore === 'number' ? item.sourceChunkScore : undefined,
        matchType: typeof item.matchType === 'string' ? item.matchType : undefined,
        keywordHits: typeof item.keywordHits === 'number' ? item.keywordHits : undefined,
        textChars: typeof item.textChars === 'number' ? item.textChars : typeof item.text === 'string' ? item.text.length : undefined,
        textFingerprint: typeof item.text === 'string' ? safeRef(item.text) : undefined,
        ...(typeof item.status === 'string' ? { status: item.status } : {}),
        ...(typeof item.reason === 'string' ? { reason: item.reason } : {}),
        ...(typeof item.text === 'string' && key === 'selected' ? { text: this.pipeline.captureText(item.text) } : {}),
      }));
    }
    this.pipeline.emitForensic(context, 'pipeline.retrieval.candidate_stage', {
      stage: `${prefix.replace(/^pipeline\./, '').replace(/\.$/, '')}.${stage}`.slice(0, 100),
      operation: stage, status: skipped ? 'skipped' : 'success',
      ...(durationMs === undefined ? {} : { durationMs }),
      reason: typeof secured.reason === 'string' ? secured.reason : skipped ? 'operation_not_executed' : null,
      metadata: { ...forensicMetadata, candidateSetCounts: Object.fromEntries([...rawLists].map(([key, entries]) => [key, entries.length])), rejectedCandidateCount: Array.isArray(rejected) ? rejected.length : undefined },
      input: { ...(query === undefined ? {} : { query: this.pipeline.captureText(query) }) },
      output: {
        ...(expandedQuery === undefined ? {} : { expandedQuery: this.pipeline.captureText(expandedQuery) }),
        candidateSets: outputCandidateSets,
        ...(Array.isArray(rejected) ? { rejectedCandidates: rejected.slice(0, 25).map((value) => { const item = value as Record<string, unknown>; return { chunkRef: typeof item.id === 'string' ? safeRef(item.id) : undefined, id: typeof item.id === 'string' ? this.pipeline.captureText(item.id) : null, reason: item.reason }; }) } : {}),
      },
    });
    const ranked = rawLists.get('ranked') ?? [];
    if (stage === 'candidate_generation') {
      const rankedChunks = ranked.filter((item) => typeof item.id === 'string' && typeof item.source === 'string' && typeof item.text === 'string' && typeof item.score === 'number').map((item) => ({ id: item.id as string, source: item.source as string, text: item.text as string, score: item.score as number }));
      this.emitCandidateTraces(context, 'pipeline.retrieval.reranked_candidate', `${prefix}reranking`, 'reranked', rankedChunks, 'advanced_rag_rerank_score');
    }
    if (stage === 'context_selection' || stage === 'context_selection_after_expansion') {
      const selected = rawLists.get('selected') ?? [];
      const selectedChunks = selected.filter((item) => typeof item.id === 'string' && typeof item.source === 'string' && typeof item.text === 'string' && typeof item.score === 'number').map((item) => ({ id: item.id as string, source: item.source as string, text: item.text as string, score: item.score as number }));
      this.emitCandidateTraces(context, 'pipeline.retrieval.context_candidate', `${prefix}context_selection`, stage, selectedChunks, 'advanced_rag_rerank_score');
    }
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
    this.pipeline.emitForensic(telemetry, 'pipeline.input.received', { stage: 'input', operation: 'retrieve_service_entry', status: 'success', metadata: { messageChars: message.length, historyMessageCount: history.length, historyChars: history.reduce((total, entry) => total + entry.content.length, 0) }, input: { question: this.pipeline.captureText(message), history: history.map((entry) => ({ role: entry.role, content: this.pipeline.captureText(entry.content) })) } });
    try {
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
    this.emitCandidateTraces(telemetry, 'pipeline.retrieval.dense.candidate', 'retrieval.dense_search', 'dense', dense, 'provider_returned_score');
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
    this.pipeline.emitForensic(telemetry, 'pipeline.retrieval.followup.query', {
      stage: 'retrieval.followup', operation: 'construct_history_contextual_query',
      status: !result.decision.sufficient && followUpQuery ? 'started' : 'skipped',
      reason: !result.decision.sufficient && followUpQuery ? 'initial_evidence_insufficient_and_history_reference_detected' : result.decision.sufficient ? 'initial_evidence_sufficient' : 'no_history_contextual_query',
      metadata: { originalQueryChars: message.length, followupQueryAvailable: Boolean(followUpQuery), initialEvidenceStatus: result.decision.status, historyMessageCount: history.length },
      input: { originalQuery: this.pipeline.captureText(message) },
      output: { contextualQuery: followUpQuery ? this.pipeline.captureText(followUpQuery) : null },
    });
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
    this.pipeline.emitForensic(telemetry, 'pipeline.retrieval.reformulation.decision', {
      stage: 'retrieval.query_reformulation', operation: 'evaluate_reformulation_trigger',
      status: shouldReformulate && typeof this.llm.reformulate === 'function' ? 'started' : 'skipped',
      reason: typeof this.llm.reformulate !== 'function' ? 'provider_does_not_support_reformulation' : result.decision.status === 'SUPPORTED' ? 'evidence_supported' : maxDenseScore >= 0.5 ? 'dense_score_at_or_above_threshold' : 'evidence_not_supported_and_dense_score_below_threshold',
      metadata: { evidenceStatus: result.decision.status, maxDenseScore, triggerThreshold: 0.5, reformulationSupported: typeof this.llm.reformulate === 'function', triggerSelected: shouldReformulate },
      input: { originalQuery: this.pipeline.captureText(message) },
    });
    if (shouldReformulate && typeof this.llm.reformulate === 'function') {
      this.emit(telemetry, 'pipeline.retrieval.reformulation.started', 'retrieval.query_reformulation', 'reformulate_query', 'started', {
        inputMetrics: { evidenceStatus: result.decision.status, maxDenseScore, evidenceCount: result.evidence.length, historyMessages: history.length, provider: this.llm.name },
      });
      const reformulationInput: LlmInput = { question: message, prompt: client.prompt, context: result.evidence, config: client.config, history, debug: this.providerDebug(telemetry), debugDetailed: this.pipeline.mode === 'detailed', debugFullText: this.pipeline.fullTextEnabled };
      const reformulated = await this.pipeline.measure(telemetry, 'llm.query_reformulation', 'reformulate_retrieval_query', () => this.llm.reformulate!(reformulationInput), {
        inputMetrics: { provider: this.llm.name, questionChars: message.length, evidenceCount: result.evidence.length, historyMessages: history.length },
      });
      this.pipeline.emitForensic(telemetry, 'pipeline.retrieval.reformulation.response', {
        stage: 'retrieval.query_reformulation', operation: 'capture_reformulation_result',
        status: reformulated ? 'success' : 'skipped', reason: reformulated ? 'provider_returned_alternative_queries' : 'provider_returned_no_usable_alternative_query',
        metadata: { provider: this.llm.name, responsePresent: Boolean(reformulated), responseChars: reformulated?.length ?? 0, alternativeQueryCount: reformulated?.split(/\r?\n/).filter(Boolean).length ?? 0 },
        input: { originalQuery: this.pipeline.captureText(message) },
        output: { reformulatedQuery: reformulated ? this.pipeline.captureText(reformulated) : null },
      });
      if (reformulated) {
        const lines = reformulated.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
        const firstLine = lines[0] ?? reformulated;
        this.emit(telemetry, 'pipeline.retrieval.reformulation.completed', 'retrieval.query_reformulation', 'reformulate_query', 'success', {
          outputMetrics: { returned: true, alternativeQueryCount: lines.length, selectedQueryChars: firstLine.length },
        });
        this.pipeline.emitForensic(telemetry, 'pipeline.retrieval.reformulation.selected_query', {
          stage: 'retrieval.query_reformulation', operation: 'select_first_alternative_query', status: 'success',
          reason: lines.length > 1 ? 'only_first_line_is_used_for_retrieval' : 'single_alternative_query_returned',
          metadata: { alternativeQueryCount: lines.length, selectedQueryChars: firstLine.length, selectedLineIndex: 1 },
          input: { reformulatedQuery: this.pipeline.captureText(reformulated) },
          output: { selectedQuery: this.pipeline.captureText(firstLine) },
        });
        const refEmbedding = await this.pipeline.measure(telemetry, 'retrieval.embedding', 'embed_reformulated_query', () => this.embedding.embed([firstLine], this.providerDebug(telemetry, 'embedding')), { inputMetrics: { provider: this.embedding.name, inputCount: 1, inputChars: firstLine.length } });
        const refVector = refEmbedding[0];
        const recovered = await this.pipeline.measure(telemetry, 'retrieval.dense_search', 'search_reformulated_query', () => this.vector.search(clientId, refVector!, topK, minSim, this.providerDebug(telemetry, 'vector')), {
          inputMetrics: { provider: this.vector.name, topK, minSimilarity: minSim, tenantScoped: true, queryVectorDimensions: refVector?.length ?? 0 },
        });
        this.emitCandidateTraces(telemetry, 'pipeline.retrieval.reformulated_dense.candidate', 'retrieval.reformulated_dense_search', 'reformulated_dense', recovered, 'provider_returned_score');
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
        this.emitCandidateTraces(telemetry, 'pipeline.retrieval.merged_dense.candidate', 'retrieval.candidate_merge', 'merged_dense', mergedDense, 'highest_provider_returned_score');
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
    this.pipeline.emitForensic(telemetry, 'pipeline.retrieval.final.selected', {
      stage: 'retrieval.final_selection', operation: 'return_selected_evidence_to_chat_or_preview',
      status: result.evidence.length ? 'success' : 'skipped', reason: result.evidence.length ? null : 'no_evidence_selected',
      metadata: { originalQueryChars: message.length, selectedCount: result.evidence.length, selectedTextChars: result.evidence.reduce((total, chunk) => total + chunk.text.length, 0), evidenceDecision: result.decision, expanded: result.diagnostics.expanded, candidateCounts: result.diagnostics.candidateCounts, rejectedCount: result.diagnostics.rejected.length },
      input: { originalQuery: this.pipeline.captureText(message), resolvedQuery: this.pipeline.captureText(result.diagnostics.query) },
      output: { selectedEvidence: this.captureChunks(result.evidence), rejectedCandidates: result.diagnostics.rejected.slice(0, 25).map((item) => ({ chunkRef: safeRef(item.id), id: this.pipeline.captureText(item.id), reason: item.reason })), selectedOrder: result.evidence.map((chunk, index) => ({ rank: index + 1, chunkRef: safeRef(chunk.id), score: chunk.score, textChars: chunk.text.length })) },
    });
    return { client, ...result };
    } catch (error) {
      this.emitTraceSummaryForContext(telemetry, 'failed', '', safeErrorCode(error), 'chat.retrieve');
      throw error;
    }
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
    return { question, prompt: finalPrompt, context: prepared.found, config: prepared.client.config, history: prepared.history, debug: this.providerDebug(prepared.telemetry), debugDetailed: this.pipeline.mode === 'detailed', debugFullText: this.pipeline.fullTextEnabled };
  }

  private async answerWithGrounding(prepared: Prepared, question: string, prompt = prepared.client.prompt) {
    const identity = tenantIdentityAnswer(question, prepared.client.config);
    if (identity) {
      this.emit(prepared.telemetry, 'pipeline.answer.selected', 'answer.selection', 'select_configured_tenant_identity', 'success', {
        reason: 'configured_tenant_identity', outputMetrics: { answerChars: identity.length, providerSkipped: true },
      });
      return this.selectAnswer(prepared, 'initial', identity, 'configured_tenant_identity');
    }
    const evidence = this.groundingEvidence(prepared, question);
    const llmInput = this.llmInput(prepared, question, prompt);
    this.emit(prepared.telemetry, 'pipeline.llm.call.started', 'llm.answer_generation', 'generate_initial_answer', 'started', {
      inputMetrics: { provider: this.llm.name, purpose: 'answer_generation', streaming: false, contextChunks: llmInput.context.length, historyMessages: llmInput.history?.length ?? 0, questionChars: question.length },
    });
    let generated: string;
    try {
      generated = await this.pipeline.measure(prepared.telemetry, 'llm.answer_generation', 'generate_initial_answer', () => this.llm.answer(llmInput), {
        inputMetrics: { provider: this.llm.name, purpose: 'answer_generation', streaming: false, contextChunks: llmInput.context.length, questionChars: question.length },
      });
    } catch (error) {
      const summary = this.traceSummary(prepared.telemetry);
      const failedStage = 'llm.answer_generation.generate_initial_answer';
      if (!summary.failedStages.includes(failedStage)) summary.failedStages.push(failedStage);

      const fallback = this.selectAnswer(
        prepared,
        'fallback',
        prepared.client.config.fallbackMessage,
        'llm_provider_failure',
      );
      this.emit(
        prepared.telemetry,
        'pipeline.fallback.selected',
        'answer.fallback',
        'llm_provider_failure',
        'fallback',
        {
          reason: safeErrorCode(error),
          outputMetrics: {
            fallbackChars: fallback.length,
            providerFailure: safeErrorCode(error),
            failedStage,
          },
        },
      );
      return fallback;
    }
    const summary = this.traceSummary(prepared.telemetry);
    summary.initialAnswerGenerated = true;
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.raw', {
      stage: 'answer.generation', operation: 'capture_initial_answer_return_value', status: 'success',
      metadata: { provider: this.llm.name, answerChars: generated.length, source: 'llm_provider_return_value' },
      output: { answer: this.pipeline.captureText(generated) },
    });
    const first = cleanAnswerScaffolding(generated);
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.cleaned.forensic', {
      stage: 'answer.cleaning', operation: 'clean_initial_answer', status: 'success',
      metadata: { changed: generated !== first, rawAnswerChars: generated.length, cleanedAnswerChars: first.length },
      input: { answer: this.pipeline.captureText(generated) }, output: { answer: this.pipeline.captureText(first) },
    });
    this.emit(prepared.telemetry, 'pipeline.answer.cleaned', 'answer.cleaning', 'clean_initial_answer', 'success', {
      inputMetrics: { rawAnswerChars: generated.length }, outputMetrics: { cleanedAnswerChars: first.length, changed: generated !== first },
    });
    const firstCheck = await this.validateWithTrace(prepared, question, first, evidence, 'initial', generated, 'validate_initial_answer');
    this.emit(prepared.telemetry, 'pipeline.grounding.validation.completed', 'grounding.validation', 'validate_initial_answer', firstCheck.ok ? 'success' : 'rejected', {
      inputMetrics: { evidenceCount: evidence.length, answerChars: first.length },
      outputMetrics: { accepted: firstCheck.ok, rejectionReasons: firstCheck.reasons, reasonCount: firstCheck.reasons.length },
      reason: firstCheck.ok ? 'answer_passed_grounding' : 'answer_failed_grounding',
    });
    if (firstCheck.ok) {
      this.emit(prepared.telemetry, 'pipeline.answer.selected', 'answer.selection', 'select_initial_answer', 'success', { outputMetrics: { answerChars: first.length, choice: 'original' } });
      return this.selectAnswer(prepared, 'initial', first, 'initial_answer_passed_grounding');
    }

    this.traceSummary(prepared.telemetry).correctionAttempted = true;
    this.emit(prepared.telemetry, 'pipeline.correction.started', 'grounding.correction', 'generate_corrected_answer', 'started', {
      inputMetrics: { rejectionReasons: firstCheck.reasons, evidenceCount: evidence.length }, reason: 'initial_answer_failed_grounding',
    });
    const correctionInstruction = groundingCorrection(firstCheck.reasons);
    const correctionInput = this.llmInput(prepared, question, `${prompt}\n${correctionInstruction}`);
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.grounding.correction.prompt', {
      stage: 'grounding.correction', operation: 'assemble_correction_input', status: 'success',
      metadata: { provider: this.llm.name, initialValidatorSnapshotId: this.traceSummary(prepared.telemetry).validatorSnapshotIds.at(-1), rejectionReasonCount: firstCheck.reasons.length, contextCount: correctionInput.context.length },
      input: { originalAnswer: this.pipeline.captureText(first), rejectionReasons: firstCheck.reasons.map((reason) => this.pipeline.captureText(reason)), correctionInstruction: this.pipeline.captureText(correctionInstruction), finalApplicationPrompt: this.pipeline.captureText(correctionInput.prompt), question: this.pipeline.captureText(correctionInput.question), evidence: this.captureChunks(correctionInput.context) },
    });
    let correctedRaw: string;
    try {
      correctedRaw = await this.pipeline.measure(prepared.telemetry, 'llm.grounding_correction', 'generate_corrected_answer', () => this.llm.answer(correctionInput), {
        inputMetrics: { provider: this.llm.name, purpose: 'grounding_correction', streaming: false, correctionReasonCount: firstCheck.reasons.length, contextChunks: correctionInput.context.length },
      });
    } catch (error) {
      const summary = this.traceSummary(prepared.telemetry);
      const failedStage = 'llm.grounding_correction.generate_corrected_answer';
      if (!summary.failedStages.includes(failedStage)) summary.failedStages.push(failedStage);

      // The initial answer already failed grounding. Never return it just because
      // the correction call timed out; return the configured fallback instead.
      const fallback = this.selectAnswer(
        prepared,
        'fallback',
        prepared.client.config.fallbackMessage,
        'llm_correction_provider_failure',
      );
      this.emit(
        prepared.telemetry,
        'pipeline.fallback.selected',
        'answer.fallback',
        'llm_correction_provider_failure',
        'fallback',
        {
          reason: safeErrorCode(error),
          outputMetrics: {
            fallbackChars: fallback.length,
            providerFailure: safeErrorCode(error),
            failedStage,
          },
        },
      );
      return fallback;
    }
    const corrected = cleanAnswerScaffolding(correctedRaw);
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.corrected.raw', {
      stage: 'grounding.correction', operation: 'capture_corrected_answer_return_value', status: 'success',
      metadata: { provider: this.llm.name, answerChars: correctedRaw.length }, output: { answer: this.pipeline.captureText(correctedRaw) },
    });
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.corrected.cleaned', {
      stage: 'answer.cleaning', operation: 'clean_corrected_answer', status: 'success',
      metadata: { changed: correctedRaw !== corrected, rawAnswerChars: correctedRaw.length, cleanedAnswerChars: corrected.length },
      input: { answer: this.pipeline.captureText(correctedRaw) }, output: { answer: this.pipeline.captureText(corrected) },
    });
    const correctedCheck = await this.validateWithTrace(prepared, question, corrected, evidence, 'corrected', correctedRaw, 'validate_corrected_answer');
    const useCorrected = correctedCheck.ok;
    this.emit(prepared.telemetry, 'pipeline.grounding.correction.completed', 'grounding.correction', 'validate_corrected_answer', useCorrected ? 'success' : 'fallback', {
      inputMetrics: { initialRejectionReasons: firstCheck.reasons, correctedAnswerChars: corrected.length },
      outputMetrics: { accepted: useCorrected, correctedRejectionReasons: correctedCheck.reasons, selected: useCorrected ? 'corrected' : 'fallback' },
      reason: useCorrected ? 'corrected_answer_passed_grounding' : 'corrected_answer_failed_grounding',
    });
    if (useCorrected) {
      this.emit(prepared.telemetry, 'pipeline.answer.selected', 'answer.selection', 'select_corrected_answer', 'success', { outputMetrics: { answerChars: corrected.length, choice: 'corrected' } });
      return this.selectAnswer(prepared, 'corrected', corrected, 'corrected_answer_passed_grounding');
    }
    this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'select_tenant_fallback', 'fallback', {
      reason: 'initial_and_corrected_answers_failed_grounding', outputMetrics: { fallbackChars: prepared.client.config.fallbackMessage.length, initialRejectionReasons: firstCheck.reasons, correctedRejectionReasons: correctedCheck.reasons },
    });
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.fallback.text.selected', {
      stage: 'answer.fallback', operation: 'select_tenant_fallback', status: 'fallback',
      metadata: { source: 'client.config.fallbackMessage', fallbackChars: prepared.client.config.fallbackMessage.length, initialRejectionReasons: firstCheck.reasons, correctedRejectionReasons: correctedCheck.reasons, validated: false },
      output: { fallbackText: this.pipeline.captureText(prepared.client.config.fallbackMessage) },
    });
    return this.selectAnswer(prepared, 'fallback', prepared.client.config.fallbackMessage, 'initial_and_corrected_answers_failed_grounding');
  }

  async preview(clientId: string, message: string, prompt: string, traceContext?: PipelineContext) {
    const telemetry = this.context(clientId, 'dashboard-prompt-preview', traceContext);
    this.pipeline.emitForensic(telemetry, 'pipeline.input.received', { stage: 'input', operation: 'prompt_preview_service_entry', status: 'success', metadata: { questionChars: message.length, promptChars: prompt.length }, input: { question: this.pipeline.captureText(message), applicationPrompt: this.pipeline.captureText(prompt) } });
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
    const canGenerate = retrieval.decision.sufficient || clearlyGeneral(message) || isTenantIdentityQuestion(message);
    this.emit(telemetry, canGenerate ? 'pipeline.stage.completed' : 'pipeline.stage.skipped', 'answer_generation', 'prompt_preview_generation', canGenerate ? 'success' : 'skipped', {
      reason: canGenerate ? (retrieval.decision.sufficient ? 'evidence_sufficient' : isTenantIdentityQuestion(message) ? 'configured_tenant_identity' : 'general_conversation') : 'evidence_insufficient',
      outputMetrics: { evidenceStatus: retrieval.decision.status, sufficient: retrieval.decision.sufficient },
    });
    const answer = canGenerate ? await this.answerWithGrounding(prepared, message, prompt) : this.selectAnswer(prepared, 'fallback', retrieval.client.config.fallbackMessage, 'evidence_insufficient_before_llm');
    if (!canGenerate) this.emit(telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'prompt_preview_fallback', 'fallback', { reason: 'evidence_insufficient_before_llm', outputMetrics: { fallbackChars: answer.length } });
    this.pipeline.emitForensic(telemetry, 'pipeline.response.finalized', { stage: 'response.finalization', operation: 'prompt_preview_return_value', status: this.traceSummary(telemetry).fallbackUsed ? 'fallback' : 'success', metadata: { answerChars: answer.length, sourceCount: retrieval.evidence.length, deliveryStage: 'admin_route_return_value_pending' }, output: { answer: this.pipeline.captureText(answer) } });
    this.emitTraceSummary(prepared, this.traceSummary(telemetry).fallbackUsed ? 'fallback' : 'success', answer);
    return { answer, retrieval };
  }

  private async prepare(clientId: string, message: string, sessionId?: string, traceContext?: PipelineContext): Promise<Prepared> {
    const telemetry = this.context(clientId, 'chat', traceContext);
    this.pipeline.emitForensic(telemetry, 'pipeline.input.received', { stage: 'input', operation: 'chat_service_entry', status: 'success', metadata: { messageChars: message.length, sessionIdProvided: Boolean(sessionId) }, input: { question: this.pipeline.captureText(message) } });
    try {
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
    const isGeneral = clearlyGeneral(message) || isTenantIdentityQuestion(message);
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
    } catch (error) {
      this.emitTraceSummaryForContext(telemetry, 'failed', '', safeErrorCode(error), 'chat.prepare');
      throw error;
    }
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
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.final.response.transformed', {
      stage: 'response.finalization', operation: 'clean_final_answer', status: 'success',
      metadata: { source: this.traceSummary(prepared.telemetry).finalResponseSource, inputChars: answer.length, outputChars: cleaned.length, changed: answer !== cleaned },
      input: { answer: this.pipeline.captureText(answer) }, output: { answer: this.pipeline.captureText(cleaned) },
    });
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
      this.pipeline.emitForensic(prepared.telemetry, 'pipeline.response.finalized', {
        stage: 'response.finalization', operation: 'build_chat_result', status: this.traceSummary(prepared.telemetry).fallbackUsed ? 'fallback' : 'success',
        metadata: { answerChars: cleaned.length, sourceCount: result.sources.length, conversationPersisted: true, usageRecorded: true, responseObjectReturnedByService: true },
        output: { answer: this.pipeline.captureText(cleaned), sourceCount: result.sources.length, sources: result.sources.slice(0, 25).map((source) => ({ chunkRef: safeRef(source.chunkId), sourceRef: safeRef(source.source), score: source.score })) },
      });
      this.emitTraceSummary(prepared, this.traceSummary(prepared.telemetry).fallbackUsed ? 'fallback' : 'success', cleaned);
      return result;
    } catch (error) {
      this.emit(prepared.telemetry, 'pipeline.persistence.failed', 'persistence', 'finalize_conversation', 'failed', {
        errorCode: safeErrorCode(error), reason: 'assistant_message_or_usage_write_failed', outputMetrics: { conversationRef, answerChars: cleaned.length },
      });
      this.traceSummary(prepared.telemetry).failedStages.push('persistence.finalize_conversation');
      this.emitTraceSummary(prepared, 'failed', cleaned, 'assistant_message_or_usage_write_failed');
      throw error;
    }
  }

  async chat(clientId: string, message: string, sessionId?: string, traceContext?: PipelineContext): Promise<ChatResult> {
    const prepared = await this.prepare(clientId, message, sessionId, traceContext);
    if (prepared.summary) {
      if (!prepared.history.length) {
        const fallback = tenantFallback(prepared.client);
        this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'summary_without_history', 'fallback', { reason: 'summary_empty_history', outputMetrics: { fallbackChars: fallback.length } });
        return this.finalize(prepared, this.selectAnswer(prepared, 'fallback', fallback, 'summary_empty_history'));
      }
      let answer: string;
      try {
        answer = await this.pipeline.measure(prepared.telemetry, 'llm.answer_generation', 'generate_conversation_summary', () => this.llm.answer(this.llmInput(prepared, message)), {
          inputMetrics: { provider: this.llm.name, purpose: 'conversation_summary', historyMessages: prepared.history.length },
        });
      } catch (error) {
        const summary = this.traceSummary(prepared.telemetry);
        const failedStage = 'llm.answer_generation.generate_conversation_summary';
        if (!summary.failedStages.includes(failedStage)) summary.failedStages.push(failedStage);
        const fallback = tenantFallback(prepared.client);
        const selectedFallback = this.selectAnswer(prepared, 'fallback', fallback, 'summary_provider_failure');
        this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'summary_provider_failure', 'fallback', {
          reason: safeErrorCode(error),
          outputMetrics: { fallbackChars: fallback.length, providerFailure: safeErrorCode(error), failedStage },
        });
        return this.finalize(prepared, selectedFallback);
      }
      this.traceSummary(prepared.telemetry).initialAnswerGenerated = true;
      return this.finalize(prepared, this.selectAnswer(prepared, 'summary', answer, 'conversation_summary_generated'));
    }
    const canUseHistory = historySupportsFollowUp(message, prepared.history);
    this.emit(prepared.telemetry, 'pipeline.history.followup_decision', 'conversation.history', 'evaluate_history_support', 'success', {
      outputMetrics: { canUseHistory, historyMessages: prepared.history.length }, reason: canUseHistory ? 'history_supports_follow_up' : 'history_not_used_as_evidence',
    });
    if (!prepared.decision.sufficient && !clearlyGeneral(message) && !isTenantIdentityQuestion(message) && !canUseHistory) {
      const fallback = tenantFallback(prepared.client);
      this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'evidence_insufficient_before_llm', 'fallback', {
        reason: 'evidence_insufficient_before_llm', outputMetrics: { evidenceStatus: prepared.decision.status, coverage: prepared.decision.coverage, evidenceCount: prepared.decision.count, fallbackChars: fallback.length, llmSkipped: true },
      });
      this.emit(prepared.telemetry, 'pipeline.stage.skipped', 'llm.answer_generation', 'generate_initial_answer', 'skipped', { reason: 'evidence_insufficient_before_llm' });
      return this.finalize(prepared, this.selectAnswer(prepared, 'fallback', fallback, 'evidence_insufficient_before_llm'));
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
        const selectedFallback = this.selectAnswer(prepared, 'fallback', fallback, 'summary_empty_history');
        for (const token of answerTokens(selectedFallback)) onToken(token);
        return this.finalize(prepared, selectedFallback);
      }
      let streamedSummary = '';
      try {
        await this.pipeline.measure(prepared.telemetry, 'llm.answer_generation', 'stream_conversation_summary_buffered', () => this.llm.streamAnswer(this.llmInput(prepared, message), (token) => { streamedSummary += token; }), {
          inputMetrics: { provider: this.llm.name, purpose: 'conversation_summary', streaming: true, historyMessages: prepared.history.length },
        });
      } catch (error) {
        const summary = this.traceSummary(prepared.telemetry);
        const failedStage = 'llm.answer_generation.stream_conversation_summary_buffered';
        if (!summary.failedStages.includes(failedStage)) summary.failedStages.push(failedStage);
        const fallback = this.selectAnswer(prepared, 'fallback', tenantFallback(prepared.client), 'stream_summary_provider_failure');
        this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'stream_summary_provider_failure', 'fallback', {
          reason: safeErrorCode(error),
          outputMetrics: { fallbackChars: fallback.length, providerFailure: safeErrorCode(error), failedStage },
        });
        for (const token of answerTokens(fallback)) onToken(token);
        return this.finalize(prepared, fallback);
      }
      const cleanedSummary = cleanAnswerScaffolding(streamedSummary);
      this.traceSummary(prepared.telemetry).initialAnswerGenerated = true;
      this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.raw', { stage: 'answer.generation', operation: 'capture_streamed_summary', status: 'success', metadata: { answerChars: streamedSummary.length, source: 'llm_provider_stream_return_value' }, output: { answer: this.pipeline.captureText(streamedSummary) } });
      const selectedSummary = this.selectAnswer(prepared, 'summary', cleanedSummary, 'conversation_summary_generated');
      for (const token of answerTokens(selectedSummary)) onToken(token);
      return this.finalize(prepared, selectedSummary);
    }
    const identity = tenantIdentityAnswer(message, prepared.client.config);
    if (identity) {
      const selected = cleanAnswerScaffolding(this.selectAnswer(prepared, 'initial', identity, 'configured_tenant_identity'));
      this.emit(prepared.telemetry, 'pipeline.answer.selected', 'answer.selection', 'select_configured_tenant_identity', 'success', {
        reason: 'configured_tenant_identity', outputMetrics: { answerChars: selected.length, providerSkipped: true, streaming: true },
      });
      for (const token of answerTokens(selected)) onToken(token);
      return this.finalize(prepared, selected);
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
      const selectedFallback = this.selectAnswer(prepared, 'fallback', fallback, 'evidence_insufficient_before_llm');
      this.pipeline.emitForensic(prepared.telemetry, 'pipeline.fallback.text.selected', { stage: 'answer.fallback', operation: 'select_tenant_fallback', status: 'fallback', metadata: { source: 'tenantFallback', validated: false, fallbackChars: selectedFallback.length }, output: { fallbackText: this.pipeline.captureText(selectedFallback) } });
      for (const token of answerTokens(selectedFallback)) onToken(token);
      return this.finalize(prepared, selectedFallback);
    }

    const evidence = this.groundingEvidence(prepared, message);
    let streamed = '';
    try {
      await this.pipeline.measure(prepared.telemetry, 'llm.answer_generation', 'stream_answer_buffered_until_grounding', () => this.llm.streamAnswer(this.llmInput(prepared, message), (token) => { streamed += token; }), {
        inputMetrics: { provider: this.llm.name, purpose: 'answer_generation', streaming: true, bufferedBeforeClientDelivery: true, contextChunks: prepared.found.length },
      });
    } catch (error) {
      const summary = this.traceSummary(prepared.telemetry);
      const failedStage = 'llm.answer_generation.stream_answer_buffered_until_grounding';
      if (!summary.failedStages.includes(failedStage)) summary.failedStages.push(failedStage);
      const fallback = this.selectAnswer(prepared, 'fallback', prepared.client.config.fallbackMessage, 'stream_llm_provider_failure');
      this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'stream_llm_provider_failure', 'fallback', {
        reason: safeErrorCode(error),
        outputMetrics: { fallbackChars: fallback.length, providerFailure: safeErrorCode(error), failedStage },
      });
      for (const token of answerTokens(fallback)) onToken(token);
      return this.finalize(prepared, fallback);
    }
    const answerBeforeClean = streamed;
    this.traceSummary(prepared.telemetry).initialAnswerGenerated = true;
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.raw', { stage: 'answer.generation', operation: 'capture_streamed_answer_return_value', status: 'success', metadata: { provider: this.llm.name, answerChars: answerBeforeClean.length, source: 'llm_provider_stream_return_value' }, output: { answer: this.pipeline.captureText(answerBeforeClean) } });
    let answer = cleanAnswerScaffolding(streamed);
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.cleaned.forensic', { stage: 'answer.cleaning', operation: 'clean_streamed_answer', status: 'success', metadata: { rawAnswerChars: answerBeforeClean.length, cleanedAnswerChars: answer.length, changed: answerBeforeClean !== answer }, input: { answer: this.pipeline.captureText(answerBeforeClean) }, output: { answer: this.pipeline.captureText(answer) } });
    this.emit(prepared.telemetry, 'pipeline.answer.cleaned', 'answer.cleaning', 'clean_streamed_answer', 'success', {
      inputMetrics: { rawAnswerChars: answerBeforeClean.length }, outputMetrics: { cleanedAnswerChars: answer.length, changed: answerBeforeClean !== answer },
    });
    const check = await this.validateWithTrace(prepared, message, answer, evidence, 'stream_initial', streamed, 'validate_streamed_answer', true);
    this.emit(prepared.telemetry, 'pipeline.grounding.validation.completed', 'grounding.validation', 'validate_streamed_answer', check.ok ? 'success' : 'rejected', {
      outputMetrics: { accepted: check.ok, rejectionReasons: check.reasons, reasonCount: check.reasons.length, mode: 'stream-buffered' },
    });
    if (!check.ok) {
      this.traceSummary(prepared.telemetry).correctionAttempted = true;
      const correctionInstruction = groundingCorrection(check.reasons);
      const correctionPrompt = `${prepared.client.prompt}\n${correctionInstruction}`;
      const correctionInput = this.llmInput(prepared, message, correctionPrompt);
      this.pipeline.emitForensic(prepared.telemetry, 'pipeline.grounding.correction.prompt', { stage: 'grounding.correction', operation: 'assemble_stream_correction_input', status: 'success', metadata: { provider: this.llm.name, initialValidatorSnapshotId: this.traceSummary(prepared.telemetry).validatorSnapshotIds.at(-1), rejectionReasonCount: check.reasons.length, contextCount: correctionInput.context.length }, input: { originalAnswer: this.pipeline.captureText(answer), rejectionReasons: check.reasons.map((reason) => this.pipeline.captureText(reason)), correctionInstruction: this.pipeline.captureText(correctionInstruction), finalApplicationPrompt: this.pipeline.captureText(correctionInput.prompt), question: this.pipeline.captureText(correctionInput.question), evidence: this.captureChunks(correctionInput.context) } });
      let correctedRaw: string;
      try {
        correctedRaw = await this.pipeline.measure(prepared.telemetry, 'llm.grounding_correction', 'generate_stream_correction_nonstreaming', () => this.llm.answer(correctionInput), {
          inputMetrics: { provider: this.llm.name, purpose: 'grounding_correction', streaming: false, correctionReasonCount: check.reasons.length, originalStream: true },
        });
      } catch (error) {
        const summary = this.traceSummary(prepared.telemetry);
        const failedStage = 'llm.grounding_correction.generate_stream_correction_nonstreaming';
        if (!summary.failedStages.includes(failedStage)) summary.failedStages.push(failedStage);
        // The initial streamed answer failed grounding. Never send it as a fallback.
        const fallback = this.selectAnswer(prepared, 'fallback', prepared.client.config.fallbackMessage, 'stream_correction_provider_failure');
        this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'stream_correction_provider_failure', 'fallback', {
          reason: safeErrorCode(error),
          outputMetrics: { fallbackChars: fallback.length, providerFailure: safeErrorCode(error), failedStage },
        });
        for (const token of answerTokens(fallback)) onToken(token);
        return this.finalize(prepared, fallback);
      }
      const corrected = cleanAnswerScaffolding(correctedRaw);
      this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.corrected.raw', { stage: 'grounding.correction', operation: 'capture_stream_correction_return_value', status: 'success', metadata: { provider: this.llm.name, answerChars: correctedRaw.length }, output: { answer: this.pipeline.captureText(correctedRaw) } });
      this.pipeline.emitForensic(prepared.telemetry, 'pipeline.answer.corrected.cleaned', { stage: 'answer.cleaning', operation: 'clean_stream_correction', status: 'success', metadata: { rawAnswerChars: correctedRaw.length, cleanedAnswerChars: corrected.length, changed: correctedRaw !== corrected }, input: { answer: this.pipeline.captureText(correctedRaw) }, output: { answer: this.pipeline.captureText(corrected) } });
      const correctedCheck = await this.validateWithTrace(prepared, message, corrected, evidence, 'stream_corrected', correctedRaw, 'validate_stream_correction', true);
      this.emit(prepared.telemetry, 'pipeline.grounding.correction.completed', 'grounding.correction', 'validate_stream_correction', correctedCheck.ok ? 'success' : 'fallback', {
        inputMetrics: { initialRejectionReasons: check.reasons }, outputMetrics: { accepted: correctedCheck.ok, correctedRejectionReasons: correctedCheck.reasons, selected: correctedCheck.ok ? 'corrected' : 'fallback' },
      });
      if (correctedCheck.ok) {
        answer = this.selectAnswer(prepared, 'corrected', corrected, 'stream_correction_passed_grounding');
      } else {
        answer = prepared.client.config.fallbackMessage;
        this.emit(prepared.telemetry, 'pipeline.fallback.selected', 'answer.fallback', 'stream_grounding_fallback', 'fallback', { reason: 'corrected_answer_failed_grounding', outputMetrics: { fallbackChars: answer.length } });
        this.pipeline.emitForensic(prepared.telemetry, 'pipeline.fallback.text.selected', { stage: 'answer.fallback', operation: 'stream_grounding_fallback', status: 'fallback', metadata: { source: 'client.config.fallbackMessage', validated: false, initialRejectionReasons: check.reasons, correctedRejectionReasons: correctedCheck.reasons, fallbackChars: answer.length }, output: { fallbackText: this.pipeline.captureText(answer) } });
        answer = this.selectAnswer(prepared, 'fallback', answer, 'stream_initial_and_corrected_answers_failed_grounding');
      }
    } else {
      answer = this.selectAnswer(prepared, 'initial', answer, 'streamed_answer_passed_grounding');
    }
    const outputTokens = answerTokens(answer);
    this.pipeline.emitForensic(prepared.telemetry, 'pipeline.response.pre_delivery', { stage: 'response.stream', operation: 'prepare_final_stream_answer', status: this.traceSummary(prepared.telemetry).fallbackUsed ? 'fallback' : 'success', metadata: { finalResponseSource: this.traceSummary(prepared.telemetry).finalResponseSource, answerChars: answer.length, tokenCount: outputTokens.length }, output: { answer: this.pipeline.captureText(answer) } });
    this.emit(prepared.telemetry, 'pipeline.stream.delivery.started', 'response.stream', 'deliver_validated_answer_tokens', 'started', { outputMetrics: { tokenCount: outputTokens.length, answerChars: answer.length } });
    for (const token of outputTokens) onToken(token);
    this.emit(prepared.telemetry, 'pipeline.stream.delivery.completed', 'response.stream', 'deliver_validated_answer_tokens', 'success', { outputMetrics: { tokenCount: outputTokens.length, answerChars: answer.length, groundingPassedOrFallbackSelected: true } });
    return this.finalize(prepared, answer);
  }
}
