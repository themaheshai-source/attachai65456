import crypto from 'node:crypto';
import type { Repository } from '../repositories/repository.js';
import type { EmbeddingProvider } from '../providers/embedding.js';
import type { VectorProvider } from '../providers/vector.js';
import type { LlmProvider } from '../providers/llm.js';
import type { Client, ChatMessage, RetrievedChunk } from '../domain/types.js';
import { AppError } from '../lib/errors.js';
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
  /\b(summarize|summary|recap|what did we discuss|what did i|what you just said|what you just told)\b/i.test(
    message,
  );
const clearlyGeneral = (message: string) =>
  /^(hi|hello|hey|thanks|thank you|okay|ok|good morning|good afternoon|good evening|how are you|tell me a joke|what is the meaning of life)[!.? ]*$/i.test(
    message.trim(),
  );
const followUpReference = (message: string) =>
  /\b(it|its|they|them|their|that|this|those|these|there|he|she|his|her)\b/i.test(message);
const answerTokens = (answer: string) => answer.match(/\S+\s*/g) ?? [];
const redact = (value: string, limit = 1200) =>
  value
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, '[REDACTED_EMAIL]')
    .replace(/(?:\+?\d[\d ()-]{7,}\d)/g, '[REDACTED_PHONE]')
    .slice(0, limit);
const trace = (requestId: string, event: string, data: Record<string, unknown>) =>
  console.log(JSON.stringify({ groundingTrace: true, requestId, event, ...data }));

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

/** Tenant-scoped orchestration. Retrieval is always attempted for the current question; history is bounded and tenant-isolated. */
export class ChatService {
  constructor(
    private repo: Repository,
    private embedding: EmbeddingProvider,
    private vector: VectorProvider,
    private llm: LlmProvider,
  ) {}

  async retrieve(clientId: string, message: string, history: ChatMessage[] = []) {
    const client = await this.repo.getClient(clientId);
    if (!client?.enabled) throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    const topK = Math.min(client.config.topK ?? 20, 40);
    const minSim = client.config.minSimilarity ?? 0.05;
    const [queryVector] = await this.embedding.embed([message]);
    let dense = await this.vector.search(clientId, queryVector!, topK, minSim);
    const corpus =
      typeof this.vector.listTenant === 'function'
        ? await this.vector.listTenant(clientId, 2000)
        : dense;
    let result = await retrieveAdvancedEvidence({ question: message, corpus, dense });

    // If the question is a conversational follow-up with pronouns and initial retrieval was not sufficient,
    // use recent history terms to resolve the entity against the tenant's corpus.
    const followUpQuery = historyContextualQuery(message, history);
    if (!result.decision.sufficient && followUpQuery) {
      const followUpResult = await retrieveAdvancedEvidence({
        question: followUpQuery,
        corpus,
        dense,
      });
      if (followUpResult.decision.sufficient && followUpResult.evidence.length > 0) {
        result = followUpResult;
      }
    }

    // One bounded LLM query reformulation when initial retrieval is weak/insufficient.
    const maxDenseScore = Math.max(0, ...dense.map((x) => x.score));
    if (
      (result.decision.status !== 'SUPPORTED' && maxDenseScore < 0.5) &&
      typeof this.llm.reformulate === 'function'
    ) {
      const reformulated = await this.llm.reformulate({
        question: message,
        prompt: client.prompt,
        context: result.evidence,
        config: client.config,
        history,
      });
      if (reformulated) {
        const firstLine =
          reformulated
            .split(/\r?\n/)
            .map((x) => x.trim())
            .find(Boolean) ?? reformulated;
        const [refVector] = await this.embedding.embed([firstLine]);
        const recovered = await this.vector.search(clientId, refVector!, topK, minSim);
        const mergedMap = new Map<string, RetrievedChunk>();
        for (const item of [...recovered, ...dense]) {
          const existing = mergedMap.get(item.id);
          if (!existing || item.score > existing.score) mergedMap.set(item.id, item);
        }
        dense = [...mergedMap.values()].sort((a, b) => b.score - a.score);
        const mergedCorpus =
          typeof this.vector.listTenant === 'function' ? corpus : dense;
        result = await retrieveAdvancedEvidence({
          question: `${message} ${firstLine}`,
          corpus: mergedCorpus,
          dense,
        });
      }
    }

    return { client, ...result };
  }

  private selectEvidence(candidates: RetrievedChunk[], question: string) {
    // Compatibility helper for callers outside the active pipeline; promotes clearly
    // lexical matches when semantic scores are near-tied while keeping strong semantic leads.
    const queryTerms = new Set(question.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
    return candidates.slice().sort((a, b) => {
      const score = (chunk: RetrievedChunk) =>
        [...queryTerms].filter((term) => chunk.text.toLowerCase().includes(term)).length;
      if (Math.abs(b.score - a.score) < 0.1) {
        return score(b) - score(a) || b.score - a.score;
      }
      return b.score - a.score || score(b) - score(a);
    });
  }

  private groundingEvidence(prepared: Prepared, question: string): RetrievedChunk[] {
    if (historySupportsFollowUp(question, prepared.history)) {
      const historyChunks = prepared.history
        .filter(
          (m) =>
            m.role === 'user' ||
            (!m.content.includes(prepared.client.config.fallbackMessage) &&
              !/provided by|don['’]?t have|no information|not found in/i.test(m.content)),
        )
        .map((m, idx) => ({
          id: `history-${idx}`,
          source: 'conversation-history',
          text: m.content,
          score: 1,
        }));
      return [...prepared.found, ...historyChunks];
    }
    return prepared.found;
  }

  /**
   * Generate once, validate against the exact selected tenant evidence, and allow
   * one bounded correction generation. A failed correction is never shown.
   */
  private async answerWithGrounding(
    prepared: Prepared,
    question: string,
    prompt = prepared.client.prompt,
  ) {
    const evidence = this.groundingEvidence(prepared, question);
    const first = cleanAnswerScaffolding(
      await this.llm.answer(this.input(prepared, question, prompt)),
    );
    const firstCheck = validateGrounding(question, first, evidence);
    trace(prepared.requestId, 'GROUNDING_FIRST', {
      rawAnswer: redact(first),
      validation: firstCheck,
      evidenceCount: evidence.length,
    });
    if (firstCheck.ok) return first;

    const corrected = cleanAnswerScaffolding(
      await this.llm.answer(
        this.input(prepared, question, `${prompt}\n${groundingCorrection(firstCheck.reasons)}`),
      ),
    );
    const correctedCheck = validateGrounding(question, corrected, evidence);
    trace(prepared.requestId, 'GROUNDING_CORRECTION', {
      reasons: firstCheck.reasons,
      correctedAnswer: redact(corrected),
      validation: correctedCheck,
      final: correctedCheck.ok ? 'corrected-answer' : 'fallback',
      fallbackMessage: prepared.client.config.fallbackMessage,
    });
    return correctedCheck.ok ? corrected : prepared.client.config.fallbackMessage;
  }

  async preview(clientId: string, message: string, prompt: string) {
    const retrieval = await this.retrieve(clientId, message);
    const prepared = {
      requestId: crypto.randomUUID(),
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
    const answer =
      retrieval.decision.sufficient || clearlyGeneral(message)
        ? await this.answerWithGrounding(prepared, message, prompt)
        : retrieval.client.config.fallbackMessage;
    return { answer, retrieval };
  }

  private async prepare(clientId: string, message: string, sessionId?: string): Promise<Prepared> {
    const client = await this.repo.getClient(clientId);
    if (!client?.enabled) throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    const sid = sessionId ?? crypto.randomUUID();
    const requestId = crypto.randomUUID();
    trace(requestId, 'REQUEST', {
      tenantId: client.id,
      knowledgeBaseId: client.id,
      question: redact(message, 500),
    });
    const conversationId = await this.repo.createConversation(clientId, sid);
    const prior = await this.repo.messages(conversationId);
    await this.repo.addMessage(conversationId, 'user', message);
    const history = boundHistory(prior);
    const started = Date.now();
    const empty: EvidenceDecision = {
      status: 'NOT_SUPPORTED',
      coverage: 0,
      count: 0,
      sources: 0,
      sufficient: false,
    };
    const emptyDiagnostics: RetrievalDiagnostics = {
      query: message,
      expanded: false,
      profile: { documents: 0, chunks: 0, vocabulary: 0, topTerms: [] },
      candidateCounts: { exact: 0, keyword: 0, bm25: 0, dense: 0, fused: 0 },
      evidence: empty,
      rejected: [],
    };
    if (summaryRequest(message)) {
      const prepared = {
        requestId,
        client,
        sid,
        conversationId,
        history,
        found: [],
        decision: empty,
        diagnostics: emptyDiagnostics,
        started,
        summary: true,
      } satisfies Prepared;
      trace(requestId, 'RETRIEVAL', {
        tenantId: client.id,
        retrievedChunks: 0,
        decision: empty,
        fallbackBranch: 'summary',
      });
      return prepared;
    }
    if (clearlyGeneral(message)) {
      const prepared = {
        requestId,
        client,
        sid,
        conversationId,
        history,
        found: [],
        decision: empty,
        diagnostics: emptyDiagnostics,
        started,
        summary: false,
      } satisfies Prepared;
      trace(requestId, 'RETRIEVAL', {
        tenantId: client.id,
        retrievedChunks: 0,
        decision: empty,
        fallbackBranch: 'general',
      });
      return prepared;
    }
    const retrieval = await this.retrieve(clientId, message, history);
    trace(requestId, 'RETRIEVAL', {
      tenantId: client.id,
      knowledgeBaseId: client.id,
      query: redact(retrieval.diagnostics.query, 500),
      retrievedChunks: retrieval.evidence.length,
      chunks: retrieval.evidence.map((x) => ({
        id: x.id,
        source: x.source,
        score: x.score,
        text: redact(x.text),
      })),
      decision: retrieval.decision,
      diagnostics: retrieval.diagnostics,
    });
    return {
      requestId,
      client,
      sid,
      conversationId,
      history,
      found: retrieval.evidence,
      decision: retrieval.decision,
      diagnostics: retrieval.diagnostics,
      started,
      summary: false,
    };
  }

  private input(prepared: Prepared, question: string, prompt = prepared.client.prompt) {
    return {
      question,
      prompt: `${prompt}\nEvidence decision: ${prepared.decision.status}. Answer only supported portions. If evidence is partial, state the supported portion and clearly qualify what is not available. If evidence is uncertain or absent for a tenant-specific fact, use the configured fallback message.`,
      context: prepared.found,
      config: prepared.client.config,
      history: prepared.history,
    };
  }

  private result(
    client: Client,
    sid: string,
    conversationId: string,
    answer: string,
    found: RetrievedChunk[],
  ): ChatResult {
    return {
      sessionId: sid,
      conversationId,
      answer,
      sources: found.map((x) => ({
        chunkId: x.id,
        source: x.source,
        score: Math.round(x.score * 1000) / 1000,
      })),
      leadCollection: client.config.collectLead
        ? {
            enabled: true,
            fields: [
              'name',
              ...(client.config.leadFields ?? ['email', 'requirement']).filter(
                (field) => field !== 'name',
              ),
            ],
          }
        : undefined,
    };
  }

  private async finalize(prepared: Prepared, answer: string) {
    const cleaned = cleanAnswerScaffolding(answer);
    trace(prepared.requestId, 'FINALIZE_START', {
      answerChars: cleaned.length,
      conversationId: prepared.conversationId,
    });
    try {
      trace(prepared.requestId, 'PERSIST_MESSAGE_START', {
        conversationId: prepared.conversationId,
      });
      await this.repo.addMessage(prepared.conversationId, 'assistant', cleaned);
      trace(prepared.requestId, 'PERSIST_MESSAGE_DONE', {
        conversationId: prepared.conversationId,
      });
      trace(prepared.requestId, 'PERSIST_USAGE_START', {
        tenantId: prepared.client.id,
        answerChars: cleaned.length,
      });
      await this.repo.usage(
        prepared.client.id,
        'chat',
        cleaned.length,
        Date.now() - prepared.started,
      );
      trace(prepared.requestId, 'PERSIST_USAGE_DONE', { tenantId: prepared.client.id });
      const result = this.result(
        prepared.client,
        prepared.sid,
        prepared.conversationId,
        cleaned,
        prepared.found,
      );
      trace(prepared.requestId, 'FINALIZE_DONE', { answerChars: cleaned.length });
      return result;
    } catch (error) {
      const detail =
        error instanceof Error
          ? { name: error.name, message: error.message, stack: error.stack }
          : { name: typeof error, message: String(error), stack: undefined };
      trace(prepared.requestId, 'POST_GROUNDING_EXCEPTION', {
        function: 'ChatService.finalize',
        ...detail,
      });
      throw error;
    }
  }

  async chat(clientId: string, message: string, sessionId?: string): Promise<ChatResult> {
    const prepared = await this.prepare(clientId, message, sessionId);
    if (prepared.summary) {
      if (!prepared.history.length) {
        const fallback = tenantFallback(prepared.client);
        trace(prepared.requestId, 'FALLBACK', { reason: 'summary-empty-history', answer: fallback });
        return this.finalize(prepared, fallback);
      }
      const summaryAnswer = await this.llm.answer(this.input(prepared, message));
      return this.finalize(prepared, summaryAnswer);
    }
    const canUseHistory = historySupportsFollowUp(message, prepared.history);
    if (!prepared.decision.sufficient && !clearlyGeneral(message) && !canUseHistory) {
      const fallback = tenantFallback(prepared.client);
      trace(prepared.requestId, 'FALLBACK', {
        reason: 'evidence-insufficient-before-llm',
        decision: prepared.decision,
        answer: fallback,
      });
      return this.finalize(prepared, fallback);
    }
    return this.finalize(prepared, await this.answerWithGrounding(prepared, message));
  }

  async stream(
    clientId: string,
    message: string,
    sessionId: string | undefined,
    onToken: (token: string) => void,
  ): Promise<ChatResult> {
    const prepared = await this.prepare(clientId, message, sessionId);
    if (prepared.summary) {
      if (!prepared.history.length) {
        const fallback = tenantFallback(prepared.client);
        trace(prepared.requestId, 'FALLBACK', { reason: 'summary-empty-history', answer: fallback });
        for (const token of answerTokens(fallback)) onToken(token);
        return this.finalize(prepared, fallback);
      }
      let streamedSummary = '';
      await this.llm.streamAnswer(this.input(prepared, message), (token) => {
        streamedSummary += token;
      });
      const cleanedSummary = cleanAnswerScaffolding(streamedSummary);
      for (const token of answerTokens(cleanedSummary)) onToken(token);
      return this.finalize(prepared, cleanedSummary);
    }
    const canUseHistory = historySupportsFollowUp(message, prepared.history);
    if (!prepared.decision.sufficient && !clearlyGeneral(message) && !canUseHistory) {
      const fallback = tenantFallback(prepared.client);
      trace(prepared.requestId, 'FALLBACK', {
        reason: 'evidence-insufficient-before-llm',
        decision: prepared.decision,
        answer: fallback,
      });
      for (const token of answerTokens(fallback)) onToken(token);
      return this.finalize(prepared, fallback);
    }

    // Do not expose provider tokens until the complete answer has passed grounding.
    const evidence = this.groundingEvidence(prepared, message);
    let streamed = '';
    await this.llm.streamAnswer(this.input(prepared, message), (token) => {
      streamed += token;
    });
    let answer = cleanAnswerScaffolding(streamed);
    const check = validateGrounding(message, answer, evidence);
    trace(prepared.requestId, 'GROUNDING_FIRST', {
      rawAnswer: redact(answer),
      validation: check,
      evidenceCount: evidence.length,
      mode: 'stream-buffered',
    });
    if (!check.ok) {
      answer = cleanAnswerScaffolding(
        await this.llm.answer(
          this.input(
            prepared,
            message,
            `${prepared.client.prompt}\n${groundingCorrection(check.reasons)}`,
          ),
        ),
      );
      const correctedCheck = validateGrounding(message, answer, evidence);
      trace(prepared.requestId, 'GROUNDING_CORRECTION', {
        reasons: check.reasons,
        correctedAnswer: redact(answer),
        validation: correctedCheck,
        final: correctedCheck.ok ? 'corrected-answer' : 'fallback',
        fallbackMessage: prepared.client.config.fallbackMessage,
        mode: 'stream-buffered',
      });
      if (!correctedCheck.ok) answer = prepared.client.config.fallbackMessage;
    }
    if (answer === prepared.client.config.fallbackMessage && check.ok)
      trace(prepared.requestId, 'FALLBACK', {
        reason: 'provider-returned-configured-fallback',
        answer,
      });
    const outputTokens = answerTokens(answer);
    trace(prepared.requestId, 'STREAM_DELIVERY_START', {
      tokenCount: outputTokens.length,
      answerChars: answer.length,
    });
    for (const token of outputTokens) onToken(token);
    trace(prepared.requestId, 'STREAM_DELIVERY_DONE', { tokenCount: outputTokens.length });
    trace(prepared.requestId, 'FINALIZE_CALL', { branch: 'stream-success' });
    return this.finalize(prepared, answer);
  }
}
