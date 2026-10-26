import crypto from 'node:crypto';
import type { ChatMessage, ClientConfig, ProviderHealth, RetrievedChunk } from '../domain/types.js';
import type { LlmConfig } from '../config/llm.js';
import { tokens } from '../services/advanced-rag.js';
export type ChatDebug = (stage: string, data: Record<string, unknown>) => void;
export interface LlmInput { question: string; prompt: string; context: RetrievedChunk[]; config: ClientConfig; history?: ChatMessage[]; debug?: ChatDebug; debugDetailed?: boolean }
export interface LlmProvider {
  readonly name: string;
  answer(input: LlmInput): Promise<string>;
  streamAnswer(input: LlmInput, onToken: (token: string) => void): Promise<string>;
  /** One bounded, tenant-scoped retrieval-query reformulation after an initial miss. */
  reformulate?(input: LlmInput): Promise<string | null>;
  health(): Promise<ProviderHealth>;
}
type Fetch = typeof fetch;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const retryable = (status: number) => [408, 409, 425, 429].includes(status) || status >= 500;
const approxTokens = (value: string) => Math.ceil(value.length / 4);
const fingerprint = (value: string) => crypto.createHash('sha256').update(value).digest('hex').slice(0, 16);

/** Customer-facing identity is tenant configuration, never the model/provider. */
export function tenantIdentityPolicy(config: ClientConfig) {
  const assistant = config.assistantName.trim();
  const company = config.companyName?.trim();
  const tenant = company ? `the company ${JSON.stringify(company)}` : 'the current tenant';
  return [
    `You are the customer-facing AI assistant for ${tenant}.`,
    `Your configured assistant identity is ${JSON.stringify(assistant)}.`,
    'Use that tenant identity when describing who you are, what you are, your name, your creator, or the company you belong to.',
    'Do not present the underlying language model, vendor, provider, or implementation as your customer-facing identity.',
    'If asked whether you are a particular AI model or provider, politely clarify that you are the tenant-branded assistant and explain your role instead.',
    'Do not invent a creator or disclose implementation details unless the tenant knowledge explicitly supports a customer-facing answer.',
  ].join(' ');
}

function explicitIdentityQuestion(question: string) {
  const text = question.trim().replace(/[!?.,]+$/g, '');
  return /^(who are you|what are you|what(?:'s| is) your name|tell me about yourself|who created you|who developed you|what company do you belong to|which ai model are you|are you\s+\S+)$/i.test(text);
}

function tenantIdentityAnswer(question: string, config: ClientConfig) {
  if (!explicitIdentityQuestion(question)) return null;
  const assistant = config.assistantName.trim();
  const company = config.companyName?.trim();
  if (company) return `I’m ${company}'s AI assistant, ${assistant}. I’m here to help with ${company}'s products, services, and information.`;
  return `I’m ${assistant}. I’m here to help with this tenant’s products, services, and information.`;
}

const QUESTION_STOPWORDS = new Set([
  'what', 'how', 'can', 'who', 'where', 'when', 'why', 'which', 'does', 'are', 'the',
  'for', 'with', 'your', 'you', 'our', 'its', 'this', 'that', 'have', 'has', 'from',
  'tell', 'about', 'please', 'could', 'would', 'should', 'there', 'their', 'they',
]);

export class LocalLlm implements LlmProvider {
  readonly name: string;
  constructor(private providerName = 'local') { this.name = providerName === 'local' ? 'local-extractive' : providerName; }
  async answer(i: LlmInput) {
    const identity = tenantIdentityAnswer(i.question, i.config);
    if (identity) return identity;
    if (/\b(summarize|summary|recap|what did i|what did we|what you just)\b/i.test(i.question) && i.history?.length)
      return i.history.slice(-4).map((x) => x.content).join(' ');
    const rawTerms = (i.question.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []).filter((t) => !QUESTION_STOPWORDS.has(t));
    const expandedTerms = new Set([...rawTerms, ...tokens(i.question).filter((t) => t.length >= 3 && !QUESTION_STOPWORDS.has(t))]);
    const activeTerms = expandedTerms.size ? expandedTerms : new Set(i.question.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []);
    const contextSentences = i.context
      .flatMap((c) => c.text.split(/(?<=[.!?])\s+/))
      .map((s) => ({ s, hit: [...activeTerms].filter((t) => s.toLowerCase().includes(t)).length }))
      .filter((x) => x.hit > 0)
      .sort((a, b) => b.hit - a.hit)
      .slice(0, 3);
    if (contextSentences.length) return contextSentences.map((x) => x.s).join(' ');
    const historySentences = (i.history ?? [])
      .filter((x) => x.role === 'user' || !x.content.includes(i.config.fallbackMessage))
      .flatMap((x) => x.content.split(/(?<=[.!?])\s+/))
      .filter((s) => !s.trim().endsWith('?'))
      .map((s) => ({ s, hit: [...activeTerms].filter((t) => s.toLowerCase().includes(t)).length }))
      .filter((x) => x.hit > 0)
      .sort((a, b) => b.hit - a.hit)
      .slice(0, 3);
    if (historySentences.length) return historySentences.map((x) => x.s).join(' ');
    if (i.context.length > 0 && (i.context[0]?.score ?? 0) >= 0.25) return i.context[0]!.text;
    return i.config.fallbackMessage;
  }
  async streamAnswer(i: LlmInput, onToken: (token: string) => void) {
    const answer = await this.answer(i);
    for (const token of answer.match(/\S+\s*/g) ?? []) onToken(token);
    return answer;
  }
  async health(): Promise<ProviderHealth> { return { provider: this.name, type: 'local', connected: true, configured: true, reachable: true, authentication: 'not-required', status: 'ready' }; }
}

export class OpenAICompatibleLlm implements LlmProvider {
  readonly name: string;
  constructor(private config: LlmConfig, private fetchImpl: Fetch = fetch) { this.name = config.providerName; }
  private async request(url: string | URL, init: RequestInit, retries: number, debug?: ChatDebug, purpose = 'llm_request'): Promise<Response> {
    let last: unknown;
    const callId = crypto.randomUUID();
    for (let attempt = 0; attempt <= retries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.config.timeoutMs);
      const started = process.hrtime.bigint();
      debug?.('LLM_HTTP_ATTEMPT_STARTED', { callId, purpose, attemptNumber: attempt + 1, maxAttempts: retries + 1, method: init.method ?? 'GET', timeoutMs: this.config.timeoutMs });
      try {
        const response = await this.fetchImpl(url, { ...init, signal: controller.signal });
        const durationMs = Number(process.hrtime.bigint() - started) / 1_000_000;
        debug?.('LLM_HTTP_ATTEMPT_COMPLETED', { callId, purpose, attemptNumber: attempt + 1, maxAttempts: retries + 1, durationMs, httpStatus: response.status, responseOk: response.ok, retryableStatus: retryable(response.status) });
        if (!retryable(response.status) || attempt === retries) return response;
        last = new Error(`retryable status ${response.status}`);
        const delayMs = this.config.retryBaseMs * 2 ** attempt;
        debug?.('LLM_RETRY_SCHEDULED', { callId, purpose, attemptNumber: attempt + 1, nextAttemptNumber: attempt + 2, maxAttempts: retries + 1, delayMs, retryReason: `http_${response.status}` });
      } catch (error) {
        last = error;
        const durationMs = Number(process.hrtime.bigint() - started) / 1_000_000;
        const timeout = controller.signal.aborted;
        debug?.('LLM_HTTP_ATTEMPT_FAILED', { callId, purpose, attemptNumber: attempt + 1, maxAttempts: retries + 1, durationMs, errorCategory: timeout ? 'timeout' : error instanceof Error ? error.name : 'unknown', timeout });
        if (attempt === retries) throw new Error(`LLM request failed after ${attempt + 1} attempt(s)`, { cause: error });
        const delayMs = this.config.retryBaseMs * 2 ** attempt;
        debug?.('LLM_RETRY_SCHEDULED', { callId, purpose, attemptNumber: attempt + 1, nextAttemptNumber: attempt + 2, maxAttempts: retries + 1, delayMs, retryReason: timeout ? 'timeout' : 'network_error' });
      } finally { clearTimeout(timer); }
      await sleep(this.config.retryBaseMs * 2 ** attempt);
    }
    throw last;
  }
  private url(path: string) { const url = new URL(path.slice(1), `${this.config.baseUrl}/`); for (const [name, value] of Object.entries(this.config.queryParams)) url.searchParams.set(name, value); return url; }
  private headers() { const key = this.config.authScheme ? `${this.config.authScheme} ${this.config.apiKey!}` : this.config.apiKey!; return { ...this.config.headers, [this.config.authHeaderName]: key, 'content-type': 'application/json' }; }
  private payload(i: LlmInput, stream: boolean) {
    const context = i.context.map((x, n) => `[Source ${n + 1}: ${x.source} | chunk ${x.id}]\n${x.text}`).join('\n\n');
    const policy = `${tenantIdentityPolicy(i.config)}\n${i.prompt}\nStrict grounding policy: For tenant-specific factual questions, answer only from TENANT KNOWLEDGE below. TENANT KNOWLEDGE is authoritative and overrides pretrained/model knowledge. The following tenant knowledge is verified evidence for this tenant and must be treated as authoritative. Do not invent or infer missing prices, availability, policies, features, guarantees, fees, locations, dates, or operating details. Preserve qualifications such as “starting from”, gym/location dependence, eligibility, and terms. If evidence explicitly says UNKNOWN or the requested fact is absent, say it is not available; do not use the fallback merely because the question is phrased differently from the evidence in the current tenant knowledge and direct the customer to the official support route when appropriate. Distinguish verified facts from assumptions. Use bounded CONVERSATION history only to resolve follow-ups. For an explicit summary or recap request, summarize that bounded conversation history and do not require tenant knowledge. For greetings and general conversation, respond naturally without requiring tenant knowledge. If the question is tenant-specific and evidence is empty, use this fallback: ${i.config.fallbackMessage}`;
    const boundedHistory = (i.history ?? []).slice(-8);
    const messages = [
      { role: 'system', content: `SYSTEM POLICY\n${policy}` },
      { role: 'system', content: `TENANT KNOWLEDGE (authoritative, tenant-scoped)\n${context || '(none)'}` },
      ...boundedHistory.map((m) => ({ role: m.role, content: `CONVERSATION\n${m.content}` })),
      { role: 'user', content: `USER\nQuestion: ${i.question}` },
    ];
    const contextText = context || '(none)';
    if (i.debug && i.debugDetailed) {
      const serializedMessages = JSON.stringify(messages);
      const serializedPayload = JSON.stringify({ model: this.config.model, stream, messages });
      i.debug('LLM_INPUT_BOUNDARY', {
        provider: this.config.providerName, model: this.config.model, stream,
        messageCount: messages.length, payloadChars: messages.reduce((total, message) => total + message.content.length, 0),
        payloadTokensApprox: approxTokens(serializedMessages), serializedPayloadChars: serializedPayload.length,
        serializedPayloadTokensApprox: approxTokens(serializedPayload), historyIncluded: boundedHistory.length > 0,
        contextChars: contextText.length, contextTokensApprox: approxTokens(contextText), contextFingerprint: fingerprint(contextText),
        systemPolicyChars: messages[0]?.content.length ?? 0, systemPolicyFingerprint: fingerprint(messages[0]?.content ?? ''),
        messageMetrics: messages.map((m) => ({ role: m.role, chars: m.content.length, tokensApprox: approxTokens(m.content), fingerprint: fingerprint(m.content) })),
        temperature: this.config.sampling.temperature, topPConfigured: this.config.sampling.topP !== undefined,
        maxOutputTokens: this.config.sampling.maxTokens ?? null,
      });
    }
    return { model: this.config.model, temperature: this.config.sampling.temperature,
      ...(this.config.sampling.topP === undefined ? {} : { top_p: this.config.sampling.topP }),
      ...(this.config.sampling.maxTokens === undefined ? {} : { max_tokens: this.config.sampling.maxTokens }), stream, messages };
  }
  async answer(i: LlmInput) {
    const identity = tenantIdentityAnswer(i.question, i.config);
    if (identity) return identity;
    const r = await this.request(this.url(this.config.chatPath), { method: 'POST', headers: this.headers(), body: JSON.stringify(this.payload(i, false)) }, this.config.maxRetries, i.debug, 'answer_generation');
    if (!r.ok) {
      i.debug?.('LLM_RESPONSE_HTTP_ERROR', { purpose: 'answer_generation', httpStatus: r.status });
      throw new Error(`LLM provider failed (${r.status})`);
    }
    let j: { choices?: { message?: { content?: string } }[]; usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } };
    try { j = await r.json() as typeof j; }
    catch (error) { i.debug?.('LLM_RESPONSE_PARSE_FAILED', { purpose: 'answer_generation', parseErrorName: error instanceof Error ? error.name : 'unknown' }); throw error; }
    const providerAnswer = j.choices?.[0]?.message?.content;
    const answer = providerAnswer ?? i.config.fallbackMessage;
    i.debug?.('LLM_RESPONSE_PARSED', { purpose: 'answer_generation', stream: false, responseChars: answer.length, responseTokensApprox: approxTokens(answer), emptyContent: !providerAnswer?.trim(), fallbackSubstituted: providerAnswer === undefined, choiceCount: j.choices?.length ?? 0, usage: j.usage ? { promptTokens: j.usage.prompt_tokens ?? null, completionTokens: j.usage.completion_tokens ?? null, totalTokens: j.usage.total_tokens ?? null, source: 'provider_reported' } : { promptTokensApprox: null, completionTokensApprox: approxTokens(answer), totalTokensApprox: null, source: 'partial_estimate' } });
    return answer;
  }
  async streamAnswer(i: LlmInput, onToken: (token: string) => void) {
    const identity = tenantIdentityAnswer(i.question, i.config);
    if (identity) { for (const token of identity.match(/\S+\s*/g) ?? []) onToken(token); return identity; }
    const r = await this.request(this.url(this.config.chatPath), { method: 'POST', headers: this.headers(), body: JSON.stringify(this.payload(i, true)) }, this.config.maxRetries, i.debug, 'stream_answer');
    if (!r.ok) { i.debug?.('LLM_RESPONSE_HTTP_ERROR', { purpose: 'stream_answer', httpStatus: r.status }); throw new Error(`LLM provider failed (${r.status})`); }
    if (!r.body) { i.debug?.('LLM_STREAM_BODY_MISSING', { purpose: 'stream_answer', fallbackCall: 'non_streaming_answer' }); return this.answer(i); }
    const reader = r.body.getReader(); const decoder = new TextDecoder(); let buffer = ''; let answer = '';
    const streamStarted = process.hrtime.bigint(); let firstTokenMs: number | undefined; let malformedFrames = 0; let frameCount = 0; let responseBytes = 0;
    const emit = (line: string) => {
      if (!line.startsWith('data: ') || line === 'data: [DONE]') return;
      frameCount++;
      try { const token = (JSON.parse(line.slice(6)) as { choices?: { delta?: { content?: string } }[] }).choices?.[0]?.delta?.content ?? ''; if (token) { if (firstTokenMs === undefined) firstTokenMs = Number(process.hrtime.bigint() - streamStarted) / 1_000_000; answer += token; onToken(token); } } catch { malformedFrames++; /* Preserve existing behavior: ignore malformed provider frames. */ }
    };
    while (true) { const { value, done } = await reader.read(); responseBytes += value?.byteLength ?? 0; buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done }); const lines = buffer.split('\n'); buffer = lines.pop() ?? ''; lines.forEach(emit); if (done) break; }
    i.debug?.('LLM_STREAM_PARSED', { purpose: 'stream_answer', responseChars: answer.length, responseTokensApprox: approxTokens(answer), responseBytes, frameCount, malformedFrames, trailingBufferChars: buffer.length, incompleteFinalFrame: Boolean(buffer.trim()), firstTokenMs: firstTokenMs ?? null, streamDurationMs: Number(process.hrtime.bigint() - streamStarted) / 1_000_000, emptyContent: !answer.trim() });
    if (!answer) {
      // Some OpenAI-compatible gateways advertise streaming but emit no parseable
      // delta frames. Recover through the same provider's bounded non-streaming
      // request rather than incorrectly presenting the tenant fallback as a
      // successful answer.
      i.debug?.('LLM_STREAM_RECOVERY_STARTED', { purpose: 'stream_answer', reason: 'no_parseable_content_frames' });
      const recovered = await this.answer(i);
      if (recovered) onToken(recovered);
      return recovered;
    }
    return answer;
  }
  async reformulate(i: LlmInput) {
    const history = (i.history ?? []).slice(-8).map((x) => `${x.role}: ${x.content}`).join('\n');
    const payload = {
      model: this.config.model, temperature: 0, max_tokens: 80, stream: false,
      messages: [
        { role: 'system', content: 'Convert the user request into one concise search query for the current tenant knowledge base. Resolve pronouns using the conversation. Return up to three concise alternative search queries, one per line, with no numbering or explanation. Resolve pronouns using the conversation. If this is a greeting or general conversational request, return NONE.' },
        { role: 'user', content: `Conversation:\n${history || '(none)'}\nCurrent request: ${i.question}` },
      ],
    };
    const reformulationMessages = payload.messages;
    if (i.debug && i.debugDetailed) {
      const reformulationChars = reformulationMessages.reduce((total, message) => total + message.content.length, 0);
      i.debug('LLM_REFORMULATION_INPUT', { provider: this.config.providerName, model: this.config.model, historyIncluded: history.length > 0, messageCount: reformulationMessages.length, payloadChars: reformulationChars, payloadTokensApprox: approxTokens(JSON.stringify(reformulationMessages)), messages: reformulationMessages.map((m) => ({ role: m.role, chars: m.content.length, tokensApprox: approxTokens(m.content), fingerprint: fingerprint(m.content) })) });
    }
    const r = await this.request(this.url(this.config.chatPath), { method: 'POST', headers: this.headers(), body: JSON.stringify(payload) }, 0, i.debug, 'query_reformulation');
    if (!r.ok) { i.debug?.('LLM_REFORMULATION_HTTP_ERROR', { httpStatus: r.status }); return null; }
    let j: { choices?: { message?: { content?: string } }[]; usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } };
    try { j = await r.json() as typeof j; }
    catch (error) { i.debug?.('LLM_REFORMULATION_PARSE_FAILED', { parseErrorName: error instanceof Error ? error.name : 'unknown' }); throw error; }
    const query = j.choices?.[0]?.message?.content?.trim().replace(/^['\"]|['\"]$/g, '');
    i.debug?.('LLM_REFORMULATION_RESPONSE', { returned: Boolean(query && query.toUpperCase() !== 'NONE'), queryChars: query?.length ?? 0, alternativeQueryCount: query?.split(/\r?\n/).filter(Boolean).length ?? 0, emptyContent: !query, usage: j.usage ? { promptTokens: j.usage.prompt_tokens ?? null, completionTokens: j.usage.completion_tokens ?? null, totalTokens: j.usage.total_tokens ?? null, source: 'provider_reported' } : { source: 'unavailable' } });
    return query && query.toUpperCase() !== 'NONE' ? query : null;
  }
  async health(): Promise<ProviderHealth> {
    const common = { provider: this.config.providerName, type: this.config.providerType, model: this.config.model, baseUrl: this.config.baseUrl, configured: true };
    try { const r = await this.request(this.url(this.config.healthPath), { method: 'GET', headers: this.headers() }, 0); const authentication = [401, 403].includes(r.status) ? 'rejected' : r.ok ? 'accepted' : 'unknown'; return { ...common, connected: r.ok, reachable: true, authentication, status: r.ok ? 'ready' : authentication === 'rejected' ? 'authentication-failed' : 'upstream-error', detail: `diagnostic HTTP ${r.status}` }; }
    catch (error) { const timeout = error instanceof Error && (error.name === 'AbortError' || String(error.cause).includes('AbortError')); return { ...common, connected: false, reachable: false, authentication: 'unknown', status: timeout ? 'timeout' : 'unreachable', detail: timeout ? `diagnostic timed out after ${this.config.timeoutMs}ms` : 'diagnostic request failed' }; }
  }
}
export function createLlmProvider(config: LlmConfig, fetchImpl: Fetch = fetch): LlmProvider { return config.providerType === 'local' ? new LocalLlm(config.providerName) : new OpenAICompatibleLlm(config, fetchImpl); }
