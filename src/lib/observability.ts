import crypto from 'node:crypto';
import { AsyncLocalStorage } from 'node:async_hooks';

export type DiagnosticMode = 'off' | 'basic' | 'detailed';
export type PipelineStatus =
  | 'started'
  | 'success'
  | 'failed'
  | 'skipped'
  | 'retrying'
  | 'fallback'
  | 'rejected'
  | 'cancelled';

export type ProviderDiagnostic = (operation: string, data: Record<string, unknown>) => void;

export type PipelineContext = {
  requestId: string;
  traceId: string;
  entryPoint: string;
  tenantRef?: string;
};

export type PipelineEvent = {
  stage: string;
  operation: string;
  status: PipelineStatus;
  durationMs?: number;
  reason?: string | null;
  errorCode?: string | null;
  inputMetrics?: Record<string, unknown>;
  outputMetrics?: Record<string, unknown>;
  parentSpanId?: string;
  spanId?: string;
  level?: 'info' | 'warn' | 'error';
  detailed?: boolean;
};

type LogSink = {
  info?: (obj: Record<string, unknown>, message?: string) => unknown;
  warn?: (obj: Record<string, unknown>, message?: string) => unknown;
  error?: (obj: Record<string, unknown>, message?: string) => unknown;
};

const idPattern = /^[A-Za-z0-9._:-]{1,128}$/;
const privateField = /^(?:(?:x[-_])?(?:authorization|proxy-authorization|api[-_ ]?key|admin[-_ ]?api[-_ ]?key|password|secret|access[-_ ]?token|refresh[-_ ]?token|headers|cookie|set-cookie|credential|credentials))$/i;
const contentField = /^(?:prompt|content|answer|question|query|text|history|body|stack|raw|rawResponse|responseBody|fallbackMessage|preview|messages|chunkText|documentText)(?:Preview|Text|Content|Body|Raw)?$/i;
const emailPattern = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi;
const phonePattern = /(?:\+?\d[\d ()-]{7,}\d)/g;
const secretPattern = /\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b|\b(?:Bearer\s+)[A-Za-z0-9._~-]{8,}/gi;

export function safeRef(value: string): string {
  return crypto.createHash('sha256').update(value).digest('hex').slice(0, 12);
}

function acceptedId(value: unknown): string | undefined {
  if (typeof value !== 'string' || !idPattern.test(value)) return undefined;
  if (/^(?:sk|pk)-[A-Za-z0-9_-]{12,}$/i.test(value) || /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i.test(value) || /^\+?\d[\d ()-]{7,}\d$/.test(value)) return undefined;
  return value;
}

function redactString(value: string): string {
  return value
    .replace(secretPattern, '[REDACTED_SECRET]')
    .replace(emailPattern, '[REDACTED_EMAIL]')
    .replace(phonePattern, '[REDACTED_PHONE]')
    .slice(0, 300);
}

function sanitize(value: unknown, key = '', depth = 0): unknown {
  if (privateField.test(key)) return '[REDACTED]';
  if (contentField.test(key)) return '[OMITTED]';
  if (depth >= 4) return '[DEPTH_LIMIT]';
  if (typeof value === 'string') return redactString(value);
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'boolean' || value === null) return value;
  if (Array.isArray(value)) return value.slice(0, 25).map((entry) => sanitize(entry, '', depth + 1));
  if (typeof value === 'object' && value !== null) {
    const out: Record<string, unknown> = {};
    for (const [childKey, childValue] of Object.entries(value).slice(0, 60)) {
      if (privateField.test(childKey) || contentField.test(childKey)) continue;
      out[childKey] = sanitize(childValue, childKey, depth + 1);
    }
    return out;
  }
  return value === undefined ? undefined : String(value);
}

function errorCode(error: unknown): string {
  if (error instanceof Error && error.name === 'ZodError') return 'VALIDATION_ERROR';
  if (error instanceof Error && (error.name === 'AbortError' || (error.cause instanceof Error && error.cause.name === 'AbortError'))) return 'TIMEOUT';
  if (typeof error === 'object' && error !== null && 'code' in error) {
    const code = (error as { code?: unknown }).code;
    if (typeof code === 'string' && /^[A-Z0-9_-]{1,64}$/i.test(code) && !/^(?:sk|pk)-[A-Za-z0-9_-]{12,}$/i.test(code)) return code;
  }
  if (error instanceof Error && /timed out|timeout/i.test(`${error.name} ${error.message}`)) return 'TIMEOUT';
  if (error instanceof Error && /(?:\(|HTTP\s*)(4\d\d|5\d\d)\)?/i.test(error.message)) {
    const status = /(?:\(|HTTP\s*)(4\d\d|5\d\d)/i.exec(error.message)?.[1];
    if (status) return `HTTP_${status}`;
  }
  if (error instanceof Error && /fetch failed|network|socket|ECONN|ENOTFOUND/i.test(`${error.name} ${error.message} ${error.cause instanceof Error ? `${error.cause.name} ${error.cause.message}` : ''}`)) return 'NETWORK_ERROR';
  if (error instanceof SyntaxError || (error instanceof Error && /json|parse/i.test(error.name))) return 'RESPONSE_PARSE_ERROR';
  return 'UNCLASSIFIED_ERROR';
}

export class PipelineObserver {
  private readonly activeSpan = new AsyncLocalStorage<string>();
  private readonly counters = { requestsStarted: 0, requestsCompleted: 0, answerRequestsStarted: 0, llmCalls: 0, llmFailures: 0, vectorSearchFailures: 0, retrievalEmpty: 0, evidenceSupported: 0, evidencePartiallySupported: 0, evidenceUncertain: 0, evidenceNotSupported: 0, groundingRejections: 0, corrections: 0, fallbacks: 0, streamDoneEventsSent: 0, streamCancellations: 0 };
  private readonly requestLatencies: number[] = [];
  private readonly stageLatencies = new Map<string, number[]>();

  constructor(
    private readonly sink: LogSink,
    readonly mode: DiagnosticMode = 'basic',
  ) {}

  createContext(input: {
    requestId?: unknown;
    traceId?: unknown;
    entryPoint: string;
    tenantId?: string;
  }): PipelineContext {
    const requestId = acceptedId(input.requestId) ?? crypto.randomUUID();
    return {
      requestId,
      traceId: acceptedId(input.traceId) ?? requestId,
      entryPoint: input.entryPoint.slice(0, 64),
      ...(input.tenantId ? { tenantRef: safeRef(input.tenantId) } : {}),
    };
  }

  setTenant(context: PipelineContext, tenantId: string) {
    context.tenantRef = safeRef(tenantId);
  }

  emit(context: PipelineContext, event: string, data: PipelineEvent) {
    if (this.mode === 'off' || (data.detailed && this.mode !== 'detailed')) return;
    const level = data.level ?? (data.status === 'failed' || data.status === 'rejected' ? 'error' : data.status === 'retrying' || data.status === 'fallback' ? 'warn' : 'info');
    const record: Record<string, unknown> = {
      timestamp: new Date().toISOString(),
      level,
      event: event.slice(0, 100),
      requestId: context.requestId,
      traceId: context.traceId,
      ...((data.parentSpanId ?? this.activeSpan.getStore()) ? { parentSpanId: data.parentSpanId ?? this.activeSpan.getStore() } : {}),
      spanId: data.spanId ?? crypto.randomUUID(),
      entryPoint: context.entryPoint,
      ...(context.tenantRef ? { tenantRef: context.tenantRef } : {}),
      stage: data.stage.slice(0, 100),
      operation: data.operation.slice(0, 100),
      status: data.status,
      ...(data.durationMs === undefined ? {} : { durationMs: Math.max(0, Math.round(data.durationMs * 100) / 100) }),
      ...(data.inputMetrics ? { inputMetrics: sanitize(data.inputMetrics) } : {}),
      ...(data.outputMetrics ? { outputMetrics: sanitize(data.outputMetrics) } : {}),
      ...(data.reason === undefined ? {} : { reason: data.reason === null ? null : redactString(data.reason) }),
      ...(data.errorCode === undefined ? {} : { errorCode: data.errorCode }),
    };
    try {
      const bounded = sanitize(record) as Record<string, unknown>;
      this.observe(bounded);
      const logger = this.sink[level] ?? this.sink.info;
      if (logger) logger.call(this.sink, bounded, event);
    } catch {
      // Diagnostic I/O must never change application behavior.
    }
  }

  private observe(record: Record<string, unknown>) {
    const event = String(record.event ?? '');
    const stage = String(record.stage ?? '');
    const operation = String(record.operation ?? '');
    const status = String(record.status ?? '');
    const entryPoint = String(record.entryPoint ?? '');
    const durationMs = typeof record.durationMs === 'number' ? record.durationMs : undefined;
    const outputMetrics = record.outputMetrics as Record<string, unknown> | undefined;
    const answerEntryPoint = ['live-widget-chat', 'live-widget-stream', 'dashboard-prompt-preview'].includes(entryPoint);
    if (event === 'pipeline.request.started') {
      this.counters.requestsStarted++;
      if (answerEntryPoint) this.counters.answerRequestsStarted++;
    }
    if (event === 'pipeline.request.completed') {
      this.counters.requestsCompleted++;
      if (durationMs !== undefined) this.pushBounded(this.requestLatencies, durationMs);
    }
    // Count external LLM request IDs once at attempt one, not service methods that may answer locally.
    if (event === 'pipeline.provider.diagnostic' && stage === 'llm.provider' && operation === 'llm_http_attempt_started' && outputMetrics?.attemptNumber === 1) this.counters.llmCalls++;
    if (event === 'pipeline.provider.diagnostic' && stage === 'llm.provider') {
      const finalAttempt = outputMetrics?.attemptNumber === outputMetrics?.maxAttempts;
      if (['llm_response_http_error', 'llm_response_parse_failed', 'llm_reformulation_http_error', 'llm_reformulation_parse_failed'].includes(operation) || (operation === 'llm_http_attempt_failed' && finalAttempt)) this.counters.llmFailures++;
    }
    if (event === 'pipeline.stage.failed' && stage === 'retrieval.dense_search') this.counters.vectorSearchFailures++;
    if (event === 'pipeline.retrieval.completed') {
      if (outputMetrics?.evidenceCount === 0) this.counters.retrievalEmpty++;
      switch (outputMetrics?.evidenceStatus) {
        case 'SUPPORTED': this.counters.evidenceSupported++; break;
        case 'PARTIALLY_SUPPORTED': this.counters.evidencePartiallySupported++; break;
        case 'UNCERTAIN': this.counters.evidenceUncertain++; break;
        case 'NOT_SUPPORTED': this.counters.evidenceNotSupported++; break;
        default: break;
      }
    }
    if (event === 'pipeline.grounding.validation.completed' && status === 'rejected') this.counters.groundingRejections++;
    if (event === 'pipeline.grounding.correction.completed') this.counters.corrections++;
    if (event === 'pipeline.fallback.selected') this.counters.fallbacks++;
    if (event === 'pipeline.stream.done_event_sent') this.counters.streamDoneEventsSent++;
    if (event === 'pipeline.stream.disconnected' || (event === 'pipeline.request.completed' && status === 'cancelled')) this.counters.streamCancellations++;
    if (event === 'pipeline.stage.completed' && durationMs !== undefined) this.pushStageBounded(stage, durationMs);
  }

  private pushStageBounded(stage: string, value: number) {
    const normalized = stage.replace(/[^a-zA-Z0-9_.-]/g, '_').slice(0, 80) || 'unknown';
    const key = this.stageLatencies.has(normalized) || this.stageLatencies.size < 64 ? normalized : 'other';
    let values = this.stageLatencies.get(key);
    if (!values) { values = []; this.stageLatencies.set(key, values); }
    this.pushBounded(values, value);
  }

  private pushBounded(values: number[], value: number) {
    values.push(value);
    if (values.length > 2048) values.shift();
  }

  private percentile(values: number[], p: number) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return Math.round(sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)]! * 100) / 100;
  }

  snapshot() {
    return {
      mode: this.mode,
      ...this.counters,
      averageLlmCallsPerAnswerRequest: this.counters.answerRequestsStarted ? Math.round((this.counters.llmCalls / this.counters.answerRequestsStarted) * 1000) / 1000 : 0,
      requestLatencyMs: { sampleCount: this.requestLatencies.length, p50: this.percentile(this.requestLatencies, 0.5), p95: this.percentile(this.requestLatencies, 0.95), p99: this.percentile(this.requestLatencies, 0.99) },
      stageLatencyMs: Object.fromEntries([...this.stageLatencies.entries()].map(([stage, values]) => [stage, { sampleCount: values.length, p50: this.percentile(values, 0.5), p95: this.percentile(values, 0.95), p99: this.percentile(values, 0.99) }])),
      sampleCapacityPerSeries: 2048, maxStageSeries: 64,
    };
  }

  prometheus() {
    const s = this.snapshot();
    return [
      '# TYPE pipeline_requests_started_total counter', `pipeline_requests_started_total ${s.requestsStarted}`,
      '# TYPE pipeline_requests_completed_total counter', `pipeline_requests_completed_total ${s.requestsCompleted}`,
      '# TYPE pipeline_llm_calls_total counter', `pipeline_llm_calls_total ${s.llmCalls}`,
      '# TYPE pipeline_llm_failures_total counter', `pipeline_llm_failures_total ${s.llmFailures}`,
      '# TYPE pipeline_vector_search_failures_total counter', `pipeline_vector_search_failures_total ${s.vectorSearchFailures}`,
      '# TYPE pipeline_retrieval_empty_total counter', `pipeline_retrieval_empty_total ${s.retrievalEmpty}`,
      '# TYPE pipeline_evidence_supported_total counter', `pipeline_evidence_supported_total ${s.evidenceSupported}`,
      '# TYPE pipeline_evidence_partially_supported_total counter', `pipeline_evidence_partially_supported_total ${s.evidencePartiallySupported}`,
      '# TYPE pipeline_evidence_uncertain_total counter', `pipeline_evidence_uncertain_total ${s.evidenceUncertain}`,
      '# TYPE pipeline_evidence_not_supported_total counter', `pipeline_evidence_not_supported_total ${s.evidenceNotSupported}`,
      '# TYPE pipeline_grounding_rejections_total counter', `pipeline_grounding_rejections_total ${s.groundingRejections}`,
      '# TYPE pipeline_corrections_total counter', `pipeline_corrections_total ${s.corrections}`,
      '# TYPE pipeline_fallbacks_total counter', `pipeline_fallbacks_total ${s.fallbacks}`,
      '# TYPE pipeline_stream_done_events_sent_total counter', `pipeline_stream_done_events_sent_total ${s.streamDoneEventsSent}`,
      '# TYPE pipeline_stream_cancellations_total counter', `pipeline_stream_cancellations_total ${s.streamCancellations}`,
      '# TYPE pipeline_request_latency_p50_ms gauge', `pipeline_request_latency_p50_ms ${s.requestLatencyMs.p50 ?? 0}`,
      '# TYPE pipeline_request_latency_p95_ms gauge', `pipeline_request_latency_p95_ms ${s.requestLatencyMs.p95 ?? 0}`,
      '# TYPE pipeline_request_latency_p99_ms gauge', `pipeline_request_latency_p99_ms ${s.requestLatencyMs.p99 ?? 0}`,
      ...Object.entries(s.stageLatencyMs as Record<string, { p50: number | null; p95: number | null; p99: number | null }>).flatMap(([stage, latency]) => {
        const label = stage.replace(/[^a-zA-Z0-9_.-]/g, '_');
        return [`pipeline_stage_latency_p50_ms{stage=\"${label}\"} ${latency.p50 ?? 0}`, `pipeline_stage_latency_p95_ms{stage=\"${label}\"} ${latency.p95 ?? 0}`, `pipeline_stage_latency_p99_ms{stage=\"${label}\"} ${latency.p99 ?? 0}`];
      }),
    ].join('\n') + '\n';
  }

  async measure<T>(
    context: PipelineContext,
    stage: string,
    operation: string,
    work: () => Promise<T>,
    metrics: { inputMetrics?: Record<string, unknown>; parentSpanId?: string; detailed?: boolean } = {},
  ): Promise<T> {
    const spanId = crypto.randomUUID();
    const start = process.hrtime.bigint();
    const parentSpanId = metrics.parentSpanId ?? this.activeSpan.getStore();
    this.emit(context, 'pipeline.stage.started', {
      stage, operation, status: 'started', inputMetrics: metrics.inputMetrics,
      parentSpanId, spanId, detailed: metrics.detailed,
    });
    try {
      const result = await this.activeSpan.run(spanId, work);
      const durationMs = Number(process.hrtime.bigint() - start) / 1_000_000;
      this.emit(context, 'pipeline.stage.completed', {
        stage, operation, status: 'success', durationMs, parentSpanId,
        spanId, detailed: metrics.detailed,
      });
      return result;
    } catch (error) {
      const durationMs = Number(process.hrtime.bigint() - start) / 1_000_000;
      const rejected = error instanceof Error && error.name === 'ZodError';
      this.emit(context, rejected ? 'pipeline.stage.rejected' : 'pipeline.stage.failed', {
        stage, operation, status: rejected ? 'rejected' : 'failed', durationMs, parentSpanId,
        spanId, errorCode: errorCode(error), detailed: metrics.detailed,
      });
      throw error;
    }
  }
}

export function entryPointFor(method: string, url: string): string {
  const pathname = url.split('?')[0] ?? url;
  if (pathname === '/v1/chat/stream') return 'live-widget-stream';
  if (pathname === '/v1/chat') return 'live-widget-chat';
  if (/^\/admin\/clients\/[^/]+\/prompts\/preview$/.test(pathname)) return 'dashboard-prompt-preview';
  if (pathname.startsWith('/admin/')) return 'dashboard-admin-api';
  return `${method.toLowerCase()}-api`;
}

export function createRequestContext(
  observer: PipelineObserver,
  request: { id?: string; method: string; url: string; headers: Record<string, unknown> },
): PipelineContext {
  const requestHeader = acceptedId(request.headers['x-request-id']);
  const traceHeader = acceptedId(request.headers['x-trace-id']);
  return observer.createContext({
    requestId: requestHeader ?? request.id,
    traceId: traceHeader,
    entryPoint: entryPointFor(request.method, request.url),
  });
}

export function safeErrorCode(error: unknown): string {
  return errorCode(error);
}
