import { describe, expect, it } from 'vitest';
import { createRequestContext, PipelineObserver, safeRef } from '../src/lib/observability.js';

type Captured = Record<string, unknown>;

function capture(mode: 'off' | 'basic' | 'detailed' = 'detailed') {
  const records: Captured[] = [];
  const logger = { debug: (record: Record<string, unknown>) => { records.push(record); }, info: (record: Record<string, unknown>) => { records.push(record); }, warn: (record: Record<string, unknown>) => { records.push(record); }, error: (record: Record<string, unknown>) => { records.push(record); } };
  return { records, observer: new PipelineObserver(logger, mode) };
}

describe('pipeline observability', () => {
  it('emits the normalized schema with a hashed tenant reference and no raw customer content', () => {
    const { records, observer } = capture();
    const context = observer.createContext({ requestId: 'request-123', traceId: 'trace-456', entryPoint: 'live-widget-chat', tenantId: 'tenant-private-id' });
    observer.emit(context, 'pipeline.input.accepted', {
      stage: 'input', operation: 'chat_message', status: 'success',
      inputMetrics: { question: 'my secret question', questionChars: 17, email: 'person@example.com', safeCount: 3 },
      outputMetrics: { messageChars: 17 },
    });
    expect(records).toHaveLength(1);
    const event = records[0]!;
    expect(event).toMatchObject({ timestamp: expect.any(String), level: 'info', event: 'pipeline.input.accepted', requestId: 'request-123', traceId: 'trace-456', entryPoint: 'live-widget-chat', stage: 'input', operation: 'chat_message', status: 'success' });
    expect(event.tenantRef).toBe(safeRef('tenant-private-id'));
    expect(JSON.stringify(event)).not.toContain('my secret question');
    expect(JSON.stringify(event)).not.toContain('person@example.com');
    expect((event.inputMetrics as Record<string, unknown>).questionChars).toBe(17);
    expect((event.inputMetrics as Record<string, unknown>).safeCount).toBe(3);
  });


  it('preserves safe correlation headers and refuses header values that resemble secrets or personal contact details', () => {
    const { observer } = capture();
    const valid = createRequestContext(observer, { id: 'fastify-req-123', method: 'POST', url: '/v1/chat?x=1', headers: { 'x-request-id': 'external-req-123', 'x-trace-id': 'trace-123' } });
    expect(valid).toMatchObject({ requestId: 'external-req-123', traceId: 'trace-123', entryPoint: 'live-widget-chat' });
    const unsafe = createRequestContext(observer, { id: 'fastify-req-456', method: 'POST', url: '/v1/chat', headers: { 'x-request-id': 'sk-1234567890abcdef', 'x-trace-id': 'person@example.com' } });
    expect(unsafe.requestId).toBe('fastify-req-456');
    expect(unsafe.traceId).toBe(unsafe.requestId);
  });

  it('uses one span ID for each measured stage and links nested stages to their parent', async () => {
    const { records, observer } = capture();
    const context = observer.createContext({ requestId: 'r-1', entryPoint: 'test' });
    await observer.measure(context, 'outer', 'parent', async () => {
      await observer.measure(context, 'inner', 'child', async () => 42);
    });
    const childStart = records.find((x) => x.event === 'pipeline.stage.started' && x.stage === 'inner')!;
    const parentStart = records.find((x) => x.event === 'pipeline.stage.started' && x.stage === 'outer')!;
    const childDone = records.find((x) => x.event === 'pipeline.stage.completed' && x.stage === 'inner')!;
    expect(childStart.parentSpanId).toBe(parentStart.spanId);
    expect(childDone.spanId).toBe(childStart.spanId);
    expect(childDone.durationMs).toEqual(expect.any(Number));
    expect(childDone.durationMs as number).toBeGreaterThanOrEqual(0);
  });

  it('does not emit detailed records in basic mode or any records in off mode', () => {
    const basic = capture('basic');
    const context = basic.observer.createContext({ requestId: 'r-basic', entryPoint: 'test' });
    basic.observer.emit(context, 'pipeline.internal.diagnostic', { stage: 'retrieval.internal', operation: 'rank_candidates', status: 'success', detailed: true });
    basic.observer.emit(context, 'pipeline.request.started', { stage: 'request', operation: 'http_request', status: 'started' });
    expect(basic.records).toHaveLength(1);
    const off = capture('off');
    off.observer.emit(off.observer.createContext({ entryPoint: 'test' }), 'pipeline.request.started', { stage: 'request', operation: 'http_request', status: 'started' });
    expect(off.records).toHaveLength(0);
  });

  it('isolates logging failures from the application operation', async () => {
    const observer = new PipelineObserver({ info: () => { throw new Error('logger failed'); }, error: () => { throw new Error('logger failed'); } }, 'detailed');
    const context = observer.createContext({ requestId: 'r-safe', entryPoint: 'test' });
    await expect(observer.measure(context, 'stage', 'work', async () => 'ok')).resolves.toBe('ok');
    await expect(observer.measure(context, 'stage', 'failure', async () => { throw new Error('application failure'); })).rejects.toThrow('application failure');
  });

  it('exposes bounded latency percentiles and operational counters', async () => {
    const { observer } = capture();
    const context = observer.createContext({ requestId: 'r-metrics', entryPoint: 'live-widget-chat' });
    observer.emit(context, 'pipeline.request.started', { stage: 'request', operation: 'http_request', status: 'started' });
    observer.emit(context, 'pipeline.request.completed', { stage: 'request', operation: 'http_response', status: 'success', durationMs: 25 });
    observer.emit(context, 'pipeline.retrieval.completed', { stage: 'retrieval', operation: 'retrieve_evidence', status: 'success', outputMetrics: { evidenceCount: 0 } });
    const metrics = observer.snapshot();
    expect(metrics.requestsStarted).toBe(1);
    expect(metrics.requestsCompleted).toBe(1);
    expect(metrics.retrievalEmpty).toBe(1);
    expect(metrics.requestLatencyMs.p50).toBe(25);
    expect(observer.prometheus()).toContain('pipeline_request_latency_p95_ms');
    expect(observer.prometheus()).toContain('pipeline_stage_latency_p95_ms{stage=\"request\"}');
  });

  it('preserves exact full-text fields only when explicitly enabled and reports truncation and redaction', () => {
    const records: Array<Record<string, unknown>> = [];
    const sink = { debug: (record: Record<string, unknown>) => records.push(record), info: (record: Record<string, unknown>) => records.push(record), warn: (record: Record<string, unknown>) => records.push(record), error: (record: Record<string, unknown>) => records.push(record) };
    const observer = new PipelineObserver(sink, 'detailed', { enabled: true, level: 'debug', fullText: true, maxTextChars: 128 });
    const exact = '  first line\nsecond line  ';
    const captured = observer.captureText(exact);
    expect(captured.value).toBe(exact);
    expect(captured.originalCharacterCount).toBe(exact.length);
    expect(captured.loggedCharacterCount).toBe(exact.length);
    expect(captured.truncated).toBe(false);
    expect(captured.complete).toBe(true);
    expect(captured.sha256).toMatch(/^[a-f0-9]{64}$/);

    const secretCapture = observer.captureText('api_key=do-not-log-this-value');
    expect(secretCapture.value).toContain('[REDACTED_SECRET]');
    expect(secretCapture.redactionsApplied).toBeGreaterThan(0);
    expect(secretCapture.complete).toBe(false);
    const serializedSecret = observer.captureText('{\"authorization\":\"Bearer do-not-log-this-value\",\"api_key\":\"also-do-not-log\"}');
    expect(serializedSecret.value).not.toContain('do-not-log-this-value');
    expect(serializedSecret.value).not.toContain('also-do-not-log');
    expect(serializedSecret.complete).toBe(false);
    expect(secretCapture.sha256).toMatch(/^[a-f0-9]{64}$/);

    const longCapture = observer.captureText('x'.repeat(300));
    expect(longCapture.originalCharacterCount).toBe(300);
    expect(longCapture.loggedCharacterCount).toBe(128);
    expect(longCapture.truncated).toBe(true);
    expect(longCapture.complete).toBe(false);

    const context = observer.createContext({ requestId: 'forensic-request', traceId: 'forensic-trace', entryPoint: 'grounding-test', tenantId: 'tenant-1' });
    observer.emitForensic(context, 'pipeline.grounding.validator.input', { stage: 'grounding.validator', operation: 'validate', status: 'started', input: { question: observer.captureText(exact) } });
    expect(records[0]).toMatchObject({ event: 'pipeline.grounding.validator.input', traceId: 'forensic-trace', input: { question: { value: exact, complete: true } } });
  });

  it('omits actual text when full-text diagnostics are disabled and filters forensic severity correctly', () => {
    const { records, observer } = capture('detailed');
    const context = observer.createContext({ requestId: 'r-forensic', entryPoint: 'test' });
    const captured = observer.captureText('private text');
    expect(captured.value).toBeNull();
    expect(captured.omissionReason).toBe('full_text_logging_disabled');
    observer.emitForensic(context, 'pipeline.forensic.success', { stage: 'test', operation: 'example', status: 'success', input: { content: captured } });
    expect(JSON.stringify(records)).not.toContain('private text');

    const onlyErrors: Array<Record<string, unknown>> = [];
    const errorObserver = new PipelineObserver({ debug: (record) => onlyErrors.push(record), info: (record) => onlyErrors.push(record), warn: (record) => onlyErrors.push(record), error: (record) => onlyErrors.push(record) }, 'detailed', { enabled: true, level: 'error', fullText: false, maxTextChars: 128 });
    const errorContext = errorObserver.createContext({ entryPoint: 'test' });
    errorObserver.emitForensic(errorContext, 'pipeline.forensic.success', { stage: 'test', operation: 'successful', status: 'success' });
    errorObserver.emitForensic(errorContext, 'pipeline.forensic.failed', { stage: 'test', operation: 'failed', status: 'failed' });
    expect(onlyErrors).toHaveLength(1);
    expect(onlyErrors[0]?.level).toBe('error');
  });


  it('keeps concurrent trace contexts and their nested spans isolated', async () => {
    const { records, observer } = capture('detailed');
    const contextA = observer.createContext({ requestId: 'request-a', traceId: 'trace-a', entryPoint: 'live-widget-chat', tenantId: 'tenant-a' });
    const contextB = observer.createContext({ requestId: 'request-b', traceId: 'trace-b', entryPoint: 'dashboard-prompt-preview', tenantId: 'tenant-b' });
    await Promise.all([
      observer.measure(contextA, 'stage.concurrent', 'work_a', async () => {
        await new Promise((resolve) => setTimeout(resolve, 8));
        observer.emitForensic(contextA, 'pipeline.concurrent.marker', { stage: 'test', operation: 'a', status: 'success' });
      }),
      observer.measure(contextB, 'stage.concurrent', 'work_b', async () => {
        await new Promise((resolve) => setTimeout(resolve, 2));
        observer.emitForensic(contextB, 'pipeline.concurrent.marker', { stage: 'test', operation: 'b', status: 'success' });
      }),
    ]);
    const markers = records.filter((record) => record.event === 'pipeline.concurrent.marker');
    expect(markers).toHaveLength(2);
    expect(markers.find((record) => record.operation === 'a')).toMatchObject({ requestId: 'request-a', traceId: 'trace-a', tenantRef: safeRef('tenant-a') });
    expect(markers.find((record) => record.operation === 'b')).toMatchObject({ requestId: 'request-b', traceId: 'trace-b', tenantRef: safeRef('tenant-b') });
  });

});
