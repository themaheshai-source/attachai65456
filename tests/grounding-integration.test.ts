/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, expect, it, vi } from 'vitest';
import { ChatService } from '../src/services/chat.js';
import { PipelineObserver } from '../src/lib/observability.js';

const evidence = { id: 'care', text: 'Call 020 3011 1002 or use the Help Centre.', source: 'knowledge.md', score: 0.9 };
const client: any = { id: 'tenant-a', enabled: true, name: 'Tenant A', slug: 'tenant-a', prompt: 'Be concise.', createdAt: '', updatedAt: '', config: { assistantName: 'A', companyName: 'Tenant A', fallbackMessage: 'UNKNOWN', topK: 4 } };
function fixture(first: string, second = first, pipeline?: PipelineObserver) {
  let calls = 0;
  const repo: any = { getClient: vi.fn(async () => client), createConversation: vi.fn(async () => 'conversation-1'), messages: vi.fn(async () => []), addMessage: vi.fn(async () => undefined), usage: vi.fn(async () => undefined) };
  const vector: any = { search: vi.fn(async () => [evidence]) };
  const llm: any = { answer: vi.fn(async () => ++calls === 1 ? first : second), streamAnswer: vi.fn(async (_input: any, onToken: (x: string) => void) => { onToken(first); return first; }) };
  return { service: new ChatService(repo, { embed: vi.fn(async () => [[1]]) } as any, vector, llm, pipeline) };
}
describe('runtime grounding integration', () => {
  it('returns a supported first answer without correction', async () => {
    const { service } = fixture(evidence.text);
    const result = await service.chat('tenant-a', 'How can I contact Customer Care?');
    expect(result.answer).toBe(evidence.text);
  });
  it('corrects one unsupported first answer and returns the grounded correction', async () => {
    const { service } = fixture('Call £25.', evidence.text);
    const result = await service.chat('tenant-a', 'How can I contact Customer Care?');
    expect(result.answer).toBe(evidence.text);
  });
  it('uses the safe fallback when the bounded correction also fails', async () => {
    const { service } = fixture('Call £25.');
    const result = await service.chat('tenant-a', 'How can I contact Customer Care?');
    expect(result.answer).toBe('UNKNOWN');
  });
  it('does not emit invalid stream tokens before correction', async () => {
    const { service } = fixture('Call £25.', evidence.text);
    const output: string[] = [];
    const result = await service.stream('tenant-a', 'How can I contact Customer Care?', undefined, (token) => output.push(token));
    expect(output.join('')).toBe('UNKNOWN');
    expect(result.answer).toBe('UNKNOWN');
  });
  it('applies grounding to preview', async () => {
    const { service } = fixture('Call £25.', evidence.text);
    const result = await service.preview('tenant-a', 'How can I contact Customer Care?', 'Be concise.');
    expect(result.answer).toBe(evidence.text);
  });

  it('logs the actual validator-entry question, answer, and evidence under the request trace', async () => {
    const records: Array<Record<string, unknown>> = [];
    const sink = {
      debug: (record: Record<string, unknown>) => { records.push(record); },
      info: (record: Record<string, unknown>) => { records.push(record); },
      warn: (record: Record<string, unknown>) => { records.push(record); },
      error: (record: Record<string, unknown>) => { records.push(record); },
    };
    const pipeline = new PipelineObserver(sink, 'detailed', { enabled: true, level: 'debug', fullText: true, maxTextChars: 4096 });
    const repo: any = { getClient: vi.fn(async () => client), createConversation: vi.fn(async () => 'conversation-1'), messages: vi.fn(async () => []), addMessage: vi.fn(async () => undefined), usage: vi.fn(async () => undefined) };
    const vector: any = { search: vi.fn(async () => [evidence]) };
    const answer = evidence.text;
    const llm: any = { name: 'mock-llm', answer: vi.fn(async () => answer), streamAnswer: vi.fn(async (_input: any, onToken: (x: string) => void) => { onToken(answer); return answer; }) };
    const service = new ChatService(repo, { name: 'mock-embedding', embed: vi.fn(async () => [[1]]) } as any, vector, llm, pipeline);
    const question = 'How can I contact Customer Care?';
    const trace = pipeline.createContext({ requestId: 'validator-request-1', traceId: 'validator-trace-1', entryPoint: 'live-widget-chat', tenantId: 'tenant-a' });

    const result = await service.chat('tenant-a', question, undefined, trace);
    const entry = records.find((record) => record.event === 'pipeline.grounding.validator.input');
    expect(entry).toBeDefined();
    expect(entry).toMatchObject({ requestId: 'validator-request-1', traceId: 'validator-trace-1', stage: 'grounding.validator', status: 'started' });
    const input = entry!.input as { question: { value: string; complete: boolean }; answer: { value: string; complete: boolean }; evidence: Array<{ id: { value: string }; text: { value: string; complete: boolean } }> };
    expect(input.question).toMatchObject({ value: question, complete: true });
    expect(input.answer).toMatchObject({ value: answer, complete: true });
    expect(input.evidence[0]?.text).toMatchObject({ value: evidence.text, complete: true });
    expect(input.evidence[0]?.id.value).toBe(evidence.id);
    const ruleEvents = records.filter((record) => record.event === 'pipeline.grounding.validator.rule.completed');
    expect(ruleEvents.length).toBeGreaterThanOrEqual(5);
    expect(ruleEvents.every((record) => record.traceId === 'validator-trace-1')).toBe(true);
    expect(records.some((record) => record.event === 'pipeline.trace.service.summary' && record.traceId === 'validator-trace-1')).toBe(true);
    expect(result.answer).toBe(answer);
  });


  it('preserves separate initial/corrected validator snapshots and records fallback as the final source', async () => {
    const records: Array<Record<string, unknown>> = [];
    const sink = { debug: (record: Record<string, unknown>) => { records.push(record); }, info: (record: Record<string, unknown>) => { records.push(record); }, warn: (record: Record<string, unknown>) => { records.push(record); }, error: (record: Record<string, unknown>) => { records.push(record); } };
    const pipeline = new PipelineObserver(sink, 'detailed', { enabled: true, level: 'debug', fullText: true, maxTextChars: 4096 });
    const { service } = fixture('Call £25.', 'Call £26.', pipeline);
    const trace = pipeline.createContext({ requestId: 'validator-fallback-request', traceId: 'validator-fallback-trace', entryPoint: 'live-widget-chat', tenantId: 'tenant-a' });
    const result = await service.chat('tenant-a', 'How can I contact Customer Care?', undefined, trace);

    const validatorInputs = records.filter((record) => record.event === 'pipeline.grounding.validator.input');
    expect(validatorInputs).toHaveLength(2);
    expect(validatorInputs.every((record) => record.traceId === 'validator-fallback-trace')).toBe(true);
    expect(validatorInputs[0]?.snapshotId).not.toBe(validatorInputs[1]?.snapshotId);
    const ruleEvents = records.filter((record) => record.event === 'pipeline.grounding.validator.rule.completed');
    expect(ruleEvents.some((record) => record.status === 'rejected' && record.operation === 'unsupported_numeric_claims')).toBe(true);
    expect(records.some((record) => record.event === 'pipeline.fallback.text.selected' && record.traceId === 'validator-fallback-trace')).toBe(true);
    const summaryRecord = records.find((record) => record.event === 'pipeline.trace.service.summary' && record.traceId === 'validator-fallback-trace');
    const metadata = summaryRecord?.metadata as Record<string, unknown>;
    expect(metadata).toMatchObject({ initialAnswerGenerated: true, initialValidation: 'rejected', correctionAttempted: true, correctedValidation: 'rejected', fallbackUsed: true, finalResponseSource: 'fallback' });
    expect((metadata.validatorSnapshotIds as string[])).toHaveLength(2);
    expect(result.answer).toBe('UNKNOWN');
  });

});
