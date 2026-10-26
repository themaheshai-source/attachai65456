import type { FastifyInstance, FastifyRequest } from 'fastify';
import type { Context } from './context.js';
import { hashKey, originMatches } from '../lib/security.js';
import { AppError } from '../lib/errors.js';
import type { PipelineContext } from '../lib/observability.js';
import { safeErrorCode } from '../lib/observability.js';
import { chatSchema, leadSchema } from '../domain/schemas.js';

type TenantRequest = FastifyRequest & { tenantId: string; pipelineContext?: PipelineContext; startedAt?: number };
const traceFor = (req: FastifyRequest, c: Context) => (req as TenantRequest).pipelineContext ?? c.pipeline.createContext({ requestId: req.id, entryPoint: 'live-widget-api' });
export async function publicRoutes(app: FastifyInstance, c: Context) {
  const tenant = async (req: FastifyRequest) => {
    const telemetry = traceFor(req, c);
    const key = req.headers['x-api-key'], origin = req.headers.origin;
    if (typeof key !== 'string') {
      c.pipeline.emit(telemetry, 'pipeline.auth.failed', { stage: 'authentication', operation: 'public_api_key', status: 'rejected', errorCode: 'INVALID_API_KEY', reason: 'missing_public_credential' });
      throw new AppError(401, 'INVALID_API_KEY', 'Valid public client credential required');
    }
    const client = await c.pipeline.measure(telemetry, 'authentication', 'public_client_lookup', () => c.repo.getClientByKeyHash(hashKey(key)), { inputMetrics: { credentialPresent: true } });
    if (!client?.enabled) {
      c.pipeline.emit(telemetry, 'pipeline.auth.failed', { stage: 'authentication', operation: 'public_api_key', status: 'rejected', errorCode: 'INVALID_API_KEY', reason: client ? 'client_disabled' : 'credential_not_found' });
      throw new AppError(401, 'INVALID_API_KEY', 'Valid public client credential required');
    }
    const domains = await c.pipeline.measure(telemetry, 'authorization', 'allowed_domains_lookup', () => c.repo.domains(client.id));
    const originAllowed = typeof origin === 'string' && originMatches(origin, domains);
    if (!originAllowed) {
      c.pipeline.emit(telemetry, 'pipeline.authorization.failed', { stage: 'authorization', operation: 'origin_allowlist', status: 'rejected', errorCode: 'ORIGIN_FORBIDDEN', reason: typeof origin === 'string' ? 'origin_not_allowlisted' : 'origin_missing', outputMetrics: { allowedDomainCount: domains.length } });
      throw new AppError(403, 'ORIGIN_FORBIDDEN', 'Origin is not allowed');
    }
    (req as TenantRequest).tenantId = client.id;
    c.pipeline.setTenant(telemetry, client.id);
    c.pipeline.emit(telemetry, 'pipeline.authorization.completed', { stage: 'authorization', operation: 'tenant_and_origin', status: 'success', outputMetrics: { originAllowed: true, allowedDomainCount: domains.length } });
  };
  app.addHook('preHandler', tenant);
  app.get('/config', async (req) => {
    const x = await c.repo.getClient((req as TenantRequest).tenantId);
    if (!x) throw new AppError(404, 'NOT_FOUND', 'Client not found');
    return { clientName: x.name, assistantName: x.config.assistantName, welcomeMessage: x.config.welcomeMessage, widget: x.config.widget ?? {}, theme: x.config.theme ?? {} };
  });
  app.post('/chat', async (req) => {
    const telemetry = traceFor(req, c);
    const b = await c.pipeline.measure(telemetry, 'request.validation', 'chat_schema', async () => chatSchema.parse(req.body), { inputMetrics: { bodyType: typeof req.body } });
    const tenantId = (req as TenantRequest).tenantId;
    c.pipeline.emit(telemetry, 'pipeline.input.accepted', { stage: 'input', operation: 'chat_message', status: 'success', inputMetrics: { messageChars: b.message.length, sessionIdProvided: Boolean(b.sessionId) } });
    return c.chat.chat(tenantId, b.message, b.sessionId, telemetry);
  });
  app.post('/chat/stream', async (req, reply) => {
    const telemetry = traceFor(req, c);
    const b = await c.pipeline.measure(telemetry, 'request.validation', 'chat_schema', async () => chatSchema.parse(req.body), { inputMetrics: { bodyType: typeof req.body } });
    c.pipeline.emit(telemetry, 'pipeline.input.accepted', { stage: 'input', operation: 'chat_stream_message', status: 'success', inputMetrics: { messageChars: b.message.length, sessionIdProvided: Boolean(b.sessionId) } });
    // Hijacking bypasses Fastify's normal response hooks, so preserve the
    // already-authorized exact origin for browser-readable tenant streaming.
    const origin = req.headers.origin;
    if (typeof origin === 'string') {
      reply.raw.setHeader('access-control-allow-origin', origin);
      reply.raw.setHeader('vary', 'Origin');
    }
    reply.hijack();
    reply.raw.statusCode = 200;
    reply.raw.setHeader('content-type', 'text/event-stream; charset=utf-8');
    reply.raw.setHeader('cache-control', 'no-cache, no-transform');
    reply.raw.setHeader('connection', 'keep-alive');
    reply.raw.flushHeaders?.();
    const streamStarted = process.hrtime.bigint();
    let tokenEvents = 0, tokenChars = 0, firstTokenMs: number | undefined;
    let disconnected = false;
    const onClose = () => {
      if (!reply.raw.writableEnded) {
        disconnected = true;
        c.pipeline.emit(telemetry, 'pipeline.stream.disconnected', { stage: 'response.stream', operation: 'client_connection', status: 'cancelled', durationMs: Number(process.hrtime.bigint() - streamStarted) / 1_000_000, outputMetrics: { tokenEvents, tokenChars } });
      }
    };
    reply.raw.on('close', onClose);
    try {
      const result = await c.chat.stream((req as TenantRequest).tenantId, b.message, b.sessionId, (token) => {
        if (firstTokenMs === undefined) firstTokenMs = Number(process.hrtime.bigint() - streamStarted) / 1_000_000;
        tokenEvents++; tokenChars += token.length;
        reply.raw.write(`event: token\ndata: ${JSON.stringify({ token })}\n\n`);
      }, telemetry);
      if (!disconnected) {
        reply.raw.write(`event: done\ndata: ${JSON.stringify(result)}\n\n`);
        c.pipeline.emit(telemetry, 'pipeline.stream.done_event_sent', { stage: 'response.stream', operation: 'sse_done_event', status: 'success', durationMs: Number(process.hrtime.bigint() - streamStarted) / 1_000_000, outputMetrics: { tokenEvents, tokenChars, firstTokenMs: firstTokenMs ?? null, responseChars: result.answer.length, sourceCount: result.sources.length, socketWritable: reply.raw.writable } });
      }
    } catch (error) {
      c.pipeline.emit(telemetry, 'pipeline.stream.failed', { stage: 'response.stream', operation: 'stream_generation_and_delivery', status: disconnected ? 'cancelled' : 'failed', durationMs: Number(process.hrtime.bigint() - streamStarted) / 1_000_000, errorCode: safeErrorCode(error), outputMetrics: { tokenEvents, tokenChars, firstTokenMs: firstTokenMs ?? null }, reason: disconnected ? 'connection_closed' : 'stream_error' });
      if (!disconnected && !reply.raw.destroyed) reply.raw.write(`event: error\ndata: ${JSON.stringify({ message: 'Unable to complete the response' })}\n\n`);
    } finally {
      reply.raw.off('close', onClose);
      reply.raw.end();
      c.pipeline.emit(telemetry, 'pipeline.request.completed', { stage: 'request', operation: 'http_stream_response', status: disconnected ? 'cancelled' : reply.raw.destroyed ? 'failed' : 'success', durationMs: Number(process.hrtime.bigint() - streamStarted) / 1_000_000, outputMetrics: { httpStatus: reply.raw.statusCode, tokenEvents, tokenChars, firstTokenMs: firstTokenMs ?? null, writableEnded: reply.raw.writableEnded }, reason: disconnected ? 'client_disconnected' : null });
    }
  });
  app.post('/leads', async (req) => {
    const b = leadSchema.parse(req.body);
    const tid = (req as TenantRequest).tenantId;
    const client = await c.repo.getClient(tid);
    if (!client?.enabled) throw new AppError(403, 'TENANT_DISABLED', 'Tenant is disabled');
    const { conversationId, ...rawData } = b;
    const data = Object.fromEntries(
      Object.entries(rawData).filter(([, value]) => value !== undefined),
    ) as Record<string, string>;
    for (const value of Object.values(data)) {
      if (/\r|\n/.test(value)) throw new AppError(400, 'INVALID_LEAD', 'Lead fields contain invalid characters');
    }
    if (conversationId && !(await c.repo.conversationForClient(tid, conversationId)))
      throw new AppError(404, 'CONVERSATION_NOT_FOUND', 'Conversation not found');
    const destination = client.config.notificationEmail ?? client.config.teamEmail;
    if (!destination && c.notification.name === 'smtp')
      throw new AppError(503, 'LEAD_DELIVERY_UNAVAILABLE', 'Lead delivery is not configured');
    const id = await c.repo.saveLead(tid, conversationId, data);
    try {
      await c.notification.notify(destination, `New lead for ${client.name}`, {
        id,
        client: client.name,
        receivedAt: new Date().toISOString(),
        name: data.name,
        email: data.email,
        message: data.message ?? data.requirement,
        phone: data.phone,
      }, tid, { replyTo: data.email });
    } catch {
      throw new AppError(503, 'LEAD_DELIVERY_FAILED', 'Unable to send your message. Please try again.');
    }
    await c.repo.audit(tid, 'lead.created', { leadId: id });
    return { id, status: 'accepted' };
  });
}
