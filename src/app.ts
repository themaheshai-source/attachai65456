import Fastify from 'fastify';
import helmet from '@fastify/helmet';
import cors from '@fastify/cors';
import rateLimit from '@fastify/rate-limit';
import multipart from '@fastify/multipart';
import staticPlugin from '@fastify/static';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ZodError } from 'zod';
import type { Env } from './config/env.js';
import { SqliteRepository } from './repositories/sqlite.js';
import { SupabaseRepository } from './repositories/supabase.js';
import { LocalStorage, SupabaseStorage } from './providers/storage.js';
import { LocalEmbedding, OpenAIEmbedding } from './providers/embedding.js';
import { LocalVector, QdrantVector, SupabaseVector } from './providers/vector.js';
import { createLlmProvider } from './providers/llm.js';
import {
  OutboxNotification,
  SmtpNotification,
  SupabaseOutboxNotification,
} from './providers/notification.js';
import { ClientService } from './services/clients.js';
import { KnowledgeService } from './services/knowledge.js';
import { ChatService } from './services/chat.js';
import { Metrics } from './lib/metrics.js';
import { createRequestContext, PipelineObserver, safeErrorCode } from './lib/observability.js';
import { AppError } from './lib/errors.js';
import { adminRoutes } from './routes/admin.js';
import { publicRoutes } from './routes/public.js';
import { systemRoutes } from './routes/system.js';
export async function buildApp(env: Env) {
  const app = Fastify({
    logger:
      env.NODE_ENV === 'test'
        ? false
        : {
            level: env.PIPELINE_DEBUG_LOG_LEVEL,
            redact: [
              'req.headers.x-api-key',
              'req.headers.x-admin-api-key',
              'req.headers.authorization',
            ],
          },
    bodyLimit: 1024 * 1024,
    requestTimeout: 30000,
  });
  const pipeline = new PipelineObserver(app.log, env.PIPELINE_DIAGNOSTICS, {
    enabled: env.PIPELINE_DEBUG_LOGS_ENABLED,
    level: env.PIPELINE_DEBUG_LOG_LEVEL,
    fullText: env.PIPELINE_DEBUG_LOG_FULL_TEXT,
    maxTextChars: env.PIPELINE_DEBUG_LOG_MAX_TEXT_CHARS,
  });
  const supabaseKey = env.SUPABASE_SERVICE_ROLE_KEY ?? env.SUPABASE_ANON_KEY;
  const repo =
    env.DATABASE_PROVIDER === 'supabase'
      ? new SupabaseRepository(env.SUPABASE_URL!, supabaseKey)
      : new SqliteRepository(path.join(env.DATA_DIR, 'metadata.sqlite'));
  await repo.init();
  const storage =
    env.STORAGE_PROVIDER === 'supabase'
      ? new SupabaseStorage(env.SUPABASE_URL!, env.SUPABASE_STORAGE_BUCKET, supabaseKey)
      : new LocalStorage(path.join(env.DATA_DIR, 'objects'));
  const embedding =
    env.EMBEDDING_PROVIDER === 'openai'
      ? new OpenAIEmbedding(env.OPENAI_BASE_URL, env.OPENAI_API_KEY, env.OPENAI_EMBEDDING_MODEL)
      : new LocalEmbedding();
  const vector =
    env.VECTOR_PROVIDER === 'qdrant'
      ? new QdrantVector(env.QDRANT_URL, env.QDRANT_API_KEY)
      : env.VECTOR_PROVIDER === 'supabase'
        ? new SupabaseVector(env.SUPABASE_URL!, supabaseKey)
        : new LocalVector(path.join(env.DATA_DIR, 'vectors.json'));
  const llm = createLlmProvider(env.llm);
  const notification =
    env.NOTIFICATION_PROVIDER === 'smtp'
      ? new SmtpNotification({
          host: env.SMTP_HOST,
          port: env.SMTP_PORT,
          user: env.SMTP_USER,
          pass: env.SMTP_PASS,
          from: env.SMTP_FROM,
        })
      : env.NOTIFICATION_PROVIDER === 'supabase'
        ? new SupabaseOutboxNotification(env.SUPABASE_URL!, env.SUPABASE_OUTBOX_TABLE, supabaseKey)
        : new OutboxNotification(path.join(env.DATA_DIR, 'outbox'));
  const metrics = new Metrics();
  const clients = new ClientService(repo, storage, vector, env.PUBLIC_BASE_URL);
  const knowledge = new KnowledgeService(storage, embedding, vector, repo, pipeline);
  const chat = new ChatService(repo, embedding, vector, llm, pipeline);
  const ctx = {
    env,
    repo,
    storage,
    embedding,
    vector,
    llm,
    notification,
    metrics,
    pipeline,
    clients,
    knowledge,
    chat,
  };
  await app.register(helmet, {
    contentSecurityPolicy: false,
    crossOriginResourcePolicy: false,
  });
  await app.register(cors, {
    delegator: (req, cb) => {
      const publicRequest = req.url === '/v1' || req.url.startsWith('/v1/');
      cb(null, {
        origin: publicRequest
          ? true
          : (origin, originCb) => {
              if (!origin || env.NODE_ENV !== 'production' || env.CORS_ORIGINS.includes(origin))
                originCb(null, true);
              else originCb(null, false);
            },
        methods: ['GET', 'POST', 'PATCH', 'PUT', 'DELETE'],
        allowedHeaders: ['content-type', 'x-admin-api-key', 'x-api-key', 'x-request-id', 'x-trace-id'],
      });
    },
  });
  await app.register(rateLimit, {
    max: 100,
    timeWindow: '1 minute',
    keyGenerator: (req) => req.ip,
  });
  await app.register(multipart, { limits: { fileSize: 10 * 1024 * 1024, files: 10, parts: 15 } });
  const publicRoot =
    [
      fileURLToPath(new URL('../public', import.meta.url)),
      fileURLToPath(new URL('../../public', import.meta.url)),
      path.resolve('public'),
    ].find((candidate) => fs.existsSync(candidate)) ?? path.resolve('public');
  await app.register(staticPlugin, {
    root: publicRoot,
    prefix: '/',
    setHeaders: (res, pathName) => {
      if (pathName.endsWith('widget.js')) {
        res.header('Cross-Origin-Resource-Policy', 'cross-origin');
        res.header('Cache-Control', 'no-store');
      }

      if (pathName.endsWith('admin/index.html')) {
        res.header('Cache-Control', 'no-store');
      }
    },
  });
  app.addHook('onRequest', async (req) => {
    const extended = req as typeof req & { startedAt: bigint; pipelineContext: ReturnType<typeof createRequestContext> };
    extended.startedAt = process.hrtime.bigint();
    extended.pipelineContext = createRequestContext(pipeline, req as typeof req & { headers: Record<string, unknown> });
    pipeline.emit(extended.pipelineContext, 'pipeline.request.started', {
      stage: 'request', operation: 'http_request', status: 'started',
      inputMetrics: { method: req.method, route: (req.routeOptions?.url ?? req.url.split('?')[0] ?? 'unknown').replace(/\/[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/gi, '/:id').slice(0, 160), bodyBytes: typeof req.headers['content-length'] === 'string' ? Number(req.headers['content-length']) || 0 : null },
    });
  });
  app.addHook('onResponse', async (req, reply) => {
    const extended = req as typeof req & { startedAt: bigint; pipelineContext?: ReturnType<typeof createRequestContext> };
    const durationMs = Number(process.hrtime.bigint() - (extended.startedAt ?? process.hrtime.bigint())) / 1_000_000;
    metrics.record(durationMs, reply.statusCode);
    if (extended.pipelineContext) {
      const rawContentLength = reply.getHeader('content-length');
      const declaredResponseBytes = typeof rawContentLength === 'number'
        ? rawContentLength
        : typeof rawContentLength === 'string' && /^\d+$/.test(rawContentLength) ? Number(rawContentLength) : null;
      pipeline.emit(extended.pipelineContext, 'pipeline.request.completed', {
        stage: 'request', operation: 'http_response', status: reply.statusCode >= 400 ? 'failed' : 'success', durationMs,
        outputMetrics: { httpStatus: reply.statusCode, declaredResponseBytes },
        reason: reply.statusCode >= 400 ? 'http_error_response' : null,
      });
      pipeline.emitForensic(extended.pipelineContext, 'pipeline.trace.summary', {
        stage: 'pipeline.trace', operation: 'summarize_http_request',
        status: reply.statusCode >= 400 ? 'failed' : 'success', durationMs,
        reason: reply.statusCode >= 400 ? 'http_error_response' : null,
        metadata: {
          terminalBoundary: 'fastify_onResponse',
          httpStatus: reply.statusCode,
          declaredResponseBytes,
          clientReceiptConfirmed: false,
          service: extended.pipelineContext.traceSummary ?? null,
          delivery: extended.pipelineContext.deliverySummary ?? null,
        },
        output: {
          httpStatus: reply.statusCode,
          declaredResponseBytes,
          finalResponseSource: extended.pipelineContext.traceSummary?.finalResponseSource ?? 'unknown',
          clientReceiptConfirmed: false,
        },
      });
    }
  });
  app.setErrorHandler((e, req, reply) => {
    const extended = req as typeof req & { pipelineContext?: ReturnType<typeof createRequestContext> };
    if (extended.pipelineContext) pipeline.emit(extended.pipelineContext, e instanceof ZodError ? 'pipeline.request.rejected' : 'pipeline.request.failed', {
      stage: 'request.validation', operation: 'request_handler', status: e instanceof ZodError ? 'rejected' : 'failed',
      errorCode: e instanceof ZodError ? 'VALIDATION_ERROR' : (e instanceof AppError ? e.code : safeErrorCode(e)),
      outputMetrics: { httpStatus: e instanceof ZodError ? 400 : e instanceof AppError ? e.statusCode : 500 },
    });
    if (e instanceof ZodError)
      return reply.code(400).send({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Request validation failed',
          details: e.issues.map((x) => ({ path: x.path.join('.'), message: x.message })),
        },
      });
    if (e instanceof AppError)
      return reply.code(e.statusCode).send({ error: { code: e.code, message: e.message } });
    if ((e as { code?: string }).code === 'SQLITE_CONSTRAINT_UNIQUE')
      return reply
        .code(409)
        .send({ error: { code: 'CONFLICT', message: 'Resource already exists' } });
    app.log.error({ requestId: extended.pipelineContext?.requestId ?? req.id, errorCode: safeErrorCode(e), errorName: e instanceof Error ? e.name : typeof e }, 'request failed');
    return reply
      .code(500)
      .send({ error: { code: 'INTERNAL_ERROR', message: 'An internal error occurred' } });
  });
  await systemRoutes(app, ctx);
  await app.register(async (a) => adminRoutes(a, ctx), { prefix: '/admin' });
  await app.register(async (a) => publicRoutes(a, ctx), { prefix: '/v1' });
  app.addHook('onClose', async () => repo.close());
  return app;
}
