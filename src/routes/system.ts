import type { FastifyInstance, FastifyRequest } from 'fastify';
import type { Context } from './context.js';
import type { PipelineContext } from '../lib/observability.js';

type HealthResult = { provider: string; connected: boolean };

export async function systemRoutes(app: FastifyInstance, c: Context) {
  const traceFor = (request: FastifyRequest) => (request as FastifyRequest & { pipelineContext?: PipelineContext }).pipelineContext
    ?? c.pipeline.createContext({ requestId: request.id, entryPoint: 'system-health-api' });
  const healthCheck = async <T extends HealthResult>(telemetry: PipelineContext, name: string, work: () => Promise<T>) => {
    const result = await c.pipeline.measure(telemetry, `health.${name}`, 'provider_health_check', work, { inputMetrics: { provider: name } });
    c.pipeline.emit(telemetry, 'pipeline.provider.health.completed', {
      stage: `health.${name}`, operation: 'provider_health_check', status: result.connected ? 'success' : 'failed',
      outputMetrics: { provider: result.provider, connected: result.connected }, reason: result.connected ? null : 'provider_reported_disconnected',
    });
    return result;
  };

  app.get('/health', async (request) => {
    const telemetry = traceFor(request);
    return {
      status: 'ok',
      providers: {
        database: await healthCheck(telemetry, 'database', () => c.repo.health()),
        storage: await healthCheck(telemetry, 'storage', () => c.storage.health()),
        vector: await healthCheck(telemetry, 'vector', () => c.vector.health()),
        embedding: await healthCheck(telemetry, 'embedding', () => c.embedding.health()),
        llm: await healthCheck(telemetry, 'llm', () => c.llm.health()),
        notification: await healthCheck(telemetry, 'notification', () => c.notification.health()),
      },
      ...c.metrics.snapshot(),
    };
  });

  app.get('/ready', async (request, reply) => {
    const telemetry = traceFor(request);
    const h = await Promise.all([
      healthCheck(telemetry, 'database', () => c.repo.health()),
      healthCheck(telemetry, 'storage', () => c.storage.health()),
      healthCheck(telemetry, 'vector', () => c.vector.health()),
      healthCheck(telemetry, 'embedding', () => c.embedding.health()),
      healthCheck(telemetry, 'llm', () => c.llm.health()),
      healthCheck(telemetry, 'notification', () => c.notification.health()),
    ]);
    const ready = h.every((x) => x.connected);
    return reply.code(ready ? 200 : 503).send({ ready, providers: h });
  });
  app.get('/metrics', async (_r, reply) =>
    reply.type('text/plain; version=0.0.4').send(c.metrics.prometheus() + c.pipeline.prometheus()),
  );
}
