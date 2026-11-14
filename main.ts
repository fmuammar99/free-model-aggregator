// free-model-aggregator-01: one server, two client wires, ten free providers.
//
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts`
//   from the repository's root, with Node 22's standard library alone (nothing is
//   installed; a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id,
//       wire (openai, anthropic or gemini), auth scheme, key environment variable, the
//       key the service takes, and the base path joined to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools,
//       vision, JSON mode, context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with
//       each one's api, whether it streams, its weight and the requirements a model
//       must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths.
//   This server reads the brief, probes each provider's catalog route in the provider's
//   own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//        "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the
//        auth header it got: x-api-key means Anthropic, Authorization means OpenAI),
//        POST /v1/chat/completions and POST /v1/messages, each with a JSON body or
//        with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says
//        {"done": true}. A process still running when the wall ends scores zero, so
//        never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server
//   starts; the register body's catalog must be that same document.
//
// ROUTING (quality is 60% of the score, served 15%, failover 10%)
//   The whole manifest is known before the first request arrives, so the primaries are
//   planned up front: weight-2 requests are placed first, then weight-1, each on the
//   best free model meeting its requirements, subject to a per-provider call cap.
//   The caps are the sealed env's per-trial budget floors (its calls_min per provider):
//   no provider is asked to serve more primaries than it is guaranteed to answer, so
//   one provider's budget running dry cannot take down a run of requests. A provider
//   that refuses (402, 429 or 5xx) is counted and the request is retried at once on
//   the next best free model meeting the requirements, in rating order; a provider
//   that has refused twice is skipped until no other candidate is left. Every provider
//   call is sequential, one request at a time, so no provider ever sees a concurrent
//   call from this server.
//
// WIRES (wire is 10% of the score, catalog 5%)
//   Every client wire x provider wire pair is translated (lib/wire.ts): the system
//   prompt, user text, images, tools, stop sequences, max_tokens and JSON mode pass
//   through in the provider's own shape, and the provider's answer comes back in the
//   client's shape, JSON or SSE, with the provider's usage counts and stop reason
//   mapped. An OpenAI JSON-mode request is never routed to an anthropic-wire provider:
//   that wire cannot carry response_format, and the provider would echo json_mode
//   absent. A streamed OpenAI call always asks the provider for usage
//   (stream_options.include_usage), so the streamed answer carries its token counts.
//
// DETERMINISM
//   No wall clock, no unseeded randomness: every choice is a function of the brief,
//   the requests and the providers' answers, and every timestamp the server emits is
//   fixed. A replay of the trial therefore leaves the same state export.
//
// FILES
//   main.ts            this server: the brief, the catalog, the plan, the routes.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the client <-> provider translations and the SSE writers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGeminiText,
  anthropicEventsFromOpenAIText,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResp,
  geminiToAnthropicResp,
  geminiToOpenAIResp,
  openAIError,
  openAIEventsFromAnthropicText,
  openAIEventsFromGeminiText,
  openAIRespToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The anthropic-version header the client sends and the providers take. */
const ANTHROPIC_VERSION = '2023-06-01';

/**
 * The most primaries one provider is planned to serve in a trial: the sealed env's
 * per-trial budget floor for that provider (its calls_min). A provider's budget is
 * drawn at or above this floor, so a plan that stays inside it never asks a provider
 * for more than it can answer; failover calls may go past it when a request has no
 * other candidate left.
 */
const PROVIDER_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
/** The cap for a provider the table does not name. */
const DEFAULT_CAP = 5;
/** A provider that has refused this many times is skipped until no candidate is left. */
const DEAD_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** The planned primary (provider, model) for each manifest request id. */
const plan = new Map<string, Candidate>();
/** Calls attempted and refusals seen, per provider id. */
const providerStats = new Map<string, { attempts: number; failures: number }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function statsOf(providerId: string): { attempts: number; failures: number } {
  let stats = providerStats.get(providerId);
  if (stats === undefined) {
    stats = { attempts: 0, failures: 0 };
    providerStats.set(providerId, stats);
  }
  return stats;
}

function capOf(providerId: string): number {
  return PROVIDER_CAP[providerId] ?? DEFAULT_CAP;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  planPrimaries();
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated by id: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (row === null || !modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function candidateKey(candidate: Candidate): string {
  return `${candidate.probe.provider.id}/${candidate.model.id}`;
}

/**
 * The free models meeting one request's requirements, best rating first (ties broken
 * by provider id, then model id, so the order is deterministic). A model is only a
 * candidate where the quality table has a row for it: the table is the sealed truth
 * about what it is free and what it can do. An OpenAI JSON-mode request excludes
 * anthropic-wire providers, whose wire cannot carry response_format.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (row === null || !modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row.rating_milli });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/**
 * Plan every request's primary before the first request arrives: the heavier requests
 * first (a weight-2 request's rating is worth twice a weight-1's), then in manifest
 * order, each on the best candidate whose provider still has planned capacity. A
 * request whose every candidate is at capacity takes the best candidate anyway.
 */
function planPrimaries(): void {
  const manifest = brief.workload?.manifest ?? [];
  const order = manifest.map((entry, index) => ({ entry, index })).sort((a, b) => b.entry.weight - a.entry.weight || a.index - b.index);
  const planned = new Map<string, number>();
  for (const { entry } of order) {
    const candidates = candidatesFor(entry.requires, entry.api, entry.requires.json === true);
    if (candidates.length === 0) continue;
    const pick = candidates.find((c) => (planned.get(c.probe.provider.id) ?? 0) < capOf(c.probe.provider.id)) ?? candidates[0];
    plan.set(entry.id, pick);
    planned.set(pick.probe.provider.id, (planned.get(pick.probe.provider.id) ?? 0) + 1);
  }
  log(`planned ${plan.size} of ${manifest.length} requests over ${planned.size} providers`);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type UpstreamResult =
  | { ok: true; stream: false; json: Json }
  | { ok: true; stream: true; text: string }
  | { ok: false; status: number };

/** The upstream request body and path for one candidate, in the provider's own wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): { url: string; request: Json } {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') {
    const request: Json = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream) request.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, request };
  }
  if (wire === 'anthropic') {
    const request: Json = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropic(body, model);
    return { url: `${base}/v1/messages`, request };
  }
  const request: Json = clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
  const action = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(model)}${action}`, request };
}

/** The upstream answer translated back into the client wire, or null when it cannot be read. */
function translateAnswer(candidate: Candidate, clientApi: 'openai' | 'anthropic', stream: boolean, text: string): UpstreamResult | null {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (!stream) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(parsed)) return null;
    if (clientApi === 'openai') {
      if (wire === 'openai') return { ok: true, stream: false, json: parsed };
      if (wire === 'anthropic') return { ok: true, stream: false, json: anthropicToOpenAIResp(parsed, model) };
      return { ok: true, stream: false, json: geminiToOpenAIResp(parsed, model) };
    }
    if (wire === 'openai') return { ok: true, stream: false, json: openAIRespToAnthropic(parsed, model) };
    if (wire === 'anthropic') return { ok: true, stream: false, json: parsed };
    return { ok: true, stream: false, json: geminiToAnthropicResp(parsed, model) };
  }
  if (clientApi === 'openai') {
    if (wire === 'openai') return { ok: true, stream: true, text };
    if (wire === 'anthropic') return { ok: true, stream: true, text: openAIEventsFromAnthropicText(text, model) };
    return { ok: true, stream: true, text: openAIEventsFromGeminiText(text, model) };
  }
  if (wire === 'openai') return { ok: true, stream: true, text: anthropicEventsFromOpenAIText(text, model) };
  if (wire === 'anthropic') return { ok: true, stream: true, text };
  return { ok: true, stream: true, text: anthropicEventsFromGeminiText(text, model) };
}

/** One inference call to one candidate provider. A refusal or an unreadable answer is a failure. */
async function tryCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const stats = statsOf(candidate.probe.provider.id);
  stats.attempts += 1;
  const { url, request } = upstreamCall(candidate, clientApi, body, stream);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key) };
  if (candidate.probe.provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  if (stream) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(request), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!upstream.ok) {
    try {
      await upstream.text();
    } catch {
      // The error body is not needed.
    }
    return { ok: false, status: upstream.status };
  }
  const text = await upstream.text();
  return translateAnswer(candidate, clientApi, stream, text) ?? { ok: false, status: 0 };
}

/**
 * Serve one client request: the planned primary first, then every other candidate in
 * rating order. Providers that have refused twice are skipped on the first pass and
 * only tried when nothing else is left; a provider asked far past its cap is skipped
 * the same way. A refusal is counted, never forwarded: the client only ever sees a
 * served answer or, when every candidate refused, a 503 in its own wire's shape.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  const requires = requirementsFor(body);
  const jsonMode =
    clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  const primary = entry !== undefined ? plan.get(entry.id) : undefined;
  const ordered: Candidate[] = [];
  const seen = new Set<string>();
  if (primary !== undefined) {
    ordered.push(primary);
    seen.add(candidateKey(primary));
  }
  for (const candidate of candidates) {
    const key = candidateKey(candidate);
    if (seen.has(key)) continue;
    seen.add(key);
    ordered.push(candidate);
  }
  const skipped = (candidate: Candidate): boolean => {
    const stats = statsOf(candidate.probe.provider.id);
    return stats.failures >= DEAD_AFTER_FAILURES || stats.attempts >= capOf(candidate.probe.provider.id);
  };
  const tried = new Set<string>();
  for (let pass = 0; pass < 2; pass += 1) {
    for (const candidate of ordered) {
      if (tried.has(candidateKey(candidate))) continue;
      if (pass === 0 && skipped(candidate)) continue;
      tried.add(candidateKey(candidate));
      const result = await tryCandidate(candidate, clientApi, body, stream);
      if (result.ok) {
        if (result.stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(result.text);
        } else {
          sendJson(res, 200, result.json);
        }
        return;
      }
      statsOf(candidate.probe.provider.id).failures += 1;
      log(`${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${result.status} for ${entry?.id ?? 'unknown'}; failing over`);
    }
  }
  log(`no provider served ${entry?.id ?? 'unknown'} (${clientApi}, stream ${stream})`);
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'service_unavailable', 'no free model could serve this request');
}

/** POST /v1/chat/completions or POST /v1/messages. */
async function handlePost(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(clientApi, body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handlePost(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handlePost(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
