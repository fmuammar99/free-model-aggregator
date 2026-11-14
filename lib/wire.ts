// The client wires and the three upstream wires this entry speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. This entry
// forwards each request to a provider on any of the three wires and translates back:
//
//   client openai     -> upstream openai:    body passed through (model replaced)
//   client openai     -> upstream anthropic: OpenAI body rewritten as a Messages request
//   client openai     -> upstream gemini:    OpenAI body rewritten as generateContent
//   client anthropic  -> upstream openai:    Messages request rewritten as Chat Completions
//   client anthropic  -> upstream anthropic: body passed through (model replaced)
//   client anthropic  -> upstream gemini:    Messages request rewritten as generateContent
//
// The translations are written so the provider's echo of the request matches the client's
// body field for field: the system prompt, user text, images (sha of the same bytes), tool
// names, max_tokens, stop sequences and JSON mode. JSON mode is only routed to wires that
// can carry it (openai's response_format, gemini's responseMimeType); the Anthropic wire
// cannot, so an OpenAI JSON-mode request never goes to an anthropic-wire provider.
//
// Streaming translations are buffered: the upstream stream is read whole (provider answers
// are one short ANSWER text), then re-emitted in the client wire's event order. That keeps
// failover possible (a refused upstream call is retried on another provider before anything
// is sent to the client) and guarantees a well-formed event sequence. Everything here is a
// pure function of its arguments: no clock and no randomness, so the same body and the same
// provider answer give the same bytes on a replay. The model listings use fixed timestamps
// for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function finishReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  if (reason === 'TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
function stopReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  if (reason === 'TOOL_CALLS') return 'tool_use';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function stringListOf(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.filter((x): x is string => typeof x === 'string');
  return [];
}

/** An OpenAI image_url part as a Gemini content part: inlineData for a data URL, fileData else. */
function geminiImagePart(url: unknown): Json | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** An Anthropic image block as a Gemini content part. */
function geminiImagePartFromAnthropic(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** One OpenAI message's content as Gemini parts (text and images). */
function geminiPartsOfOpenAIContent(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url)) {
      const part = geminiImagePart(block.image_url.url);
      if (part !== null) parts.push(part);
    }
  }
  return parts;
}

/** One Anthropic message's content as Gemini parts (text, images, tool results). */
function geminiPartsOfAnthropicContent(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const part = geminiImagePartFromAnthropic(block.source);
      if (part !== null) parts.push(part);
    } else if (block.type === 'tool_result') {
      parts.push({
        functionResponse: {
          name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          response: { content: toolResultText(block.content) },
        },
      });
    }
  }
  return parts;
}

/** OpenAI tools as Gemini functionDeclarations. */
function geminiDeclarationsOfOpenAITools(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        ...(typeof raw.function.description === 'string' ? { description: raw.function.description } : {}),
        ...(isRecord(raw.function.parameters) ? { parameters: raw.function.parameters } : {}),
      },
    ];
  });
}

/** Anthropic tools as Gemini functionDeclarations. */
function geminiDeclarationsOfAnthropicTools(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        ...(typeof raw.description === 'string' ? { description: raw.description } : {}),
        ...(isRecord(raw.input_schema) ? { parameters: raw.input_schema } : {}),
      },
    ];
  });
}

/** The generationConfig for a Gemini call, from the client's caps (max tokens, stops, JSON mode). */
function geminiConfig(opts: {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stops?: string[];
  jsonMode?: boolean;
}): Json {
  const config: Json = {};
  if (opts.maxTokens !== undefined) config.maxOutputTokens = opts.maxTokens;
  if (opts.temperature !== undefined) config.temperature = opts.temperature;
  if (opts.topP !== undefined) config.topP = opts.topP;
  if (opts.stops !== undefined && opts.stops.length > 0) config.stopSequences = opts.stops;
  if (opts.jsonMode === true) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the system prompt
 * becomes the systemInstruction, messages become contents (user/model roles, images as
 * inlineData, tool calls as functionCall, tool results as functionResponse), tools become
 * functionDeclarations, and the caps become the generationConfig.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemText = '';
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = geminiPartsOfOpenAIContent(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      contents.push({ role: 'user', parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: text } } }] });
      continue;
    }
    const parts = geminiPartsOfOpenAIContent(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = geminiDeclarationsOfOpenAITools(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  out.generationConfig = geminiConfig({
    maxTokens: numberOrUndefined(body.max_tokens) ?? numberOrUndefined(body.max_completion_tokens),
    temperature: numberOrUndefined(body.temperature),
    topP: numberOrUndefined(body.top_p),
    stops: stringListOf(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
  });
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system prompt becomes
 * the systemInstruction, user messages become user contents (images as inlineData, tool
 * results as functionResponse), assistant messages become model contents (tool uses as
 * functionCall), tools become functionDeclarations.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const parts = geminiPartsOfAnthropicContent(raw.content);
    if (raw.role === 'assistant') {
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = geminiDeclarationsOfAnthropicTools(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  out.generationConfig = geminiConfig({
    maxTokens: numberOrUndefined(body.max_tokens),
    temperature: numberOrUndefined(body.temperature),
    topP: numberOrUndefined(body.top_p),
    stops: stringListOf(body.stop_sequences),
  });
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request: the system prompt
 * becomes `system`, messages become Anthropic blocks (images as base64 image blocks, tool
 * calls as tool_use, tool results as tool_result), tools and tool_choice are rewritten, and
 * the caps pass through. JSON mode cannot be carried on this wire: the caller must not
 * route a JSON-mode request here.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemText = (() => {
    const parts: string[] = [];
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (isRecord(raw) && raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') parts.push(text);
      }
    }
    return parts.join('\n');
  })();
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): void => {
    if (pendingToolResults.length > 0) {
      messages.push({ role: 'user', content: pendingToolResults });
      pendingToolResults = [];
    }
  };
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      });
      continue;
    }
    flushToolResults();
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    const content: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') content.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') content.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) content.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else content.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (content.length > 0) messages.push({ role: 'user', content });
  }
  flushToolResults();

  const out: Json = { model, messages };
  if (systemText !== '') out.system = systemText;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringListOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          ...(typeof raw.function.description === 'string' ? { description: raw.function.description } : {}),
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIRespToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIResp(body: Json, model: string): Json {
  const content: Json[] = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text = text === '' ? block.text : `${text}${block.text}`;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/** The text and usage of a Gemini answer: its candidates' text parts, finish reason and token counts. */
function geminiAnswerOf(body: Json): { text: string; finish: unknown; input: number; output: number; toolCalls: Json[] } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: { name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
      });
    }
  }
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, finish: candidate.finishReason, input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount), toolCalls };
}

/** A Gemini answer as a Chat Completions object. */
export function geminiToOpenAIResp(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) message.tool_calls = answer.toolCalls;
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: answer.toolCalls.length > 0 ? 'tool_calls' : finishReasonOfGemini(answer.finish), logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiToAnthropicResp(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    if (!isRecord(call.function)) continue;
    content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: answer.toolCalls.length > 0 ? 'tool_use' : stopReasonOfGemini(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One SSE frame of raw text: its event name (null when unnamed) and its data lines joined. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** One OpenAI chunk: its delta's content, tool calls, finish reason and usage, when the frame holds one. */
interface OpenAIChunk {
  content: string;
  toolCalls: { index: number; id: string; name: string; args: string }[];
  finish: string | null;
  usage: { input: number; output: number } | null;
}

function openAIChunkOf(payload: string): OpenAIChunk | null {
  let chunk: unknown;
  try {
    chunk = JSON.parse(payload);
  } catch {
    return null;
  }
  if (!isRecord(chunk) || !Array.isArray(chunk.choices)) return null;
  const out: OpenAIChunk = { content: '', toolCalls: [], finish: null, usage: null };
  if (isRecord(chunk.usage)) out.usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
  if (chunk.choices.length === 0) return out;
  const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
  if (choice === null) return out;
  const delta = isRecord(choice.delta) ? choice.delta : {};
  if (typeof delta.content === 'string') out.content = delta.content;
  for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
    if (!isRecord(call)) continue;
    const fn = isRecord(call.function) ? call.function : {};
    out.toolCalls.push({
      index: typeof call.index === 'number' ? call.index : 0,
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof fn.name === 'string' ? fn.name : '',
      args: typeof fn.arguments === 'string' ? fn.arguments : '',
    });
  }
  if (typeof choice.finish_reason === 'string') out.finish = choice.finish_reason;
  return out;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the
 * first chunk, content_block_start/delta/stop per text block or tool call, then
 * message_delta with the stop reason and usage, then message_stop. The whole upstream
 * stream is read first, so message_start carries the true input token count.
 */
export function anthropicEventsFromOpenAIText(text: string, model: string): string {
  let content = '';
  const calls = new Map<number, { id: string; name: string; args: string }>();
  let finish: string | null = null;
  let usage: { input: number; output: number } = { input: 0, output: 0 };
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = openAIChunkOf(frame.data);
    if (chunk === null) continue;
    content += chunk.content;
    for (const call of chunk.toolCalls) {
      const at = calls.get(call.index) ?? { id: '', name: '', args: '' };
      if (call.id !== '') at.id = call.id;
      if (call.name !== '') at.name = call.name;
      at.args += call.args;
      calls.set(call.index, at);
    }
    if (chunk.finish !== null) finish = chunk.finish;
    if (chunk.usage !== null) usage = chunk.usage;
  }
  const events: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
  ];
  let index = 0;
  if (content !== '') {
    events.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: content } }));
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of [...calls.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v)) {
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.args !== '') events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.args } }));
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  events.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReasonOf(finish ?? 'stop'), stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events.join('');
}

/** One Anthropic SSE stream, parsed: its text, tool calls, stop reason and token counts. */
interface AnthropicStream {
  text: string;
  calls: { id: string; name: string; args: string }[];
  stop: string | null;
  input: number;
  output: number;
}

function anthropicStreamOf(text: string): AnthropicStream {
  const out: AnthropicStream = { text: '', calls: [], stop: null, input: 0, output: 0 };
  const byIndex = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) out.input = numberOrZero(data.message.usage.input_tokens);
    else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : byIndex.size;
      byIndex.set(index, { id: typeof data.content_block.id === 'string' ? data.content_block.id : '', name: typeof data.content_block.name === 'string' ? data.content_block.name : '', args: '' });
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out.text += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const index = typeof data.index === 'number' ? data.index : 0;
        const at = byIndex.get(index) ?? { id: '', name: '', args: '' };
        at.args += data.delta.partial_json;
        byIndex.set(index, at);
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') out.stop = data.delta.stop_reason;
      if (isRecord(data.usage)) out.output = numberOrZero(data.usage.output_tokens);
    }
  }
  out.calls = [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
  return out;
}

/**
 * An Anthropic SSE stream as OpenAI chunks: a role chunk, the text deltas, the tool-call
 * deltas, one finish chunk, the usage chunk and [DONE].
 */
export function openAIEventsFromAnthropicText(text: string, model: string): string {
  const stream = anthropicStreamOf(text);
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (stream.text !== '') frames.push(sseEvent(null, chunk({ content: stream.text }, null)));
  stream.calls.forEach((call, i) => {
    frames.push(
      sseEvent(null, chunk({ tool_calls: [{ index: i, id: call.id !== '' ? call.id : `call_${i}`, type: 'function', function: { name: call.name, arguments: call.args } }] }, null)),
    );
  });
  frames.push(sseEvent(null, chunk({}, stream.calls.length > 0 ? 'tool_calls' : finishReasonOf(stream.stop ?? 'end_turn'))));
  frames.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: stream.input, completion_tokens: stream.output, total_tokens: stream.input + stream.output } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One Gemini SSE stream, parsed: its text pieces, finish reason and token counts. */
function geminiStreamOf(text: string): { text: string; finish: unknown; input: number; output: number } {
  let outText = '';
  let finish: unknown = null;
  let input = 0;
  let output = 0;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const answer = geminiAnswerOf(data);
    outText += answer.text;
    if (answer.finish !== undefined && answer.finish !== null) finish = answer.finish;
    if (isRecord(data.usageMetadata)) {
      input = numberOrZero(data.usageMetadata.promptTokenCount);
      output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: outText, finish, input, output };
}

/** A Gemini SSE stream as OpenAI chunks: role, text, finish, usage, [DONE]. */
export function openAIEventsFromGeminiText(text: string, model: string): string {
  const stream = geminiStreamOf(text);
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (stream.text !== '') frames.push(sseEvent(null, chunk({ content: stream.text }, null)));
  frames.push(sseEvent(null, chunk({}, finishReasonOfGemini(stream.finish))));
  frames.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: stream.input, completion_tokens: stream.output, total_tokens: stream.input + stream.output } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A Gemini SSE stream as the Anthropic event sequence. */
export function anthropicEventsFromGeminiText(text: string, model: string): string {
  const stream = geminiStreamOf(text);
  const events: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: stream.input, output_tokens: 0 } },
    }),
  ];
  if (stream.text !== '') {
    events.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
    events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: stream.text } }));
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  }
  events.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReasonOfGemini(stream.finish), stop_sequence: null }, usage: { output_tokens: stream.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events.join('');
}
