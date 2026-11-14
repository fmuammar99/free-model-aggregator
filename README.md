# Free AI model aggregator — make.fun entry

Server that serves OpenAI Chat Completions and Anthropic Messages behind one
endpoint, routing every request to the best free model that meets its
requirements, with failover when a provider refuses.

Built for the make.fun competition `free-model-aggregator-01` by
@fadlilmu-ammar. Node 22, standard library only, one process.

- `main.ts` — the server: `POST /v1/chat/completions`, `POST /v1/messages`,
  `GET /v1/models`, registration with the trial environment, `--list` catalog.
- `lib/wire.ts` — translation between OpenAI, Anthropic and Gemini wires.
- `lib/providers.ts` — brief parsing, per-provider auth, catalog probing.

Grade-local (3 practice trials): **99/100** mean — quality 99, served 100,
wire 100, failover 100, catalog 100, replay byte-for-byte identical.
