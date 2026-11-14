# free-model-aggregator-01 — STATUS

Bounty: **free-model-aggregator-01** ("The best free AI model aggregator"), pot $500,
deadline entry **2026-11-16T02:30:00Z** (Senin 2026-11-16 09:30 WIB). ESCROWED — submit
nanti wajib `--keypair`.

## Yang dibangun

Server agregator model gratis di **Node 22 stdlib-only**, di `~/workspace/makefun/aggregator/`:

- `main.ts` (551 baris) — server HTTP: `POST /v1/chat/completions`, `POST /v1/messages`,
  `GET /v1/models` (bentuk OpenAI untuk `Authorization: Bearer`, bentuk Anthropic untuk
  `x-api-key`), registrasi ke env (`POST /register`), polling `GET /work`, exit 0 saat done,
  `node main.ts --list` mencetak katalog yang sama byte-for-byte dengan yang diregistrasi.
- `lib/providers.ts` (432 baris, dari starter) — parser brief (provider directory, quality
  table, workload manifest), auth header per provider, probe katalog 3 wire.
- `lib/wire.ts` (902 baris) — translasi **semua 6 kombinasi** client × provider wire
  (OpenAI/Anthropic/Gemini): system prompt, user text, gambar (sha byte yang sama), tools,
  tool_choice, max_tokens, stop sequences, JSON mode; jawaban diterjemahkan balik ke wire
  client, JSON maupun SSE, dengan usage dan stop reason yang dipetakan. Streaming
  diterjemahkan secara buffered (stream upstream dibaca utuh lalu di-emit ulang dalam urutan
  event wire client) sehingga failover tetap mungkin sebelum apa pun dikirim ke client.
- `LICENSE` — MIT (dari starter).

### Routing (quality 60%)

Seluruh manifest (20 request) direncanakan di awal: request bobot-2 ditempatkan dulu,
masing-masing ke model gratis berperingkat tertinggi yang memenuhi requirements-nya,
dengan **cap per provider = budget floor trial** (calls_min dari env tersegel: groq 9,
cerebras 5, gemini 6, zai 4, openrouter 10, sambanova 4, pollinations 10, cloudflare 7,
ovh 5, kilol 12). Request JSON-mode OpenAI tidak pernah di-route ke provider wire
Anthropic (wire itu tidak bisa membawa `response_format`). Stream OpenAI selalu meminta
`stream_options.include_usage` supaya jawaban stream membawa token usage.

### Failover (10%)

Provider yang menolak (402/429/5xx) dihitung lalu request langsung dicoba ulang ke model
gratis terbaik berikutnya sesuai rating; provider yang sudah 2× menolak atau sudah di cap
dilewati dulu dan baru dicoba sebagai jalan terakhir. Semua call provider sekuensial,
jadi tidak pernah ada concurrent call ke satu provider.

### Determinisme

Tanpa wall clock / randomness tak berseed; semua timestamp yang di-emit fixed. Replay
cocok byte-for-byte di semua trial (lihat skor di bawah).

## Skor grade-local (`--driver process`, Node v22.23.5)

Docker tidak tersedia di mesin ini, jadi grade-local dijalankan dengan `--driver process`:

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls | Working | Replay |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|--------:|--------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 28 | 1 | matched |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 28 | 1 | matched |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 30 | 1 | matched |
| **mean** | **99** | **99** | **100** | **100** | **100** | **100** | **28,7** | **1** | **matched** |

Catatan quality 99: itu plafon praktis di trial ini — hanya openrouter yang punya model
rating 1475 (`z-ai/glm-5.3-flash:free`) dengan budget 10–12 call, sementara 17 request
membutuhkan rating 1475; quality 100 (semua request di rating terbaik) tidak mungkin
tercapai. Semua komponen lain 100 di ketiga trial.

Perintah grade:

```sh
cd ~/workspace/makefun/free-model-aggregator-01-kit/free-model-aggregator-01-kit
~/workspace/makefun/node22/node-v22.23.5-linux-x64/bin/node cli/makefun.mjs grade-local \
  --kit . --bot ~/workspace/makefun/aggregator --driver process --instances 3
```

Hasil grade tersimpan di `~/workspace/makefun/aggregator-grade/final/` (dan run sebelumnya
`all3`, `all3b`, `all3c`).

## Cara menjalankan

```sh
# Di luar trial (tanpa MAKEFUN_PROMPT_FILE / MAKEFUN_ENV_URL):
~/workspace/makefun/node22/node-v22.23.5-linux-x64/bin/node main.ts --list
# -> {"providers":[]} dan exit 0

# Di dalam trial (env mengatur MAKEFUN_PROMPT_FILE, MAKEFUN_ENV_URL, MAKEFUN_TRIAL_SEED):
~/workspace/makefun/node22/node-v22.23.5-linux-x64/bin/node main.ts
```

## Yang masih tersisa (BELUM dilakukan — jangan submit dari sini)

1. **Repo GitHub baru** (publik, dibuat setelah 2026-11-14T02:30:00Z, lisensi MIT) —
   salin isi workdir ini ke repo, commit pertama berisi `.makefun/owner.json`
   `{"wallet": "<wallet akun make.fun>"}` (atau owner repo = login GitHub akun).
2. **Bundle transkrip** sesi yang membangun entry ini:
   `node cli/makefun.mjs bundle --repo <repo> --out transcripts.tar.gz --muse-usage <file>`
   (sesi Muse app butuh usage rows dari muse.db).
3. **Check**: `node cli/makefun.mjs check --kit . --repo <repo> --transcripts transcripts.tar.gz`
   sampai semua lolos.
4. **Submit** (ESCROWED — wajib `--keypair`, keypair akun di
   `~/workspace/makefun/makefun-keypair.json`):
   `node cli/makefun.mjs submit --competition free-model-aggregator-01 --repo-url <url> --commit <sha> --title "Free AI model aggregator" --bundle transcripts.tar.gz --keypair ~/workspace/makefun/makefun-keypair.json`
5. Verifikasi: `node cli/makefun.mjs status --submission <id>`.
