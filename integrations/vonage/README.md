# Vonage Voice Memory Companion

Call a phone number and talk to an agent that **remembers you** — across the call,
across calls, and forever.

Built at the London Voice Agent HackNight. The successor to the
[BimpeAI integration](../bimpeai/): where that design could only poll a cloud agent's
conversation from outside (so memory commands were spoken aloud on voice and lookups
took 10–20s), here **we own the call** — Vonage does ASR/TTS at its edge and our server
runs the agent, so memory is silent and a normal chat turn answers in a couple of
seconds.

```
caller ──phone── Vonage (ASR + TTS, NCCO)
                    │  transcripts / NCCO JSON over one public webhook URL
                    ▼
              server.py (FastAPI)
                    │  Runner.run per caller turn          .state/conversations/
                    ▼                                       <number>.jsonl
        Voice Memory Companion agent ──────────────────► persistent chat thread
        (openai-agents + LiteLLM, DeepInfra Gemma)
                    │  ask_memory("plain English") tool
                    ▼
        BrainDB /api/v1/agent/query  (localhost only — never exposed)
```

## Two memory layers

| | mechanism | survives |
|---|---|---|
| the conversation | append-only JSONL per caller number, reloaded each turn/call | hangups, redials, server restarts |
| long-term memory | the agent's `ask_memory` tool → BrainDB's internal agent | everything; shared with every other BrainDB client |

The conversational agent is **multiturn** (never starts blank); BrainDB's internal
agent stays one-shot and untouched — it is the memory *subagent*, driven in plain
English, the same way Claude uses BrainDB.

## Run it

1. **BrainDB stack** (isolated instance, repo root):

   ```
   docker compose -f docker-compose.yml -f docker-compose.bimpeai.local.yml \
     --env-file .env --env-file .env.bimpeai.local -p braindb_bimpeai up -d --no-build
   ```

   Verify: `curl http://localhost:24700/health` → `{"status":"ok",...}`.

2. **Python env + config** (from the repo root):

   ```
   python -m venv integrations/vonage/.venv
   integrations/vonage/.venv/Scripts/python -m pip install "openai-agents[litellm]==0.17.2" fastapi uvicorn requests
   cp integrations/vonage/.env.example integrations/vonage/.env   # then set DEEPINFRA_API_KEY
   ```

3. **Server** (from `integrations/vonage/`):

   ```
   .venv/Scripts/python -m uvicorn server:app --port 24710
   ```

4. **Tunnel** (Vonage must reach the webhooks; BrainDB stays on loopback):

   ```
   cloudflared tunnel --url http://localhost:24710
   ```

   Copy the printed `https://<something>.trycloudflare.com` URL.
   (Fallback: `ngrok http 24710` — its localhost:4040 inspector is handy for
   replaying webhook payloads when debugging.)

5. **Vonage dashboard** ([dashboard.nexmo.com](https://dashboard.nexmo.com)):
   - Applications → Create application → enable **Voice**:
     - Answer URL: `https://<tunnel>/webhooks/answer` (GET)
     - Event URL: `https://<tunnel>/webhooks/event` (POST)
   - Numbers → Buy numbers → rent one (credits cover it) → link it to the application.

6. **Call the number.**

## Demo script

1. Call. Say: *"Remember that I drive a Mitsubishi Colt."* → "Got it, saving that…"
   → a few seconds → confirmation.
2. Hang up. Call again — it greets you as a returning caller (the chat thread
   persisted). Ask: *"What car do I drive?"* → "Let me check my memory, one moment."
   → "You drive a Mitsubishi Colt."
3. Restart `server.py` between calls to show the conversation survives that too.
4. Cross-channel bonus: start the [BimpeAI bridge](../bimpeai/) and ask the WhatsApp
   companion the same question — same BrainDB, same memories, different channel.

## Notes

- **The 5-second rule.** Vonage webhooks must answer within ~5s, but a BrainDB memory
  turn takes 5–30s. Turns therefore run as background tasks: plain chat is answered
  within a 3s in-webhook fast path; memory turns get a short spoken filler and each
  following `input` timeout acts as a poll tick until the answer is ready.
- **Plumbing test without any LLM**: set `ECHO_MODE=true` in `.env` — the server just
  speaks your transcript back. Use this to verify tunnel + dashboard + ASR first;
  it also surfaces the raw webhook payloads in the server log.
- **Tunnel URL changed?** Only re-paste the dashboard Answer/Event URLs — the NCCO
  `eventUrl` is derived per-request from the Host header.
- **Latency expectations**: chat turns ~1–3s; memory turns ~10–30s with fillers.
  Single user, no auth — keep BrainDB and this server on loopback; only the tunnel
  is public.
- If number rental is unavailable, an outbound test call works too: `POST
  https://api.nexmo.com/v1/calls` with this app's JWT and `answer_url` pointing at
  the tunnel (needs the application private key; see Vonage docs).
