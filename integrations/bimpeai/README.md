# BrainDB memory for a BimpeAI agent

Give a [BimpeAI](https://bimpe.ai) cloud agent (WhatsApp / voice / phone) a long-term
memory backed by a self-hosted [BrainDB](https://github.com/dimknaf/braindb) — and
**the agent decides** when to save and when to recall. **No tunnel:** the bridge only
makes outbound calls.

This is the cloud-agent analogue of [`../hermes/braindb`](../hermes/braindb/) (which gives
the local Hermes agent a `braindb_ask` tool). BimpeAI runs in the cloud and can't call
into your laptop, so instead of a live tool the agent issues a **memory command in the
conversation** and a local **bridge** fulfils it.

## How it works

```
 you ─▶ BimpeAI agent (cloud) ──reply containing──▶  [[MEMORY: <save or recall>]]
                  ▲                                           │  (poll, outbound)
                  │  [[MEMORY_RESULT: <answer>]]  ◀──── bridge.py ──▶ BrainDB /agent/query
                  └───────────── agent answers the user ─────┘        (save OR recall)
```

- The agent's **Workflow prompt** tells it: to **save**, emit `[[MEMORY: remember that …]]`;
  to **recall**, emit `[[MEMORY: <question>]]`; tell the user it takes a moment and be patient.
- `bridge.py` polls the agent's conversations; on a `[[MEMORY: q]]` it calls BrainDB's own
  agent (`POST /api/v1/agent/query` — saves or recalls in plain English, same gateway the
  hermes integration uses), then posts `[[MEMORY_RESULT: …]]` back so the agent answers.
- The bridge **never ingests messages or decides anything** — the agent decides; the bridge
  only fulfils its commands.

## Files
| file | what |
|---|---|
| `setup_agent.py` | create the Workflow (memory prompt) + Agent; delete any stale KB (idempotent) |
| `bridge.py`      | the always-on local fulfiller (poll → fulfil `[[MEMORY]]` → post result) |
| `.env`           | `BIMPEAI_API_KEY`, `BIMPEAI_AGENT_ID`, `BRAINDB_BASE_URL` (gitignored) |

## Run it

1. **Have a BrainDB running** that the bridge can reach at `BRAINDB_BASE_URL`
   (here: the isolated `bimpeai` stack on `http://localhost:24700` — see the repo-root
   `docker-compose.bimpeai.local.yml` overlay). BrainDB has no auth — keep it on loopback.
2. **Python env + provision the agent**
   ```bash
   python -m venv integrations/bimpeai/.venv
   integrations/bimpeai/.venv/Scripts/python -m pip install bimpeai requests   # /bin/python on mac/linux
   cp integrations/bimpeai/.env.example integrations/bimpeai/.env              # set BIMPEAI_API_KEY
   integrations/bimpeai/.venv/Scripts/python integrations/bimpeai/setup_agent.py
   ```
   Prints the `agent_id` (written to `.env`) + a WhatsApp test link. The workflow appears in
   **BimpeAI Dashboard → Workflows → "BrainDB Memory Companion"**.
3. **Connect a channel** — message the test number with the `start <code>` shown, or in the
   dashboard connect your own WhatsApp / assign a UK number to the agent.
4. **Start the bridge** (leave it running):
   ```bash
   integrations/bimpeai/.venv/Scripts/python integrations/bimpeai/bridge.py
   ```

## Try it
- *"Remember I drive a Mitsubishi Colt."* → the agent saves it (and tells you it takes a sec).
- *"What car do I drive?"* → the agent checks BrainDB and answers. Memory persists across chats.

## Notes
- **Latency is expected.** A save/recall round-trips through BrainDB's LLM agent, so it can take
  several seconds; the prompt makes the agent tell the user to bear with it. Tune `BIMPEAI_POLL_SECONDS`.
- **No tunnel ⇒ the `[[MEMORY: …]]` command is part of the agent's message** (and is spoken on a
  voice call). It's the only signal a no-tunnel bridge can poll. A tunnelled native BimpeAI tool
  would hide it — out of scope by design here.
- Single user, no auth — co-locate the bridge and BrainDB.
