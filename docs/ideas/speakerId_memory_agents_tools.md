**One-paragraph TL;DR**
Maestro will grow from a single-user, memory-less voice assistant into a fully modular **speaker-aware agent platform** by wiring three battle-tested OSS services—**Diglett** for real-time speaker verification, **A-MEM** for dynamic long-term memory, and **Letta** for agent orchestration & tool execution—behind optional Compose profiles.  Each component is container-first, FastAPI-based, and already exposes REST/WebSocket APIs, so the upgrade boils down to: add Compose files, stream Diglett’s `speaker_id` through the request lifecycle, swap in A-MEM’s Python manager (or side-car), and proxy complex turns to Letta.  Users stay KISS—one flag per feature—and the code stays DRY—one memory interface, one agent proxy.  Latency target: ≤650 ms p95.([github.com][1], [github.com][2], [github.com][3], [docs.letta.com][4], [docs.letta.com][5])

---

## 1  Background

* **Maestro today** is a <500 ms E2E voice agent that streams STT to Ollama and back, but ships with `MEMORY_ENABLED=false` and no agent loop.([github.com][1])
* **A-MEM** introduces graph-like, Zettelkasten-style notes over Chroma & Redis for scalable recall and evolution.([github.com][3], [arxiv.org][6])
* **Diglett** adds a 128-D speaker embedding per user and a live `/stream` WS that tags each audio chunk.([github.com][2], [speechbrain.readthedocs.io][7])
* **Letta** provides a one-liner Docker image exposing `/v1/chat/completions`, function-calling via OpenAI JSON schema, and local tool sandboxes.([docs.letta.com][4], [docs.letta.com][5], [letta.com][8])

---

## 2  Goals

| #  | Goal                             | Success metric                           |
| -- | -------------------------------- | ---------------------------------------- |
| G1 | **Isolate memories by speaker**  | 0 cross-speaker retrievals in tests      |
| G2 | **First-class long-term memory** | CRUD ops ⩽50 ms median                   |
| G3 | **Agent + tool loop**            | 100 % of sample tasks resolved via Letta |
| G4 | **One-command opt-in**           | `make dev-full` brings up all services   |

---

## 3  Component Deep-Dive

### 3.1  Diglett (speaker layer)

* REST `/embed` → returns speaker embedding from 5 s sample; WS `/stream` → pushes `{speaker_id, start, end}` frames in real time.([github.com][2])
* Powered by **SpeechBrain EncoderClassifier** (TDNN x-vector).([speechbrain.readthedocs.io][7])
* Stateless and scales horizontally; ideal as a Docker side-car (port 3210).

### 3.2  A-MEM (memory layer)

* Python package **`agentic_memory`** or stand-alone API; stores notes in Chroma with Redis cache.([github.com][3], [trychroma.com][9])
* Auto-links notes, evolves context, and supports arbitrary `user_id`—we’ll pass `speaker_id`.([arxiv.org][6])
* Ships MIT-licensed; no DB migrations needed.

### 3.3  Letta (agent/tool layer)

* One-line Docker run maps PG data and exposes port 8283.([docs.letta.com][4])
* Accepts multiple LLM back-ends (`OPENAI_API_KEY`, `OLLAMA_BASE_URL` etc.) and can mount a repo for local tool exec via `TOOL_EXEC_DIR` + `TOOL_EXEC_VENV_NAME`.([docs.letta.com][4], [docs.letta.com][5], [docs.letta.com][10])
* Agents and tools conform to the OpenAI function-calling JSON schema, enabling cross-framework reuse.([letta.com][8])

---

## 4  Proposed Architecture

```
Browser ─┬─► Orchestrator (FastAPI)
         │       ├─► Diglett  (ws://diglett:3210/stream)  # speaker_id
         │       ├─► A-MEM    (HTTP /retrieve,/store)     # long-term notes
         │       └─► Letta    (POST /v1/chat/completions) # agent + tools
         └─► Ollama  (LLM inference)
```

* `speaker_id` threads every call → key for A-MEM collection and Letta agent.
* Flags: `SPEAKER_ID_ENABLED`, `MEMORY_MODE=off|sidecar|inproc`, `AGENTS_ENABLED`.

---

## 5  Milestones

| Sprint | Deliverable                                                               |
| ------ | ------------------------------------------------------------------------- |
| **M0** | Import `diglett_client.py`; cache embeddings.                             |
| **M1** | Strategy pattern for `MemoryStore` (HTTP vs in-proc).                     |
| **M2** | Compose files: `docker-compose.speaker.yml`, `docker-compose.agents.yml`. |
| **M3** | FastAPI `/agents/chat` proxy with timeouts & tracing.                     |
| **M4** | E2E tests: dual-speaker scenario, memory isolation, tool call round-trip. |

---

## 6  Acceptance Criteria

* `make dev-memory` → memories persist across sessions.
* `make dev-full` → two concurrent speakers see isolated timelines and Letta resolves a weather tool call.
* 95-th percentile latency ≤ 650 ms on CUDA / MPS rigs.

---

## 7  Risks & Mitigations

| Risk                     | Mitigation                                      |
| ------------------------ | ----------------------------------------------- |
| Diglett adds \~40 ms RTT | Batch audio frames; run on same Docker network. |
| Memory bloat in Chroma   | TTL pruning job; configurable `MAX_NOTES`.      |
| Tool sandbox security    | Require signed tools; mount read-only volumes.  |

---

## 8  References

1. Diglett README – real-time verification & VAD([github.com][2])
2. A-MEM repo – dynamic Zettelkasten notes([github.com][3])
3. A-MEM paper (Feb 2025) – design & benchmarks([arxiv.org][6])
4. Letta self-hosting guide – Docker run flags([docs.letta.com][4])
5. Letta local tool execution – `TOOL_EXEC_DIR` / `TOOL_EXEC_VENV_NAME`([docs.letta.com][5])
6. Letta pgAdmin docs – Postgres port exposure([docs.letta.com][10])
7. Letta blog – JSON schema compatibility across agents([letta.com][8])
8. SpeechBrain API – `EncoderClassifier` for speaker verification([speechbrain.readthedocs.io][7])
9. Chroma site – vector DB backbone for memory([trychroma.com][9])
10. Medium post – vector DBs as memory for agents([medium.com][11])
11. Pyannote embedding model card – x-vector style features([huggingface.co][12])
12. Maestro README – current no-memory architecture([github.com][1])
13. Maestro `.env.example` – placeholder for future flags([github.com][13])
14. Open-WebUI thread – using `host.docker.internal` for Ollama base URL([github.com][14])
15. SpeechBrain docs – ready-to-use speaker recognition pipeline([speechbrain.readthedocs.io][15])

---

> **Next step:** thumbs-up to convert this RFC into tracked issues & PRs. Let’s ship a **speaker-aware, memory-savvy, tool-calling Maestro**—simple toggles, zero duplication, maximum power.

[1]: https://github.com/dusterbloom/maestro "GitHub - dusterbloom/maestro"
[2]: https://github.com/8igMac/diglett "GitHub - 8igMac/diglett: Real-time speaker verification for long conversations."
[3]: https://github.com/agiresearch/A-mem "GitHub - agiresearch/A-mem: A-MEM: Agentic Memory for LLM Agents"
[4]: https://docs.letta.com/guides/selfhosting "Self-hosting Letta | Letta"
[5]: https://docs.letta.com/guides/tool-execution/local "Local tool execution | Letta"
[6]: https://arxiv.org/abs/2502.12110?utm_source=chatgpt.com "A-MEM: Agentic Memory for LLM Agents"
[7]: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.speaker.html?utm_source=chatgpt.com "speechbrain.inference.speaker module"
[8]: https://www.letta.com/blog/ai-agents-stack "The AI agents stack  | Letta"
[9]: https://www.trychroma.com/ "Chroma"
[10]: https://docs.letta.com/guides/selfhosting/pgadmin "Inspecting your database | Letta"
[11]: https://medium.com/sopmac-ai/vector-databases-as-memory-for-your-ai-agents-986288530443?utm_source=chatgpt.com "Vector Databases as Memory for your AI Agents - Medium"
[12]: https://huggingface.co/pyannote/embedding "pyannote/embedding · Hugging Face"
[13]: https://github.com/dusterbloom/maestro/blob/main/.env.example "maestro/.env.example at main · dusterbloom/maestro · GitHub"
[14]: https://github.com/open-webui/open-webui/discussions/1685?utm_source=chatgpt.com "I can't save the Ollama Base URL :( #1685 - GitHub"
[15]: https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-classification-from-scratch.html?utm_source=chatgpt.com "Speech Classification From Scratch"
