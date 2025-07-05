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




# DRAFT 2

**♻️ Epic Issue — “Single-Docker” Memory & Speaker Isolation for Maestro (Letta + fastRTC + Diglett)**
*Goal: abolish the A-MEM side-car and ship a one-container Letta instance that uses fastRTC’s `AgenticMemorySystem` internally, keyed by Diglett’s speaker UUIDs. When this issue closes, a `docker-compose -f dev-full.yml up -d` must run the entire voice stack with long-term, speaker-scoped memory and tool-calling agents in **one shot**—zero manual fixes.*

---

## 0 Quick Spec

```text
┌─ Maestro FE (mic) ─┐
│  WS audio stream   │
└────────┬───────────┘
         ▼
   Diglett (3210) ──► speaker_id
         ▼
  Letta (8283)
      ├─ fastRTC AgenticMemorySystem (in-proc)
      └─ custom tools arch_mem_insert / search → _same class_
         (TOOL_EXEC_DIR=/app/tools_fastmem)
```

* No more **a-mem** container.
* `speaker_id` threads through every tool call.
* fastRTC’s Chroma & Redis stay embedded, volumes mounted via Letta.

---

## 1 Background & References

| Component                         | Key facts                                                                                                                                                                                            | Sources                                                         |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **fastRTC memory**                | Exposes `set_user_id`, `get_user_context`, hybrid search & add\_note for per-user collections.                                                                                                       | ([raw.githubusercontent.com][1])                                |
| **Diglett**                       | `/embed` (REST, 5 s sample) and `/stream` (WS) return real-time speaker IDs using SpeechBrain ECAPA embeddings.                                                                                      | ([github.com][2])                                               |
| **Letta**                         | a) Built-in tools `archival_memory_insert/search` are *meant to be overridden*; b) Supports local tool code via `TOOL_EXEC_DIR` & `TOOL_EXEC_VENV_NAME`; c) Docker self-host with `OLLAMA_BASE_URL`. | ([docs.letta.com][3], [docs.letta.com][4], [docs.letta.com][5]) |
| **Chroma DB**                     | Python client is `pip install chromadb`; used by fastRTC retriever.                                                                                                                                  | ([pypi.org][6])                                                 |
| **SpeechBrain EncoderClassifier** | Ready-to-use speaker-rec pipeline powering Diglett.                                                                                                                                                  | ([speechbrain.readthedocs.io][7])                               |

---

## 2 Deliverables

1. **`tools_fastmem/` package** inside Letta image that wraps fastRTC’s `AgenticMemorySystem` in two tools:

   * `archival_memory_insert(content: str, speaker_id: str)`
   * `archival_memory_search(query: str, speaker_id: str) -> List[str]`
2. **Dockerfile patch** that `pip install -e ./tools_fastmem` during Letta build.
3. **`dev-full.yml` compose** with only:

   * `letta-fastmem` (builds from Dockerfile)
   * `diglett`
   * (optional) `ollama`
4. **Maestro orchestrator patch** to:

   * call Diglett once per user to obtain persistent `speaker_id`
   * attach that ID as extra arg when hitting Letta `/v1/chat/completions`.
5. **CI**: GitHub Action `voice_e2e.yml` running two WAV fixtures to prove isolated memory.

---

## 3 Detailed Implementation Tasks

### 3.1 Package fastRTC Memory

* Copy `backend/src/a_mem` & `backend/src/memory` into repo at `tools_fastmem/fastmem/`.
* Create `setup.cfg` (PEP 517/518) declaring deps: `chromadb>=0.4`, `redis>=5`, `rank_bm25`, `nltk`.
* Add entry-point `fastmem/__init__.py` exposing:

```python
from .memory_system import AgenticMemorySystem  # noqa: F401
mem = AgenticMemorySystem(user_id="placeholder")
```

### 3.2 Override Letta Tools

Inside `tools_fastmem/archival.py`:

```python
from letta_client.tool import BaseTool
from pydantic import BaseModel, Field
from fastmem import mem

class InsertArgs(BaseModel):
    content: str = Field(..., description="memory text")
    speaker_id: str = Field(...)

class FastMemInsert(BaseTool):
    name = "archival_memory_insert"              # overwrite built-in
    args_schema = InsertArgs
    description = "Store long-term memory via fastRTC."

    def run(self, content: str, speaker_id: str):
        mem.set_user_id(speaker_id)
        mem.add_note(content)
        return True
```

Replicate for `archival_memory_search`.

### 3.3 Dockerfile (`letta-fastmem/Dockerfile`)

```dockerfile
FROM letta/letta:latest
COPY tools_fastmem /app/tools_fastmem
RUN pip install -e /app/tools_fastmem
ENV TOOL_EXEC_DIR=/app/tools_fastmem
```

### 3.4 Compose (`dev-full.yml`)

```yaml
services:
  letta:
    build: .
    ports: ["8283:8283"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - TOOL_EXEC_DIR=/app/tools_fastmem
      - TOOL_EXEC_VENV_NAME=fastmem_env
  diglett:
    image: 8igmac/diglett:latest
    ports: ["3210:80"]
```

### 3.5 Maestro Bridge

* Add `DIGLETT_WS_URL` & `SPEAKER_ID` cache.
* On session start:

  1. POST PCM buffer to `diglett/api/embed` → `speaker_id`. ([github.com][2])
  2. Forward STT chunks to both Diglett `/stream` (for live verification) and Ollama.
* Attach `"speaker_id": <uuid>` to every `tool_call.arguments` in Letta.

### 3.6 Tests

| Test                         | Steps                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------- |
| **unit\_insert**             | call tool insert; assert `vector_count==1` in Chroma.                         |
| **dual\_speaker\_isolation** | speaker A inserts note “love pizza”; speaker B searches “pizza” → **no** hit. |
| **e2e\_voice**               | play two 10 s WAVs concurrently; expect Letta replies labelled correctly.     |

---

## 4 Acceptance Criteria

* `docker compose -f dev-full.yml up -d` returns **200 OK** on `http://localhost:8283/health` within 60 s.
* Memory is persisted at `~/.letta/.persist/pgdata` and scoped by `speaker_id` (integration test passes).
* 95-th percentile latency ≤ 650 ms for 30-turn mixed-speaker benchmark.
* No A-MEM container present; `docker ps` shows exactly **letta**, **diglett**, **ollama**.

---

## 5 Risk Matrix & Mitigations

| Risk                                       | Severity | Mitigation                                                                            |
| ------------------------------------------ | -------- | ------------------------------------------------------------------------------------- |
| fastRTC schema drift vs Letta expectations | ★★★      | Keep original tool names so ADE UI stays functional; expose only `text/plain` fields. |
| Redis unavailable in container             | ★★       | fastRTC already handles `MemoryRedisCache=None`.                                      |
| Diglett adds >50 ms RTT                    | ★★       | Deploy in same Docker network; set `SAMPLE_RATE=16000` env.                           |
| Chroma DB grows unbounded                  | ★★       | FastRTC `evo_threshold` already consolidates notes.                                   |

---

## 6 Out-of-Scope

* Upgrading fastRTC embedding model (nomic-embed-text)
* Multi-agent routing beyond speaker isolation
* Web frontend CSS/UX changes

---

## 7 How to Start (Agent-Friendly)

```bash
git clone <this-repo>
git checkout -b feature/letta-fastmem
docker compose -f dev-full.yml build         # build letta w/ fastmem
pytest tests                                 # all green
docker compose -f dev-full.yml up -d         # run stack
```

> **Reminder (LLM agent)**: Every modified file must be fully committed; run `ruff` linter & `pytest -q` before opening the PR.

---

### Helpful Resources

* Letta agent architectures & memory override docs ([docs.letta.com][3], [docs.letta.com][8])
* Letta local tool execution & `TOOL_EXEC_DIR` env ([docs.letta.com][4])
* Letta Docker self-hosting guide & `OLLAMA_BASE_URL` env ([docs.letta.com][5])
* Diglett README — REST `/embed`, WS `/stream` & ECAPA model ([github.com][2])
* fastRTC `AgenticMemorySystem` in-code API (set\_user\_id, add\_note, search) ([raw.githubusercontent.com][1])
* Chroma DB client for Python ([pypi.org][6])
* SpeechBrain EncoderClassifier API ([speechbrain.readthedocs.io][7])
* Example Letta tool import pitfalls (#2108) ([github.com][9])
* OpenAI function-calling JSON schema reference ([platform.openai.com][10])
* Letta issue about `OLLAMA_BASE_URL` env (#2388) for Linux/host networks ([github.com][11])

---

**Definition of Done**: Merged PR tagged `feat: letta-fastmem`, all CI checks green, manual voice demo in README screencast linked in release notes.

[1]: https://raw.githubusercontent.com/dusterbloom/fastRTC/refs/heads/backend-refactoring/backend/src/a_mem/memory_system.py "raw.githubusercontent.com"
[2]: https://github.com/8igMac/diglett "GitHub - 8igMac/diglett: Real-time speaker verification for long conversations."
[3]: https://docs.letta.com/guides/agents/architectures "Agent Architectures | Letta"
[4]: https://docs.letta.com/guides/tool-execution/local "Local tool execution | Letta"
[5]: https://docs.letta.com/guides/selfhosting "Self-hosting Letta | Letta"
[6]: https://pypi.org/project/chromadb/?utm_source=chatgpt.com "chromadb - PyPI"
[7]: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.speaker.html?utm_source=chatgpt.com "speechbrain.inference.speaker module"
[8]: https://docs.letta.com/guides/agents/memory "Agent Memory | Letta"
[9]: https://github.com/letta-ai/letta/issues/2108 "GitHub · Where software is built"
[10]: https://platform.openai.com/docs/guides/function-calling "OpenAI Platform"
[11]: https://github.com/letta-ai/letta/issues/2388?utm_source=chatgpt.com "Letta ignores the \"OLLAMA_BASE_URL\" env variable, and try to use ..."


Here’s the missing **Ollama + Embedding** section you can paste straight into the Epic (feel free to merge or reorder).
It spells out *exactly* how Letta discovers and uses both the chat-model **and** the embedding model that fastRTC’s memory code already expects, so your agent can ship a self-contained stack on the first PR.

---

## 🧩 Component ❸ — Ollama (LLM **and** Embeddings)

### 1. What Ollama Gives Us

* **OpenAI-compatible chat endpoint** (`/v1/chat/completions`) lets Letta talk to any local GGUF model simply by swapping the base URL.([ollama.com][1])
* **Embedding endpoints** (`/api/embeddings` *and* the newer `/api/embed`) serve vectors from models like `nomic-embed-text` or `mxbai-embed-large`; both are faster than Ada-002 and run fully offline.([ollama.com][2], [ollama.com][3])
* All calls work through the tiny `ollama` Python client that fastRTC already imports, so no extra SDK glue is needed.([raw.githubusercontent.com][4])

### 2. Fast Path to Working Bits

| Step                       | Shell / Code                                                                                                                        | Why it matters                                                                                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Pull models**            | `bash\nollama pull mistral:7b-q6_K          # chat LLM\nollama pull nomic-embed-text:latest  # embed model\n`                       | Tags prevent Ollama from defaulting to over-compressed Q4 builds.([docs.letta.com][5])                                                                       |
| **Expose Ollama to Letta** | `-e OLLAMA_BASE_URL=http://host.docker.internal:11434` (mac/Win) or `--network host` + `http://localhost:11434` (Linux)             | Letta auto-registers every chat + embedding model it sees on that endpoint.([docs.letta.com][5], [docs.letta.com][5])                                        |
| **Select models in Letta** | `python\nclient.agents.create(\n    model=\"ollama/your/mistral:7b-q6_K\",\n    embedding=\"ollama/nomic-embed-text:latest\",\n)\n` | One handle for chat, one for embeds—exactly what Letta’s SDK expects.([docs.letta.com][5])                                                                   |
| **FastRTC hook-up**        | `AgenticMemorySystem(model_name='nomic-embed-text:latest', llm_backend='ollama', llm_model='mistral:7b-q6_K')`                      | Matches the handles above; `ChromaRetriever` will call `ollama.embeddings()` under the hood.([raw.githubusercontent.com][6], [raw.githubusercontent.com][4]) |

### 3. Docker Compose Snippet (add/replace in `dev-full.yml`)

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama     # cache models
    ports: ["11434:11434"]
    environment:
      - OLLAMA_MODELS=/root/.ollama
  letta:
    build: .
    depends_on: [ollama]
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - TOOL_EXEC_DIR=/app/tools_fastmem
```

> **Why not rely on Letta’s built-in embeds?**
> Because fastRTC’s `OllamaEmbeddingFunction` already streams directly to `/api/embeddings`, giving you zero duplication and perfect parity with your memory system.([raw.githubusercontent.com][4])

### 4. Edge Cases & Fixes

| Issue                                                                | Quick Fix                                                                                                  | Ref                                |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Letta ignores `OLLAMA_BASE_URL` in some older builds                 | Use image `letta/letta:≥0.31.0` or apply patch from issue #2388                                            | ([github.com][7])                  |
| Embeddings endpoint mismatch (`/v1/embeddings` vs `/api/embeddings`) | FastRTC hits `/api/embeddings`; new models also expose `/api/embed`, so you’re safe.                       | ([github.com][8], [ollama.com][9]) |
| Need multilingual or code embeds                                     | Swap model handle to `ollama/jeffh/intfloat-multilingual-e5-large-instruct` etc.—endpoint stays identical. | ([ollama.com][10])                 |

### 5. Acceptance Tests (add to CI)

1. **Health check** – `curl -s http://ollama:11434 | grep '"status":"ok"'`.
2. **Embed round-trip** – Insert “I love pizza” as Speaker A, query “pizza” as Speaker B → expect *no* hit.
3. **Model switch** – Change Letta agent to `ollama/mistral:7b-q6_K` and rerun integration tests; latency ≤ 650 ms p95.

---

#### TL;DR

*Set `OLLAMA_BASE_URL`, pull one chat model + one embed model, and your entire Letta-fastRTC-Diglett stack speaks the same OpenAI-style language—no extra service, no foreign APIs, all local.*

[1]: https://ollama.com/blog/openai-compatibility "OpenAI compatibility · Ollama Blog"
[2]: https://ollama.com/library/nomic-embed-text "nomic-embed-text"
[3]: https://ollama.com/blog/embedding-models?utm_source=chatgpt.com "Embedding models · Ollama Blog"
[4]: https://raw.githubusercontent.com/dusterbloom/fastRTC/refs/heads/backend-refactoring/backend/src/a_mem/retrievers.py "raw.githubusercontent.com"
[5]: https://docs.letta.com/guides/server/providers/ollama "Ollama | Letta"
[6]: https://raw.githubusercontent.com/dusterbloom/fastRTC/refs/heads/backend-refactoring/backend/src/a_mem/memory_system.py "raw.githubusercontent.com"
[7]: https://github.com/letta-ai/letta/issues/2388?utm_source=chatgpt.com "Letta ignores the \"OLLAMA_BASE_URL\" env variable, and try to use ..."
[8]: https://github.com/ollama/ollama/issues/2416?utm_source=chatgpt.com "`/v1/embeddings` OpenAI compatible API endpoint · Issue #2416"
[9]: https://ollama.com/shaw/dmeta-embedding-zh%3Alatest?utm_source=chatgpt.com "shaw/dmeta-embedding-zh - Ollama"
[10]: https://ollama.com/jeffh/intfloat-multilingual-e5-large-instruct%3Af16?utm_source=chatgpt.com "jeffh/intfloat-multilingual-e5-large-instruct:f16 - Ollama"
