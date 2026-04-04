# VedaGPT (Sanskrit chatbot)

A **web chat assistant** that answers from a **fixed knowledge base** built from Sri Aurobindo’s Vedic works (Complete Works volumes used as source documents) and a **Rig Veda–style spreadsheet** when present—not from the open web. Questions are matched to relevant passages with **TF‑IDF retrieval**, and those passages are sent to a **language model** so replies stay tied to your indexed material. If the model API is unavailable, the app can still return **raw search hits** from the same index.

**In short:** ingest documents from `data/`, search them at query time, and chat in the browser with answers grounded in that corpus.

**Layout:** the browser UI lives under **`src/client/`**; the Python and Node servers live under **`src/server/`** (`server.py`, `retrieval.py`, `server.js`, `train.js`). Run commands from the **project root** (where `ingest.py` is).

## Tests

Server-side **retrieval and KB-only formatting** (TF–IDF search, Rig Veda `M.S.V` key handling, `format_kb_only_reply`, RAG prompt building) live in **`src/server/retrieval.py`** and are covered by unit tests under **`test/`**. They do not call the Claude or OpenAI APIs.

From the project root:

```bash
python3 -m unittest discover -s test -v
```

## Adding data

1. Put files in the **`data/`** folder at the project root (same level as `ingest.py`). Only **files directly in `data/`** are scanned; subfolders are ignored.
2. Use these types only; anything else is skipped when you run ingest:
   - **`.docx`** — Any number of Word files, any filenames. Body text and tables are extracted and split into searchable chunks. Optional: install [Pandoc](https://pandoc.org/) for nicer plain-text extraction; otherwise the script uses `python-docx`.
   - **`.xlsx`** — Any number of Excel workbooks.
     - A sheet named **`RV Database`** is treated as the **Rig Veda blueprint**: rows use the original column layout (Devanagari in column 13 / index 12, etc.). Those rows become **mantra** records in the index.
     - **Other sheets** are turned into plain text (one row per line, cells joined with ` | `) and chunked like a document.
     - Tabs whose names should **not** be indexed (e.g. design notes) can be skipped with the environment variable **`VEDAGPT_SKIP_XLSX_SHEETS`**: comma-separated names, case-insensitive. If unset, the sheet **`Website Design`** is skipped by default. Set it to empty to skip nothing: `export VEDAGPT_SKIP_XLSX_SHEETS=`.

3. Do not commit secrets inside `data/` if the repo is public; `.docx` / `.xlsx` are ordinary files on disk.

## Refreshing the knowledge base

After you add, remove, or change files under **`data/`**, rebuild the index and JSON export:

```bash
# From the project root, with dependencies installed
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

python3 ingest.py
```

That overwrites **`knowledge-base/search_index.pkl`** and **`knowledge-base/knowledge_base.json`**.

Then **restart the server** so it reloads the index (it reads the pickle only at startup). Press **Ctrl+C** once to stop a running server, then start it again — see **Starting the server** below.

## Starting the server

**Prerequisite:** run **`python3 ingest.py`** at least once so **`knowledge-base/search_index.pkl`** exists.

From the **project root**:

```bash
source venv/bin/activate    # if you use the venv from above; Windows: venv\Scripts\activate
export ANTHROPIC_API_KEY='sk-ant-...'   # required for Claude unless using KB-only or OpenAI
python3 src/server/server.py
```

Open **http://localhost:8000/** in a browser (static UI + API on the same port).

| Environment variable | Purpose |
| --- | --- |
| `PORT` | Listen port (default **8000**). |
| `ANTHROPIC_API_KEY` | Claude API key (default provider is Anthropic). |
| `ANTHROPIC_MODEL` | Model id (default `claude-sonnet-4-6`). |
| `LLM_PROVIDER` | Set to **`openai`** to use Chat Completions instead of Claude. |
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | Used when `LLM_PROVIDER=openai` (base URL supports OpenAI-compatible hosts). |
| `VEDAGPT_KB_ONLY` | Set to **`1`** to never call an LLM; chat returns retrieval text only. |
| `VEDAGPT_CHAT_LOG` | Set to **`0`** to silence extra chat logs on the terminal. |

The server uses only the **Python standard library** for HTTP; no extra pip packages are required for **`src/server/server.py`** itself (ingest still needs **`requirements.txt`**).

**Stop:** **Ctrl+C** triggers a clean shutdown and releases the port.

**Alternate stack:** a Node/Express server lives under **`src/server/server.js`** (different port and knowledge-base shape); the flow above is the supported Python path.
