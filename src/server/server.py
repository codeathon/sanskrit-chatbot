"""
VedaGPT Server
A chatbot server grounded in Sri Aurobindo's Complete Works (Vol 14, 15, 16)
and the complete Rig Veda database.

LLM: default Anthropic (Claude). OpenAI-compatible option:
  LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python3 src/server/server.py
  Optional: OPENAI_MODEL=gpt-4o-mini OPENAI_BASE_URL=https://api.openai.com/v1
KB-only (skip LLM): VEDAGPT_KB_ONLY=1
On API failure (e.g. low credits), chat falls back to the same retrieval text automatically.
"""

import os
import json
import pickle
import re
import math
import signal
from datetime import datetime, timezone
import urllib.request
import urllib.parse
import urllib.error
import http.server
import threading
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ = .../src/server/server.py → repo root is two levels up.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
KB_DIR   = os.path.join(BASE_DIR, 'knowledge-base')
INDEX_PATH = os.path.join(KB_DIR, 'search_index.pkl')
KB_PATH    = os.path.join(KB_DIR, 'knowledge_base.json')

# Set VEDAGPT_CHAT_LOG=0 to silence [VedaGPT chat] lines on stdout.
CHAT_LOG_STDOUT = os.environ.get('VEDAGPT_CHAT_LOG', '1').strip() not in ('0', 'false', 'no')
# If set, never call the LLM; chat returns formatted search hits only.
KB_ONLY_MODE = os.environ.get('VEDAGPT_KB_ONLY', '').strip().lower() in ('1', 'true', 'yes')


def chat_log(msg):
    """Print chat debug lines to the terminal (stderr keeps HTTP log suppression clear)."""
    if CHAT_LOG_STDOUT:
        print(msg, flush=True)

# ── Load index once at startup ─────────────────────────────────────────────────
print("Loading search index from disk...")
with open(INDEX_PATH, 'rb') as f:
    INDEX = pickle.load(f)

ALL_DOCS    = INDEX['all_docs']
TFIDF_VECS  = INDEX['tfidf_vecs']
IDF         = INDEX['idf']

print(f"Loaded {len(ALL_DOCS)} documents into memory.")

# ── Retrieval ──────────────────────────────────────────────────────────────────
STOPWORDS = {
    'the','a','an','and','or','of','in','to','is','are','was','were','it','this','that',
    'with','for','on','at','by','from','as','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','not','but','we',
    'they','he','she','i','you','his','her','their','our','its','which','who','what',
    'when','where','how','all','also','more','into','than','them','these','those',
    'so','if','about','up','out','no','can','my','your','there','then','only','after',
    'me','us','him','just','like','very','some','any','each','both','between','such',
    'through','during','before','above','below','few','while','because','although'
}

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Zāīūṛḷṃḥśṣṭḍṇñṅ]+\b', text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def normalize_msv_key_string(key):
    """Normalize '1.01.2' or composite_key to '1.1.2' for lookup."""
    if not key or not isinstance(key, str):
        return None
    parts = key.strip().split('.')
    if len(parts) != 3:
        return None
    try:
        return f'{int(parts[0])}.{int(parts[1])}.{int(parts[2])}'
    except ValueError:
        return key.strip()


def extract_msv_keys_from_query(query):
    """
    Find Rig-Veda-style references M.S.V (mandala.sukta.verse), e.g. 'Translate Rig Veda 1.1.2'.
    Returns normalized keys like ['1.1.2'] in order of first occurrence.
    """
    if not query:
        return []
    keys = []
    for m in re.finditer(r'(\d+)\.(\d+)\.(\d+)', query):
        try:
            keys.append(f'{int(m.group(1))}.{int(m.group(2))}.{int(m.group(3))}')
        except ValueError:
            continue
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def doc_msv_key(doc):
    """Stable M.S.V string for a mantra doc (composite_key or mandala/sukta/verse)."""
    if doc.get('type') != 'mantra':
        return None
    md = doc.get('mantra_data') or {}
    ck = md.get('composite_key')
    if ck is not None and str(ck).strip():
        return normalize_msv_key_string(str(ck))
    m, s, v = md.get('mandala'), md.get('sukta'), md.get('verse')
    if m is None or s is None or v is None:
        return None
    try:
        return (
            f'{int(float(str(m).strip()))}.'
            f'{int(float(str(s).strip()))}.'
            f'{int(float(str(v).strip()))}'
        )
    except ValueError:
        return normalize_msv_key_string(f'{m}.{s}.{v}')


# Score for exact M.S.V match — well above TF–IDF magnitudes.
_RV_KEY_HIT_SCORE = 1e9
# If any hit exceeds this, retrieval is restricted to those hits only (no weaker TF–IDF filler).
_STRONG_HIT_SCORE_MIN = 100.0


def lookup_mantras_by_msv_keys(keys):
    """Return mantra docs whose composite / B.C.D key matches any requested key."""
    if not keys:
        return []
    want = set(keys)
    out = []
    seen_ids = set()
    for doc in ALL_DOCS:
        if doc.get('type') != 'mantra':
            continue
        k = doc_msv_key(doc)
        if not k or k not in want:
            continue
        did = doc.get('id')
        if did in seen_ids:
            continue
        seen_ids.add(did)
        out.append((_RV_KEY_HIT_SCORE, doc))
    return out


def search(query, top_k=8):
    """
    Hybrid retrieval: exact Rig Veda M.S.V keys (e.g. 1.1.2) first, then TF–IDF.
    If any hit has score > _STRONG_HIT_SCORE_MIN (exact key match), only those hits are returned.
    """
    key_hits = lookup_mantras_by_msv_keys(extract_msv_keys_from_query(query))

    q_tokens = tokenize(query)
    q_vec = {}
    for tok in q_tokens:
        if tok in IDF:
            q_vec[tok] = IDF[tok]

    tfidf_results = []
    if q_vec:
        scores = []
        for i, vec in enumerate(TFIDF_VECS):
            score = sum(q_vec.get(tok, 0) * vec.get(tok, 0) for tok in q_vec)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        tfidf_results = [(s, ALL_DOCS[i]) for s, i in scores[:top_k]]

    if not key_hits and not tfidf_results:
        return []

    seen = set()
    out = []
    for score, doc in key_hits + tfidf_results:
        did = doc.get('id')
        if did in seen:
            continue
        seen.add(did)
        out.append((score, doc))
        if len(out) >= top_k:
            break

    # Above 100 (e.g. exact Rig Veda key match): return only those hits, ignore lower scores.
    if out and any(s > _STRONG_HIT_SCORE_MIN for s, _ in out):
        out = [(s, d) for s, d in out if s > _STRONG_HIT_SCORE_MIN]
    return out[:top_k]

# ── LLM backends (switch with LLM_PROVIDER) ───────────────────────────────────
# anthropic — default (Claude). openai — Chat Completions or OpenAI-compatible URL.
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'anthropic').strip().lower()

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
ANTHROPIC_MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-6').strip()
ANTHROPIC_VERSION = os.environ.get('ANTHROPIC_VERSION', '2023-06-01').strip()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini').strip()
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')

SYSTEM_PROMPT = """You are VedaGPT, a scholarly assistant specializing exclusively in:
1. Sri Aurobindo's Complete Works - Volume 14 (Vedic and Philological Studies), Volume 15 (The Secret of the Veda), and Volume 16 (Hymns to the Mystic Fire / Agni hymns)
2. The complete Rig Veda database with Sanskrit mantras, PadaPatha, Devanagari, and transliterations

Your knowledge is STRICTLY LIMITED to these sources. You must:
- ONLY answer questions about the Veda, Vedic hymns, Sri Aurobindo's interpretations, Sanskrit mantras, Vedic deities (Agni, Indra, Varuna, Mitra, Saraswati, Vayu, Ashvins, etc.), Vedic philosophy, philology, and related topics covered in these volumes
- REFUSE politely any questions outside this scope (current events, general knowledge, coding, math, etc.)
- You may respond in ENGLISH or SANSKRIT as appropriate, or mix both when discussing Sanskrit terms and mantras
- When discussing mantras, include the Devanagari script, transliteration, and Sri Aurobindo's interpretive insights
- Always cite which Volume or source your information comes from
- Show your reasoning process: explain how you interpret the question, what relevant context you found, and how you arrive at your answer

IMPORTANT: If retrieved context does not contain relevant information for the query, say so honestly rather than fabricating information.

THINKING FORMAT: Structure your response as:
🔍 **Understanding the Query**: [briefly explain what is being asked]
📚 **Relevant Context Found**: [mention which sources/volumes/mantras are relevant]  
💡 **Answer**: [your actual response, possibly in English and/or Sanskrit]
📖 **Sources**: [cite volume numbers and relevant sections]"""


def sanitize_messages_for_anthropic(history, final_user_content):
    """
    Anthropic returns 400 for empty content, invalid roles, or non-alternating turns.
    Merge consecutive same-role turns; drop empties; ensure the sequence starts with user.
    """
    raw = []
    for h in history[-12:]:
        if not isinstance(h, dict):
            continue
        role = h.get('role')
        if role not in ('user', 'assistant'):
            continue
        c = h.get('content', '')
        if not isinstance(c, str):
            c = '' if c is None else str(c)
        c = c.strip()
        if not c:
            continue
        raw.append({'role': role, 'content': c})
    merged = []
    for m in raw:
        if merged and merged[-1]['role'] == m['role']:
            merged[-1]['content'] += '\n\n' + m['content']
        else:
            merged.append(dict(m))
    while merged and merged[0]['role'] != 'user':
        merged.pop(0)
    # Drop trailing user turns (e.g. failed prior request with no assistant reply, or
    # duplicate user): the RAG prompt is the single authoritative user message for this call.
    while merged and merged[-1]['role'] == 'user':
        merged.pop()
    merged.append({'role': 'user', 'content': final_user_content})
    return merged


def anthropic_http_error_detail(exc):
    """Extract JSON error.message from Anthropic 4xx/5xx bodies for clearer logs and UI."""
    body = exc.read().decode('utf-8', errors='replace')
    try:
        parsed = json.loads(body)
        err = parsed.get('error') or {}
        msg = err.get('message') or body
        return f'{exc.code} {exc.reason}: {msg}'
    except json.JSONDecodeError:
        return f'{exc.code} {exc.reason}: {body or "(empty body)"}'


def call_anthropic_api(messages, stream=False):
    """Call Anthropic API with streaming support."""
    if not ANTHROPIC_API_KEY:
        raise ValueError('ANTHROPIC_API_KEY not set')

    payload = {
        'model': ANTHROPIC_MODEL,
        'max_tokens': 2048,
        'system': SYSTEM_PROMPT,
        'messages': messages,
        'stream': stream,
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': ANTHROPIC_VERSION,
        },
        method='POST',
    )
    try:
        return urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        detail = anthropic_http_error_detail(e)
        chat_log(f'[VedaGPT chat] Anthropic API HTTPError: {detail}')
        raise ValueError(f'Anthropic API: {detail}') from e


def openai_http_error_detail(exc):
    body = exc.read().decode('utf-8', errors='replace')
    try:
        parsed = json.loads(body)
        err = parsed.get('error') or {}
        msg = err.get('message') if isinstance(err, dict) else str(err)
        if not msg:
            msg = body
        return f'{exc.code} {exc.reason}: {msg}'
    except json.JSONDecodeError:
        return f'{exc.code} {exc.reason}: {body or "(empty body)"}'


def call_openai_api(openai_messages, stream=False):
    """OpenAI Chat Completions or compatible servers (same request/response shape)."""
    if not OPENAI_API_KEY:
        raise ValueError('OPENAI_API_KEY not set (required when LLM_PROVIDER=openai)')
    url = f'{OPENAI_BASE_URL}/chat/completions'
    payload = {
        'model': OPENAI_MODEL,
        'messages': openai_messages,
        'max_tokens': 2048,
        'stream': stream,
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}',
        },
        method='POST',
    )
    try:
        return urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        detail = openai_http_error_detail(e)
        chat_log(f'[VedaGPT chat] OpenAI-compatible API HTTPError: {detail}')
        raise ValueError(f'OpenAI API: {detail}') from e


def iter_anthropic_stream_text(response):
    """Yield text fragments from an Anthropic streaming HTTP response."""
    for line in response:
        line = line.decode('utf-8').strip()
        if not line.startswith('data: '):
            continue
        data_str = line[6:]
        if data_str == '[DONE]':
            break
        try:
            evt = json.loads(data_str)
            if evt.get('type') == 'content_block_delta':
                delta = evt.get('delta', {})
                if delta.get('type') == 'text_delta':
                    text = delta.get('text', '')
                    if text:
                        yield text
        except json.JSONDecodeError:
            pass


def iter_openai_stream_text(response):
    """Yield text fragments from an OpenAI-style SSE stream."""
    for line in response:
        line = line.decode('utf-8').strip()
        if not line.startswith('data: '):
            continue
        data_str = line[6:]
        if data_str == '[DONE]':
            break
        try:
            chunk = json.loads(data_str)
            if chunk.get('error'):
                err = chunk['error']
                msg = err.get('message', str(err)) if isinstance(err, dict) else str(err)
                raise ValueError(msg)
            choices = chunk.get('choices') or []
            if not choices:
                continue
            delta = choices[0].get('delta') or {}
            text = delta.get('content')
            if text:
                yield text
        except json.JSONDecodeError:
            pass


def mantra_ref_label(md):
    """Prefer B.C.D composite_key from ingest; else mandala.sukta.verse (legacy rows)."""
    if not isinstance(md, dict):
        return ''
    ck = (md.get('composite_key') or '').strip()
    if ck:
        return ck
    parts = [str(md.get('mandala', '') or ''), str(md.get('sukta', '') or ''), str(md.get('verse', '') or '')]
    return '.'.join(p for p in parts if p)


def build_rag_prompt(query, results):
    """Build a RAG prompt from retrieved results"""
    context_parts = []
    
    for score, doc in results:
        source = doc['source'].replace('_', ' ')
        if doc['type'] == 'mantra':
            md = doc.get('mantra_data', {})
            ctx = (f"[Rig Veda {mantra_ref_label(md)}] "
                   f"Devanagari: {md.get('mantra_devanagari','')} | "
                   f"Transliteration: {md.get('transliteration','')} | "
                   f"Devata: {md.get('devata','')} | Chhandas: {md.get('chhandas','')}")
            if md.get('page_vol15'):
                ctx += f" | Sri Aurobindo Vol.15 p.{md['page_vol15']}"
            if md.get('page_vol16'):
                ctx += f" | Vol.16 p.{md['page_vol16']}"
            if md.get('page_vol14'):
                ctx += f" | Vol.14 p.{md['page_vol14']}"
        else:
            ctx = f"[{source} | relevance: {score:.2f}]\n{doc['text'][:800]}"
        context_parts.append(ctx)
    
    context = "\n\n---\n\n".join(context_parts)

    translate_hint = ''
    if 'translat' in query.lower() and any(
        isinstance(d, dict) and d.get('type') == 'mantra' for _, d in results
    ):
        translate_hint = (
            ' When the user asks to translate or render the mantra, use the **Transliteration** '
            '(and Devanagari) from the Rig Veda rows in the context.'
        )

    return f"""Based on the following retrieved passages from Sri Aurobindo's Complete Works (Volumes 14, 15, 16) and the Rig Veda database, please answer the query.

RETRIEVED CONTEXT:
{context}

USER QUERY: {query}

Please answer using ONLY the information in the retrieved context above. If the context doesn't contain enough information, say so clearly.{translate_hint}"""


def write_sse_text_chunks(wfile, text, reply_parts, chunk_size=400):
    """Emit text as type:text SSE events; accumulate into reply_parts for logging."""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        reply_parts.append(chunk)
        out = json.dumps({'type': 'text', 'text': chunk}, ensure_ascii=False)
        wfile.write(f'data: {out}\n\n'.encode('utf-8'))
        wfile.flush()


def format_kb_only_reply(query, results, api_error=None, kb_only_preface=False):
    """
    Readable fallback when the LLM is off or the API failed (credits, network, etc.).
    """
    lines = []
    if kb_only_preface:
        lines.append(
            '**Knowledge base only** — VEDAGPT_KB_ONLY is set; the language model is not called.\n'
        )
    elif api_error is not None:
        brief = str(api_error)
        if len(brief) > 300:
            brief = brief[:300] + '…'
        lines.append(
            '**Model unavailable** (quota, credits, network, or configuration). '
            'Below are retrieval results from the indexed knowledge base only.\n'
        )
        lines.append(f'_Diagnostic:_ {brief}\n')
    lines.append('')
    if not results:
        lines.append(
            'No passages matched your query. Try different keywords (English or transliterated Sanskrit).'
        )
        return '\n'.join(lines)
    lines.append(f'Top matches for: _{query}_\n')
    for idx, (score, doc) in enumerate(results[:8], 1):
        src = doc['source'].replace('_', ' ')
        if doc['type'] == 'mantra':
            md = doc.get('mantra_data', {})
            dev = md.get('mantra_devanagari', '') or ''
            lines.append(
                f'### {idx}. Rig Veda {mantra_ref_label(md)} '
                f'(score {score:.3f})'
            )
            lines.append(f'- **Devata:** {md.get("devata", "")}')
            if dev:
                tail = '…' if len(dev) > 220 else ''
                lines.append(f'- **Devanagari:** {dev[:220]}{tail}')
            tr = (md.get('transliteration') or '').strip()
            if tr:
                ttail = '…' if len(tr) > 220 else ''
                lines.append(f'- **Transliteration:** {tr[:220]}{ttail}')
        else:
            body = doc.get('text') or ''
            preview = body[:1200]
            ell = '…' if len(body) > 1200 else ''
            lines.append(f'### {idx}. {src} (score {score:.3f})')
            lines.append('')
            lines.append(preview + ell)
        lines.append('')
    return '\n'.join(lines).strip() + '\n'


# ── HTTP Handler ───────────────────────────────────────────────────────────────
class VedaGPTHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        pass  # Suppress default logs
    
    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/':
            self.serve_file(os.path.join(BASE_DIR, 'src', 'client', 'index.html'), 'text/html')
        elif self.path == '/health':
            self.send_json({'status': 'ok', 'docs': len(ALL_DOCS)})
        elif self.path == '/api/info':
            with open(KB_PATH, 'r') as f:
                kb = json.load(f)
            self.send_json({'metadata': kb['metadata']})
        elif self.path == '/api/kb-info':
            self.handle_kb_info()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/chat':
            self.handle_chat()
        elif self.path == '/api/search':
            self.handle_search()
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_kb_info(self):
        """JSON shape expected by src/client/index.html (Knowledge Base sidebar)."""
        try:
            with open(KB_PATH, 'r', encoding='utf-8') as f:
                kb = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.send_json({'error': str(e)}, 500)
            return
        meta = kb.get('metadata', {})
        sources_out = []
        for s in meta.get('sources', []):
            if s.get('mantras') is not None:
                sources_out.append({
                    'type': 'xlsx',
                    'title': s.get('description', s.get('name', 'Rig Veda')),
                    'chunkCount': s['mantras'],
                })
            else:
                sources_out.append({
                    'type': 'docx',
                    'title': s.get('description', s.get('name', '')),
                    'chunkCount': s.get('chunks', 0),
                })
        total = meta.get('total_text_chunks', 0) + meta.get('total_rv_mantras', 0)
        mtime = os.path.getmtime(KB_PATH)
        built_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        self.send_json({
            'sources': sources_out,
            'totalChunks': total,
            'builtAt': built_at,
        })

    def read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(length).decode('utf-8'))
    
    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_cors_headers()
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)
    
    def serve_file(self, path, content_type):
        try:
            with open(path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
    
    def handle_search(self):
        body = self.read_body()
        query = body.get('query', '')
        results = search(query, top_k=5)
        out = []
        for score, doc in results:
            out.append({
                'score': round(score, 3),
                'source': doc['source'],
                'type': doc['type'],
                'preview': doc['text'][:200]
            })
        self.send_json({'results': out})
    
    def handle_chat(self):
        body = self.read_body()
        query   = body.get('message', '')
        history = body.get('history', [])
        stream  = body.get('stream', True)
        
        if not query.strip():
            self.send_json({'error': 'Empty query'}, 400)
            return
        
        # Retrieve context
        results = search(query, top_k=8)
        rag_prompt = build_rag_prompt(query, results)
        
        # Build message list (validated for Anthropic: no empty turns, user-first, alternating)
        messages = sanitize_messages_for_anthropic(history, rag_prompt)
        
        # Build retrieved sources info
        sources_info = []
        for score, doc in results[:5]:
            if doc['type'] == 'mantra':
                md = doc.get('mantra_data', {})
                sources_info.append({
                    'type': 'mantra',
                    'label': f"RV {mantra_ref_label(md)}",
                    'score': round(score, 3),
                    'devata': md.get('devata', ''),
                    'text': md.get('mantra_devanagari', '')[:80]
                })
            else:
                sources_info.append({
                    'type': 'text',
                    'label': doc['source'].replace('_', ' '),
                    'score': round(score, 3),
                    'text': doc['text'][:100]
                })
        
        if stream:
            # SSE streaming response
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_cors_headers()
            self.end_headers()

            q_preview = query if len(query) <= 240 else query[:240] + '…'
            chat_log(f'[VedaGPT chat] query ({len(query)} chars): {q_preview}')

            # First send sources
            sources_event = f"data: {json.dumps({'type': 'sources', 'sources': sources_info})}\n\n"
            self.wfile.write(sources_event.encode('utf-8'))
            self.wfile.flush()

            reply_parts = []

            try:
                if KB_ONLY_MODE:
                    fb = format_kb_only_reply(
                        query, results, api_error=None, kb_only_preface=True
                    )
                    write_sse_text_chunks(self.wfile, fb, reply_parts)
                elif LLM_PROVIDER == 'openai':
                    oa_msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}] + messages
                    response = call_openai_api(oa_msgs, stream=True)
                    for text in iter_openai_stream_text(response):
                        reply_parts.append(text)
                        out = json.dumps({'type': 'text', 'text': text}, ensure_ascii=False)
                        self.wfile.write(f"data: {out}\n\n".encode('utf-8'))
                        self.wfile.flush()
                else:
                    response = call_anthropic_api(messages, stream=True)
                    for text in iter_anthropic_stream_text(response):
                        reply_parts.append(text)
                        out = json.dumps({'type': 'text', 'text': text}, ensure_ascii=False)
                        self.wfile.write(f"data: {out}\n\n".encode('utf-8'))
                        self.wfile.flush()
            except Exception as e:
                chat_log(f'[VedaGPT chat] ERROR: {e}')
                fb = format_kb_only_reply(
                    query, results, api_error=e, kb_only_preface=False
                )
                write_sse_text_chunks(self.wfile, fb, reply_parts)

            full_reply = ''.join(reply_parts)
            preview = full_reply if len(full_reply) <= 1200 else full_reply[:1200] + '…'
            if KB_ONLY_MODE:
                chat_log(f'[VedaGPT chat] KB-only reply ({len(full_reply)} chars)')
            else:
                chat_log(
                    f'[VedaGPT chat] reply ({len(full_reply)} chars): '
                    f'{preview if preview else "(empty — check API key, model, or stream format)"}'
                )

            done_out = json.dumps({'type': 'done'})
            self.wfile.write(f"data: {done_out}\n\n".encode('utf-8'))
            self.wfile.flush()
        else:
            # Non-streaming fallback
            try:
                if KB_ONLY_MODE:
                    text = format_kb_only_reply(
                        query, results, api_error=None, kb_only_preface=True
                    )
                elif LLM_PROVIDER == 'openai':
                    oa_msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}] + messages
                    resp = call_openai_api(oa_msgs, stream=False)
                    result = json.loads(resp.read().decode('utf-8'))
                    text = result['choices'][0]['message']['content']
                else:
                    response = call_anthropic_api(messages, stream=False)
                    result = json.loads(response.read().decode('utf-8'))
                    text = result['content'][0]['text']
                self.send_json({'response': text, 'sources': sources_info})
            except Exception as e:
                text = format_kb_only_reply(
                    query, results, api_error=e, kb_only_preface=False
                )
                self.send_json({'response': text, 'sources': sources_info, 'fallback': True})


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8000))
    server = HTTPServer(('0.0.0.0', PORT), VedaGPTHandler)
    if KB_ONLY_MODE:
        _prov_line = 'KB-only (no LLM)'
    elif LLM_PROVIDER == 'openai':
        _prov_line = f'openai · {OPENAI_MODEL}'
    else:
        _prov_line = f'anthropic · {ANTHROPIC_MODEL}'
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              VedaGPT Server                              ║
╠══════════════════════════════════════════════════════════╣
║  URL:       http://localhost:{PORT}                         ║
║  LLM:       {_prov_line:<44} ║
║  Docs:      {len(ALL_DOCS)} indexed documents                    ║
║  Sources:   Vol.14, Vol.15, Vol.16 + RV Database          ║
╚══════════════════════════════════════════════════════════╝
    """)
    if KB_ONLY_MODE:
        print('  VEDAGPT_KB_ONLY=1 — chat never calls the LLM; retrieval text only.\n')
    elif LLM_PROVIDER == 'openai':
        print(f'  OPENAI_BASE_URL={OPENAI_BASE_URL}')
        if not OPENAI_API_KEY:
            print('  WARNING: OPENAI_API_KEY is empty — chat will fail until set.\n')
    elif not ANTHROPIC_API_KEY:
        print('  WARNING: ANTHROPIC_API_KEY is empty — chat will fail until set.\n')

    # Ctrl+C (SIGINT) / SIGTERM: stop serve_forever and release the socket cleanly.
    def _graceful_shutdown():
        print('\n  Shutting down…', flush=True)
        server.shutdown()

    def _on_stop_signal(signum, frame):
        threading.Thread(target=_graceful_shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _on_stop_signal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _on_stop_signal)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        print('  Server stopped.', flush=True)
