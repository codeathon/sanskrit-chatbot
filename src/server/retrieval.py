"""
Retrieval and KB-only reply formatting (no HTTP, no LLM).
Imported by server.py after the search index is loaded.
"""

import re

STOPWORDS = {
	'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'is', 'are', 'was', 'were', 'it', 'this', 'that',
	'with', 'for', 'on', 'at', 'by', 'from', 'as', 'be', 'been', 'being', 'have', 'has', 'had',
	'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'not', 'but', 'we',
	'they', 'he', 'she', 'i', 'you', 'his', 'her', 'their', 'our', 'its', 'which', 'who', 'what',
	'when', 'where', 'how', 'all', 'also', 'more', 'into', 'than', 'them', 'these', 'those',
	'so', 'if', 'about', 'up', 'out', 'no', 'can', 'my', 'your', 'there', 'then', 'only', 'after',
	'me', 'us', 'him', 'just', 'like', 'very', 'some', 'any', 'each', 'both', 'between', 'such',
	'through', 'during', 'before', 'above', 'below', 'few', 'while', 'because', 'although',
}

# Exact M.S.V match score — above TF–IDF magnitudes.
_RV_KEY_HIT_SCORE = 1e9
_STRONG_HIT_SCORE_MIN = 100.0


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


def lookup_mantras_by_msv_keys(keys, all_docs):
	"""Return mantra docs whose composite / B.C.D key matches any requested key."""
	if not keys:
		return []
	want = set(keys)
	out = []
	seen_ids = set()
	for doc in all_docs:
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


def search(query, top_k, all_docs, tfidf_vecs, idf):
	"""
	Hybrid retrieval: exact Rig Veda M.S.V keys first, then TF–IDF.
	If any hit has score > _STRONG_HIT_SCORE_MIN, only those hits are returned.
	"""
	key_hits = lookup_mantras_by_msv_keys(extract_msv_keys_from_query(query), all_docs)

	q_tokens = tokenize(query)
	q_vec = {}
	for tok in q_tokens:
		if tok in idf:
			q_vec[tok] = idf[tok]

	tfidf_results = []
	if q_vec:
		scores = []
		for i, vec in enumerate(tfidf_vecs):
			score = sum(q_vec.get(tok, 0) * vec.get(tok, 0) for tok in q_vec)
			if score > 0:
				scores.append((score, i))
		scores.sort(reverse=True)
		tfidf_results = [(s, all_docs[i]) for s, i in scores[:top_k]]

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

	if out and any(s > _STRONG_HIT_SCORE_MIN for s, _ in out):
		out = [(s, d) for s, d in out if s > _STRONG_HIT_SCORE_MIN]
	return out[:top_k]


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
	"""Build a RAG prompt from retrieved results."""
	context_parts = []

	for score, doc in results:
		source = doc['source'].replace('_', ' ')
		if doc['type'] == 'mantra':
			md = doc.get('mantra_data', {})
			ctx = (f"[Rig Veda {mantra_ref_label(md)}] "
				   f"Devanagari: {md.get('mantra_devanagari', '')} | "
				   f"Transliteration: {md.get('transliteration', '')} | "
				   f"Devata: {md.get('devata', '')} | Chhandas: {md.get('chhandas', '')}")
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
