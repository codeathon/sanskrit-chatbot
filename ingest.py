#!/usr/bin/env python3
"""
VedaGPT — Data Ingestion Script
Reads all source files, builds knowledge base and search index, writes to disk.
Run this once (or whenever source files change).

Usage:
    python3 ingest.py

Source files expected in ./data/:
    VOLUME_14.docx   — Sri Aurobindo Vol.14 Vedic & Philological Studies
    VOLUME_15.docx   — Sri Aurobindo Vol.15 The Secret of the Veda
    VOLUME_16.docx   — Sri Aurobindo Vol.16 Hymns to the Mystic Fire
    VedaGPT-Blueprint.xlsx — Rig Veda Database
"""

import os
import json
import re
import pickle
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from openpyxl import load_workbook

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
KB_DIR   = os.path.join(BASE_DIR, 'knowledge-base')
os.makedirs(KB_DIR, exist_ok=True)

STOPWORDS = {
    'the','a','an','and','or','of','in','to','is','are','was','were','it','this','that',
    'with','for','on','at','by','from','as','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','not','but','we',
    'they','he','she','i','you','his','her','their','our','its','which','who','what',
    'when','where','how','all','also','more','into','than','them','these','those',
    'so','if','about','up','out','no','can','my','your','there','then','only','after'
}

def _extract_docx_via_python_docx(path):
    # Pure-Python path when the pandoc binary is not installed (see extract_docx).
    from docx import Document
    doc = Document(path)
    parts = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(' '.join(cells))
    return '\n\n'.join(parts)

def extract_docx(path):
    text = None
    pandoc_bin = shutil.which('pandoc')
    if pandoc_bin:
        result = subprocess.run(
            [pandoc_bin, path, '-t', 'plain'], capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            text = result.stdout
    if not text:
        text = _extract_docx_via_python_docx(path)
    text = re.sub(r'\nPage \d+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text, source, chunk_size=1200, overlap=200):
    chunks = []
    words = text.split()
    i = 0
    cid = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append({'id': f"{source}_{cid}", 'source': source, 'text': chunk, 'type': 'text'})
        cid += 1
        i += chunk_size - overlap
    return chunks

def tokenize(text):
    tokens = re.findall(r'\b[a-zA-Zāīūṛḷṃḥśṣṭḍṇñṅ]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def build_tfidf(all_docs):
    print("  Tokenizing documents...")
    doc_tokens = [tokenize(d['text']) for d in all_docs]
    N = len(all_docs)
    df = defaultdict(int)
    for tokens in doc_tokens:
        for tok in set(tokens):
            df[tok] += 1
    idf = {tok: math.log(N / (1 + df[tok])) for tok in df}
    print(f"  Vocabulary: {len(idf)} terms")
    tfidf_vecs = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {tok: (cnt/total)*idf[tok] for tok, cnt in tf.items() if tok in idf}
        tfidf_vecs.append(vec)
    return tfidf_vecs, idf

def main():
    print("=" * 60)
    print("  VedaGPT — Data Ingestion")
    print("=" * 60)

    # ── Extract volumes ──────────────────────────────────────────
    volumes = [
        ('VOLUME_14.docx', 'Volume_14_Vedic_Philological_Studies',
         'Sri Aurobindo Vol.14 — Vedic & Philological Studies'),
        ('VOLUME_15.docx', 'Volume_15_Secret_of_the_Veda',
         'Sri Aurobindo Vol.15 — The Secret of the Veda'),
        ('VOLUME_16.docx', 'Volume_16_Hymns_to_Mystic_Fire',
         'Sri Aurobindo Vol.16 — Hymns to the Mystic Fire'),
    ]

    all_docs = []
    source_meta = []

    for filename, source_id, description in volumes:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️  Not found: {path}")
            continue
        print(f"\nExtracting {filename}...")
        text = extract_docx(path)
        chunks = chunk_text(text, source_id)
        all_docs.extend(chunks)
        source_meta.append({'name': source_id, 'description': description, 'chunks': len(chunks)})
        print(f"  → {len(chunks)} chunks, {len(text):,} chars")

    # ── Extract RV Database ──────────────────────────────────────
    xlsx_path = os.path.join(DATA_DIR, 'VedaGPT-Blueprint.xlsx')
    rv_mantras = []
    if os.path.exists(xlsx_path):
        print("\nExtracting Rig Veda database...")
        wb = load_workbook(xlsx_path, read_only=True)
        ws = wb['RV Database']
        rows = list(ws.iter_rows(values_only=True))
        for row in rows[1:]:
            if row[0] is None or not row[12]:
                continue
            mantra_devanagari = str(row[12])
            devata = str(row[9]) if row[9] else ''
            transliteration = str(row[14]) if row[14] else ''
            md = {
                'mantra_number': row[0], 'mandala': row[1],
                'sukta': row[2], 'verse': row[3],
                'devata': devata, 'chhandas': str(row[10]) if row[10] else '',
                'mantra_devanagari': mantra_devanagari,
                'pada_patha': str(row[13]) if row[13] else '',
                'transliteration': transliteration,
                'page_vol15': str(row[5]) if row[5] else '',
                'page_vol14': str(row[6]) if row[6] else '',
                'page_vol16': str(row[7]) if row[7] else '',
            }
            text = (f"Rig Veda {row[1]}.{row[2]}.{row[3]}: {mantra_devanagari} "
                    f"Devata: {devata} Transliteration: {transliteration}")
            doc = {'id': f"RV_{row[0]}", 'source': 'Rig_Veda_Database',
                   'type': 'mantra', 'text': text, 'mantra_data': md}
            rv_mantras.append(doc)
        all_docs.extend(rv_mantras)
        source_meta.append({'name': 'Rig_Veda_Database',
                            'description': 'Complete Rig Veda — Sanskrit, PadaPatha, Transliteration',
                            'mantras': len(rv_mantras)})
        print(f"  → {len(rv_mantras)} mantras")
    else:
        print(f"⚠️  Not found: {xlsx_path}")

    print(f"\nTotal documents: {len(all_docs)}")

    # ── Build TF-IDF index ───────────────────────────────────────
    print("\nBuilding TF-IDF search index...")
    tfidf_vecs, idf = build_tfidf(all_docs)

    # ── Save to disk ─────────────────────────────────────────────
    print("\nSaving knowledge base...")
    kb = {
        'metadata': {
            'sources': source_meta,
            'total_text_chunks': sum(s.get('chunks',0) for s in source_meta),
            'total_rv_mantras': len(rv_mantras)
        },
        'text_chunks': [d for d in all_docs if d['type'] == 'text'],
        'rv_mantras': [d['mantra_data'] for d in all_docs if d['type'] == 'mantra']
    }
    with open(os.path.join(KB_DIR, 'knowledge_base.json'), 'w', encoding='utf-8') as f:
        json.dump(kb, f, ensure_ascii=False)

    index = {'all_docs': all_docs, 'tfidf_vecs': tfidf_vecs, 'idf': idf}
    with open(os.path.join(KB_DIR, 'search_index.pkl'), 'wb') as f:
        pickle.dump(index, f)

    size = os.path.getsize(os.path.join(KB_DIR, 'search_index.pkl')) / (1024*1024)
    print(f"  search_index.pkl — {size:.1f} MB")
    print(f"  knowledge_base.json — saved")
    print("\n✅ Ingestion complete! Run: python3 server/server.py")

if __name__ == '__main__':
    main()
