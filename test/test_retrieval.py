"""
Unit tests for server-side retrieval and KB-only formatting (no LLM / HTTP).
Run from repo root: python3 -m unittest discover -s test -v
"""

import os
import sys
import unittest

# Import retrieval from src/server (same layout as running server.py from that directory).
_SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'server'))
if _SERVER_DIR not in sys.path:
	sys.path.insert(0, _SERVER_DIR)

import retrieval  # noqa: E402


class TestNormalizeMsvKey(unittest.TestCase):
	def test_normalizes_leading_zeros(self):
		self.assertEqual(retrieval.normalize_msv_key_string('1.01.002'), '1.1.2')

	def test_invalid_returns_none_or_strip(self):
		self.assertIsNone(retrieval.normalize_msv_key_string(''))
		self.assertIsNone(retrieval.normalize_msv_key_string('1.2'))
		self.assertIsNone(retrieval.normalize_msv_key_string(None))


class TestExtractMsvKeys(unittest.TestCase):
	def test_extracts_and_dedupes_in_order(self):
		self.assertEqual(
			retrieval.extract_msv_keys_from_query('See 1.2.3 and 1.2.3 again 10.20.30'),
			['1.2.3', '10.20.30'],
		)

	def test_empty(self):
		self.assertEqual(retrieval.extract_msv_keys_from_query(''), [])
		self.assertEqual(retrieval.extract_msv_keys_from_query(None), [])


class TestDocMsvKey(unittest.TestCase):
	def _mantra(self, **kwargs):
		base = {'type': 'mantra', 'id': 'x', 'source': 'rv', 'mantra_data': {}}
		base['mantra_data'].update(kwargs)
		return base

	def test_composite_key(self):
		doc = self._mantra(composite_key='01.02.03')
		self.assertEqual(retrieval.doc_msv_key(doc), '1.2.3')

	def test_legacy_mandala_sukta_verse(self):
		doc = self._mantra(mandala='1', sukta='5', verse='7')
		self.assertEqual(retrieval.doc_msv_key(doc), '1.5.7')

	def test_non_mantra(self):
		doc = {'type': 'text', 'id': 't', 'source': 's', 'text': 'hi'}
		self.assertIsNone(retrieval.doc_msv_key(doc))


class TestMantraRefLabel(unittest.TestCase):
	def test_prefers_composite(self):
		md = {'composite_key': '1.1.1', 'mandala': '9'}
		self.assertEqual(retrieval.mantra_ref_label(md), '1.1.1')

	def test_legacy_join(self):
		self.assertEqual(
			retrieval.mantra_ref_label({'mandala': '2', 'sukta': '3', 'verse': '4'}),
			'2.3.4',
		)


class TestSearch(unittest.TestCase):
	def _mantra_doc(self, doc_id, key, **extra_md):
		md = {'composite_key': key, 'devata': 'Agni'}
		md.update(extra_md)
		return {
			'id': doc_id,
			'type': 'mantra',
			'source': 'rig_veda',
			'mantra_data': md,
		}

	def _text_doc(self, doc_id, text):
		return {'id': doc_id, 'type': 'note', 'source': 'vol_14', 'text': text}

	def test_tfidf_ranks_by_overlap(self):
		all_docs = [
			self._text_doc('a', 'alpha beta gamma'),
			self._text_doc('b', 'delta epsilon zeta'),
		]
		# Vectors aligned with docs; query matches first doc via shared token "gamma" / "alpha".
		idf = {'alpha': 1.0, 'gamma': 1.0, 'delta': 1.0}
		tfidf_vecs = [
			{'alpha': 2.0, 'gamma': 1.0},
			{'delta': 2.0},
		]
		out = retrieval.search('alpha gamma', top_k=4, all_docs=all_docs, tfidf_vecs=tfidf_vecs, idf=idf)
		self.assertTrue(out)
		self.assertEqual(out[0][1]['id'], 'a')

	def test_msv_key_beats_tfidf_and_filters_weak(self):
		m = self._mantra_doc('m1', '5.5.5')
		t = self._text_doc('t1', '5.5.5 extra words for tfidf overlap')
		all_docs = [m, t]
		# If query is literally "5.5.5", tokenize may not produce q_vec tokens (digits only).
		idf = {'extra': 1.0}
		tfidf_vecs = [{}, {'extra': 3.0}]
		out = retrieval.search('Rig Veda 5.5.5', top_k=8, all_docs=all_docs, tfidf_vecs=tfidf_vecs, idf=idf)
		ids = [d['id'] for _, d in out]
		self.assertIn('m1', ids)
		self.assertNotIn('t1', ids)

	def test_no_match_empty(self):
		self.assertEqual(
			retrieval.search('zzzunusedtoken', top_k=4, all_docs=[], tfidf_vecs=[], idf={}),
			[],
		)


class TestFormatKbOnlyReply(unittest.TestCase):
	def test_kb_only_preface(self):
		text = retrieval.format_kb_only_reply('q', [], kb_only_preface=True)
		self.assertIn('Knowledge base only', text)
		self.assertIn('VEDAGPT_KB_ONLY', text)

	def test_api_error_truncates(self):
		long_err = 'x' * 400
		text = retrieval.format_kb_only_reply('q', [], api_error=long_err)
		self.assertIn('Model unavailable', text)
		self.assertIn('x' * 300, text)
		self.assertNotIn('x' * 301, text)

	def test_mantra_section(self):
		results = [
			(
				99.0,
				{
					'id': '1',
					'type': 'mantra',
					'source': 'rig_veda',
					'mantra_data': {
						'composite_key': '1.1.1',
						'devata': 'Agni',
						'mantra_devanagari': 'ॐ',
						'transliteration': 'om',
					},
				},
			)
		]
		text = retrieval.format_kb_only_reply('hello', results)
		self.assertIn('Rig Veda 1.1.1', text)
		self.assertIn('99.000', text)
		self.assertIn('Transliteration', text)


class TestBuildRagPrompt(unittest.TestCase):
	def test_translate_hint_with_mantra(self):
		results = [
			(
				1.0,
				{
					'type': 'mantra',
					'source': 'rig_veda',
					'mantra_data': {'composite_key': '1.1.1'},
				},
			)
		]
		p = retrieval.build_rag_prompt('Please translate mantra 1.1.1', results)
		self.assertIn('Transliteration', p)

	def test_no_translate_hint_without_keyword(self):
		results = [
			(
				1.0,
				{
					'type': 'mantra',
					'source': 'rig_veda',
					'mantra_data': {'composite_key': '1.1.1'},
				},
			)
		]
		p = retrieval.build_rag_prompt('What is the devata', results)
		self.assertNotIn('**Transliteration**', p)


if __name__ == '__main__':
	unittest.main()
