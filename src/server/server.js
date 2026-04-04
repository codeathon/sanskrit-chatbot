#!/usr/bin/env node
/**
 * server.js - Express server for the Sanskrit/Yoga Knowledge Chatbot
 * Loads pre-built knowledge base from disk; uses Anthropic Claude only (no OpenAI path).
 * Env: ANTHROPIC_API_KEY (required), ANTHROPIC_MODEL (optional, default claude-sonnet-4-6).
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3456;
// Claude only (same defaults as src/server/server.py).
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
const ANTHROPIC_MODEL = process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-6';

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../client'))); // sibling: src/client

// Load knowledge base from disk (repo root)
const KB_PATH = path.join(__dirname, '../../knowledge-base/knowledge_base.json');
let knowledgeBase = null;

function loadKnowledgeBase() {
  if (!fs.existsSync(KB_PATH)) {
    console.error('❌ Knowledge base not found! Please run: node src/server/train.js');
    process.exit(1);
  }
  knowledgeBase = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
  console.log(`✅ Knowledge base loaded: ${knowledgeBase.allChunks.length} chunks from ${knowledgeBase.totalDocuments} documents`);
  console.log(`   Sources: ${knowledgeBase.sources.map(s => s.title).join(', ')}`);
}

// Retrieve relevant chunks using keyword search + scoring
function retrieveRelevantChunks(query, topK = 8) {
  const queryWords = query.toLowerCase()
    .replace(/[^a-zA-Z\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2);

  const scores = new Map();

  queryWords.forEach(word => {
    // Direct keyword match
    if (knowledgeBase.keywordIndex[word]) {
      knowledgeBase.keywordIndex[word].forEach(idx => {
        scores.set(idx, (scores.get(idx) || 0) + 2);
      });
    }
    // Partial keyword match
    Object.keys(knowledgeBase.keywordIndex).forEach(kw => {
      if (kw.includes(word) || word.includes(kw)) {
        knowledgeBase.keywordIndex[kw].forEach(idx => {
          scores.set(idx, (scores.get(idx) || 0) + 1);
        });
      }
    });
  });

  // Sort by score and return top chunks
  const ranked = [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([idx, score]) => ({ ...knowledgeBase.allChunks[idx], score }));

  return ranked;
}

// Format context for the LLM
function buildContext(chunks) {
  const bySource = {};
  chunks.forEach(chunk => {
    if (!bySource[chunk.source]) bySource[chunk.source] = [];
    bySource[chunk.source].push(chunk.text);
  });

  return Object.entries(bySource)
    .map(([src, texts]) => `[Source: ${src}]\n${texts.join('\n')}`)
    .join('\n\n');
}

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  const { message, history = [] } = req.body;

  if (!message || !message.trim()) {
    return res.status(400).json({ error: 'Message is required' });
  }

  if (!ANTHROPIC_API_KEY) {
    return res.status(500).json({ error: 'ANTHROPIC_API_KEY is not set' });
  }

  try {
    // Retrieve relevant chunks
    const relevantChunks = retrieveRelevantChunks(message);
    const hasRelevantContext = relevantChunks.length > 0 && relevantChunks[0].score >= 2;
    const context = hasRelevantContext ? buildContext(relevantChunks) : '';
    const sources = hasRelevantContext
      ? [...new Set(relevantChunks.map(c => c.source))].join(', ')
      : '';

    // Build system prompt
    const systemPrompt = `You are a scholarly assistant specializing in Sanskrit, Yoga philosophy, and Vedantic texts. You have access to a knowledge base containing specific documents about these topics.

KNOWLEDGE BASE CONTEXT:
${context || 'No relevant content found in the knowledge base for this query.'}

SOURCES CONSULTED: ${sources || 'None'}

CRITICAL RULES:
1. You MUST ONLY answer questions that are directly addressed in the knowledge base context provided above.
2. If the context is empty or does not contain information relevant to the query, respond with: "This question is outside the scope of my knowledge base. I can only answer questions about Yoga Sutras, Sanskrit grammar and vocabulary, and Vedanta philosophy based on my trained documents."
3. Do NOT use your general training knowledge to answer questions outside the provided context.
4. You MAY respond in Sanskrit when quoting terms or giving their Sanskrit names, but keep explanations in English unless the user requests Sanskrit.
5. Always cite which source document your answer comes from.
6. When giving Sanskrit terms, include: the Devanagari script if known, transliteration, and English meaning.

THINKING PROCESS (always include this):
Before your answer, briefly show your reasoning in <thinking> tags: which sources you found relevant, what key concepts are involved, and whether the question is within scope.`;

    // Build messages array (include history for context)
    const messages = [];
    
    // Add history
    history.slice(-6).forEach(msg => {
      messages.push({ role: msg.role, content: msg.content });
    });
    
    // Add current message
    messages.push({ role: 'user', content: message });

    // Call Claude API (streaming)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'anthropic-beta': 'interleaved-thinking-2025-05-14'
      },
      body: JSON.stringify({
        model: ANTHROPIC_MODEL,
        max_tokens: 16000,
        thinking: {
          type: 'enabled',
          budget_tokens: 10000
        },
        system: systemPrompt,
        messages
      })
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Claude API error: ${err}`);
    }

    // Stream the response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            // Forward the event to client
            res.write(`data: ${JSON.stringify({
              type: parsed.type,
              index: parsed.index,
              delta: parsed.delta,
              content_block: parsed.content_block,
              sources: sources || null,
              inScope: hasRelevantContext
            })}\n\n`);
          } catch (e) {
            // Skip malformed JSON
          }
        }
      }
    }

    res.write('data: [DONE]\n\n');
    res.end();

  } catch (error) {
    console.error('Chat error:', error);
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
    } else {
      res.write(`data: ${JSON.stringify({ type: 'error', message: error.message })}\n\n`);
      res.end();
    }
  }
});

// Knowledge base info endpoint
app.get('/api/kb-info', (req, res) => {
  if (!knowledgeBase) return res.status(500).json({ error: 'KB not loaded' });
  res.json({
    builtAt: knowledgeBase.builtAt,
    sources: knowledgeBase.sources,
    totalChunks: knowledgeBase.allChunks.length
  });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', kbLoaded: !!knowledgeBase });
});

loadKnowledgeBase();

app.listen(PORT, () => {
  console.log(`\n🚀 Sanskrit Chatbot Server (Claude) at: http://localhost:${PORT}`);
  console.log(`   Model: ${ANTHROPIC_MODEL}`);
  console.log('   Open this URL in your browser to start chatting!\n');
});
