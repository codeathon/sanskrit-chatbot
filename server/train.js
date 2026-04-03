#!/usr/bin/env node
/**
 * train.js - Reads all data files and builds a knowledge base index stored to disk.
 * This simulates "training" on static documents by extracting, chunking,
 * and storing structured knowledge that the server will load at runtime.
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '../data');
const KB_PATH = path.join(__dirname, '../knowledge-base/knowledge_base.json');

console.log('📚 Starting knowledge base construction...\n');

function chunkText(text, source, sourceType) {
  // Split long content into overlapping chunks for better retrieval
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks = [];
  const CHUNK_SIZE = 3;
  
  for (let i = 0; i < sentences.length; i += Math.max(1, CHUNK_SIZE - 1)) {
    const chunk = sentences.slice(i, i + CHUNK_SIZE).join(' ').trim();
    if (chunk.length > 20) {
      chunks.push({
        id: `${source}_chunk_${i}`,
        text: chunk,
        source,
        sourceType,
        keywords: extractKeywords(chunk)
      });
    }
  }
  return chunks;
}

function extractKeywords(text) {
  // Extract meaningful Sanskrit and English terms
  const stopWords = new Set(['the','a','an','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should','may','might',
    'shall','can','of','in','on','at','to','for','with','by','from','as','or','and',
    'but','if','that','this','these','those','it','its','they','them','their','which',
    'who','what','when','where','how','all','each','every','both','few','more','most']);
  
  const words = text.toLowerCase()
    .replace(/[^a-zA-Z\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 3 && !stopWords.has(w));
  
  return [...new Set(words)];
}

function processDocx(filepath) {
  const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
  console.log(`  ✓ Processing ${data.filename}: "${data.title}"`);
  
  const allChunks = [];
  data.content.forEach(paragraph => {
    const chunks = chunkText(paragraph, data.filename, 'docx');
    allChunks.push(...chunks);
  });
  
  return {
    source: data.filename,
    title: data.title,
    type: 'docx',
    totalParagraphs: data.content.length,
    chunks: allChunks,
    rawContent: data.content
  };
}

function processXlsx(filepath) {
  const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
  console.log(`  ✓ Processing ${data.filename}: "${data.title}"`);
  
  const allChunks = [];
  
  data.sheets.forEach(sheet => {
    const colNames = sheet.columns;
    sheet.rows.forEach(row => {
      // Build a natural language description of each row
      const entry = {};
      colNames.forEach((col, i) => { entry[col] = row[i]; });
      
      const text = `${entry['Sanskrit Term'] || ''} (${entry['Transliteration'] || ''}) means "${entry['English Meaning'] || ''}". Category: ${entry['Category'] || ''}. ${entry['Usage Context'] || ''}`;
      
      const chunks = chunkText(text, data.filename, 'xlsx');
      allChunks.push(...chunks);
    });
  });
  
  return {
    source: data.filename,
    title: data.title,
    type: 'xlsx',
    totalRows: data.sheets.reduce((acc, s) => acc + s.rows.length, 0),
    chunks: allChunks,
    rawSheets: data.sheets
  };
}

// Build knowledge base
const knowledgeBase = {
  version: '1.0',
  builtAt: new Date().toISOString(),
  sources: [],
  allChunks: [],
  totalDocuments: 0
};

const dataFiles = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.json'));

dataFiles.forEach(file => {
  const filepath = path.join(DATA_DIR, file);
  let doc;
  
  if (file.includes('.docx.')) {
    doc = processDocx(filepath);
  } else if (file.includes('.xlsx.')) {
    doc = processXlsx(filepath);
  }
  
  if (doc) {
    knowledgeBase.sources.push({
      source: doc.source,
      title: doc.title,
      type: doc.type,
      chunkCount: doc.chunks.length
    });
    knowledgeBase.allChunks.push(...doc.chunks);
    knowledgeBase.totalDocuments++;
  }
});

// Build an inverted keyword index for fast retrieval
const keywordIndex = {};
knowledgeBase.allChunks.forEach((chunk, idx) => {
  chunk.keywords.forEach(kw => {
    if (!keywordIndex[kw]) keywordIndex[kw] = [];
    keywordIndex[kw].push(idx);
  });
});

knowledgeBase.keywordIndex = keywordIndex;

// Ensure output dir exists
fs.mkdirSync(path.dirname(KB_PATH), { recursive: true });
fs.writeFileSync(KB_PATH, JSON.stringify(knowledgeBase, null, 2));

console.log('\n✅ Knowledge base built successfully!');
console.log(`   📄 Documents processed: ${knowledgeBase.totalDocuments}`);
console.log(`   🔍 Total chunks indexed: ${knowledgeBase.allChunks.length}`);
console.log(`   🗝️  Unique keywords: ${Object.keys(keywordIndex).length}`);
console.log(`   💾 Saved to: ${KB_PATH}`);
console.log('\nRun `node server/server.js` to start the chatbot server.\n');
