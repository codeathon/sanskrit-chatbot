#!/usr/bin/env python3
"""
VedaGPT — Startup Script
Serves both the Flask API and the client from a single process.
"""
import os
import sys

# Ensure we're in the right directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'server'))

# Check for API key
if not os.environ.get('ANTHROPIC_API_KEY'):
    print("\n⚠️  WARNING: ANTHROPIC_API_KEY is not set!")
    print("   Set it with: export ANTHROPIC_API_KEY=your_key_here\n")

# Check knowledge base exists
kb_path = os.path.join(BASE_DIR, 'knowledge-base', 'search_index.pkl')
if not os.path.exists(kb_path):
    print("❌ Knowledge base not found. Run the ingestion script first:")
    print("   python3 ingest.py")
    sys.exit(1)

print("Starting VedaGPT...")
os.chdir(BASE_DIR)

# Import and run server
import server
