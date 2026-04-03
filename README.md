# VedaGPT (Sanskrit chatbot)

A **web chat assistant** that answers from a **fixed knowledge base** built from Sri Aurobindo’s Vedic works (Complete Works volumes used as source documents) and a **Rig Veda–style spreadsheet** when present—not from the open web. Questions are matched to relevant passages with **TF‑IDF retrieval**, and those passages are sent to a **language model** so replies stay tied to your indexed material. If the model API is unavailable, the app can still return **raw search hits** from the same index.

**In short:** ingest documents from `data/`, search them at query time, and chat in the browser with answers grounded in that corpus.
