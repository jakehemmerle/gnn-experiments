---
stepsCompleted: [1, 2, 3]
status: complete
inputDocuments:
  - technical-llamaindex-rag-research-2025-12-31.md
date: 2025-12-31
author: Jake
project_name: llamaindex-rag
---

# Product Brief: LlamaIndex RAG System

## Executive Summary

A personal research paper RAG system that enables semantic search and synthesis across a growing collection of academic papers. Built as an MCP server for integration with Claude Code, the system provides citation-aware responses grounded in the source material. This is a learning-focused project exploring LlamaIndex's cloud capabilities for document ingestion, retrieval, and synthesis.

---

## Core Vision

### Problem Statement

Research papers accumulate faster than they can be organized. Notes scatter across tools. The knowledge contained in saved papers becomes inaccessible - you know you read something relevant, but finding it means manually searching through dozens of PDFs.

### Problem Impact

- Insights from past reading are lost or forgotten
- Cross-paper connections remain invisible
- Time wasted re-reading or searching for half-remembered findings
- No unified way to query a personal research collection

### Why Existing Solutions Fall Short

- **Reference managers** (Zotero, Mendeley): Organize metadata, don't enable semantic search across content
- **General RAG products** (ChatGPT uploads, Notion AI): Not designed for academic papers, no citation tracking, limited scale
- **Academic tools** (Elicit, Semantic Scholar): Search public corpora, not personal collections
- **None** offer MCP integration for seamless Claude Code workflow

### Proposed Solution

A LlamaIndex-powered RAG system exposed as an MCP server that:
- Ingests PDFs into a cloud-based vector store
- Supports incremental paper additions without full reprocessing
- Enables natural language queries with citation-backed responses
- Surfaces cross-citations and concept relationships within the collection
- Integrates directly into Claude Code via MCP protocol

### Key Differentiators

| Differentiator | Value |
|----------------|-------|
| **MCP-native** | Query your research library from Claude Code without context switching |
| **Citation-aware** | Every response traces back to specific source chunks |
| **Personal collection** | Your papers, your highlights, your knowledge base |
| **Learning-focused** | Clean LlamaIndex implementation for skill development |
| **Incremental** | Add papers over time, pipeline updates automatically |

---

## Target Users & Usage

### Primary User

**Jake** - Developer/researcher who accumulates papers faster than he can organize them. Needs to query his collection without leaving his coding environment.

### Usage Patterns

| Trigger | Query Type | Example |
|---------|------------|---------|
| Curiosity | Semantic search | "What papers do I have on graph neural networks?" |
| Project work | Related work discovery | "Find papers relevant to knowledge graph embeddings" |
| Writing/coding | Synthesis | "Summarize what my collection says about attention mechanisms" |

### Paper Ingestion

- **Batch**: Dump a folder of PDFs when starting a new research area
- **Incremental**: Add individual papers as discovered

### Integration Context

- **Primary**: Mid-coding query from Claude Code via MCP
- **Secondary**: Dedicated research exploration sessions

---

## MVP Scope

### In Scope
- PDF ingestion pipeline (batch + incremental)
- Cloud vector store (LlamaIndex managed)
- Semantic search with citation-backed responses
- MCP server exposing query tools
- Cross-citation awareness within collection

### Out of Scope (Post-MVP)
- Highlight/annotation extraction
- Agent skills beyond basic query
- Scale beyond hundreds of papers
- Local/offline mode
