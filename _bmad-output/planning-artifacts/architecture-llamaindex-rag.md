---
stepsCompleted: [1, 2, 3, 4, 5]
inputDocuments:
  - product-brief-llamaindex-rag-2025-12-31.md
  - technical-llamaindex-rag-research-2025-12-31.md
workflowType: 'architecture'
project_name: 'llamaindex-rag'
user_name: 'Jake'
date: '2025-12-31'
---

# Architecture Decision Document: LlamaIndex RAG System

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
- PDF ingestion with batch and incremental modes
- Semantic search across paper collection ("papers about X")
- Synthesis queries with source citations ("summarize what I have on Y")
- Cross-citation discovery within collection
- MCP server interface for Claude Code integration

**Non-Functional Requirements:**
- Cloud-first using LlamaCloud managed services
- Scale target: hundreds of papers without added complexity
- Learning-focused: clean, understandable implementation
- Citation-aware: every response traces to source chunks

**Scale & Complexity:**
- Primary domain: Backend/API (MCP server + RAG pipeline)
- Complexity level: Low-Medium
- Estimated architectural components: 4-5 modules

### Technical Constraints & Dependencies

| Constraint | Value |
|------------|-------|
| Framework | LlamaIndex (v0.10+ patterns) |
| PDF Parsing | LlamaParse (LlamaCloud) |
| Vector Store | LlamaCloud managed |
| Indexing | LlamaCloud managed |
| Retrieval | LlamaCloud API |
| MCP Server | Custom Python implementation |
| LLM Access | OpenAI/Anthropic via LlamaIndex |
| Environment | Python with uv |

### Responsibility Split

**What You Build:**
1. MCP server wrapper exposing query tools
2. Ingestion script to push PDFs to LlamaCloud
3. Query interface translating MCP calls → LlamaCloud API

**What LlamaCloud Handles:**
- PDF parsing and structure extraction
- Chunking and embedding
- Vector storage and indexing
- Retrieval and ranking

### Cross-Cutting Concerns Identified

- **Citation tracking**: Must persist through all query paths
- **Incremental indexing**: Avoid full reprocessing on paper add
- **Metadata management**: Paper titles, authors, dates for filtering/display

---

## Project Structure

### Primary Technology Domain

Python Backend (MCP Server + LlamaCloud client)

### Technical Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| Package Manager | uv |
| MCP SDK | mcp (Anthropic official) |
| RAG Framework | llama-index, llama-cloud, llama-parse |

### Directory Structure

```
research-rag/
├── pyproject.toml          # uv managed dependencies
├── src/
│   └── research_rag/
│       ├── __init__.py
│       ├── server.py       # MCP server entry point
│       ├── tools/          # MCP tool definitions
│       │   ├── __init__.py
│       │   ├── search.py   # Semantic search tool
│       │   └── ingest.py   # Paper ingestion tool
│       └── cloud/          # LlamaCloud client wrapper
│           ├── __init__.py
│           └── client.py
├── papers/                 # Local PDF staging folder
└── README.md
```

### Core Dependencies

```toml
[project]
dependencies = [
    "mcp",                    # Anthropic MCP SDK
    "llama-index",            # LlamaIndex core
    "llama-cloud",            # LlamaCloud client
    "llama-parse",            # PDF parsing
]
```

### Architectural Patterns

- **Standard Python package layout** with src/ directory
- **MCP server pattern** from Anthropic's SDK
- **LlamaCloud client wrapper** for all RAG operations
- **Tool-based organization** matching MCP tool exposure

---

## Core Architectural Decisions

### LlamaCloud Configuration

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Index Strategy | Single index | Simplicity for personal collection, can partition later if needed |
| Parsing | LlamaParse scientific preset | Best for academic papers with equations/tables |
| Chunking | LlamaCloud default (hierarchical) | Leverage platform defaults |

### MCP Tool Design

| Tool | Parameters | Returns |
|------|------------|---------|
| `search_papers` | `query: str`, `top_k: int = 5` | Results with citations to source chunks |
| `ingest_paper` | `file_path: str` | Confirmation + paper metadata |
| `list_papers` | none | Paper titles, authors, dates in index |

**Deferred:** `summarize_topic` - handled by subagent orchestration outside MCP server (Claude does synthesis)

### Configuration

| Setting | Source |
|---------|--------|
| `LLAMACLOUD_API_KEY` | Environment variable |
| Index name | Environment variable or hardcoded default |

**Note:** No LLM API key needed - Claude (the client) handles all synthesis. MCP server only retrieves chunks.

### Paper Ingestion Flow

```
Claude Code → MCP ingest_paper(file_path) → Server reads file → LlamaParse → LlamaCloud index
```

- File paths passed directly to tool
- Supports single file or batch via multiple calls
- Server handles upload to LlamaCloud

### Query Flow

```
Claude Code → MCP search_papers(query) → LlamaCloud retrieval → Return chunks with citations → Claude synthesizes
```

- MCP server is retrieval-only
- Claude (client) receives chunks and generates response
- Citations preserved in chunk metadata

---

## Implementation Patterns

### Python Naming (PEP 8)

| Element | Convention | Example |
|---------|------------|---------|
| Files | snake_case | `search.py`, `cloud_client.py` |
| Functions | snake_case | `search_papers()`, `ingest_paper()` |
| Classes | PascalCase | `CloudClient`, `PaperMetadata` |
| Constants | UPPER_SNAKE | `DEFAULT_TOP_K`, `INDEX_NAME` |

### MCP Tool Response Format

Tools return data directly - MCP protocol handles errors:

```python
# search_papers returns list of results
[
    {
        "paper_title": "Attention Is All You Need",
        "chunk_text": "The dominant sequence transduction models...",
        "chunk_id": "abc123",
        "score": 0.92,
        "metadata": {
            "authors": ["Vaswani et al."],
            "page": 3,
            "section": "Introduction",
            # ... any additional LlamaIndex metadata
        }
    },
    ...
]
```

**Key rule:** Every chunk includes `paper_title` + all metadata from LlamaIndex.

### Error Handling

| Scenario | Approach |
|----------|----------|
| File not found | Raise `FileNotFoundError` with path |
| LlamaCloud API error | Raise `ConnectionError` with message |
| Invalid file type | Raise `ValueError` with guidance |

Standard Python exceptions - MCP SDK translates to error responses.

### Logging

```python
import logging
logger = logging.getLogger(__name__)
```

| Level | Use For |
|-------|---------|
| DEBUG | API call details, chunk counts |
| INFO | Paper ingested, search executed |
| WARNING | Retry attempted |
| ERROR | Failed operations |
