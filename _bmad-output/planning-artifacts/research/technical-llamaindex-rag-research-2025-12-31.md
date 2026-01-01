# Technical Research: Building an Intermediate RAG System with LlamaIndex

**Research Type:** Technical
**Topic:** LlamaIndex RAG for Research Papers
**Date:** 2025-12-31
**Status:** Complete

---

## Executive Summary

This research covers how to build an intermediate RAG (Retrieval-Augmented Generation) system using LlamaIndex, optimized for ingesting and querying dozens to hundreds of research papers. The system leverages the latest 2025 techniques in document parsing, hierarchical indexing, hybrid retrieval, and evaluation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  PDFs → LlamaParse → HierarchicalNodeParser → Metadata Extraction → Index   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Vector Store (Qdrant/Chroma)  +  DocStore (for hierarchical retrieval)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             QUERY PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Hybrid Retriever → AutoMergingRetriever → Cohere Rerank → LLM Synthesis   │
│  (BM25 + Vector)       (context expansion)     (precision)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Document Ingestion

### 1.1 PDF Parsing with LlamaParse

LlamaParse is LlamaIndex's official API for advanced PDF parsing, specifically designed for complex documents with tables, figures, and equations. For academic papers, this is the **strongly recommended** choice.

**Installation:**
```bash
pip install llama-parse llama-index
```

**Configuration for Research Papers:**
```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="your-llamacloud-api-key",
    result_type="markdown",
    parse_mode="agentic_plus",      # Best accuracy, uses Anthropic Sonnet 4.0
    preset="scientific-v-1",         # Optimized for academic publications
    extract_charts=True,
    auto_mode=True,
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
    parsing_instruction="Output equations in LaTeX format between $$.",
)
```

**LlamaParse Modes:**

| Mode | Best For | Features |
|------|----------|----------|
| **Cost-Effective** | Standard documents | Basic OCR, adaptive tables |
| **Agentic** | Documents with diagrams | LaTeX equations, Mermaid diagrams |
| **Agentic Plus** | Research papers, complex layouts | Best accuracy, Anthropic Sonnet 4.0 |

### 1.2 Chunking Strategies

#### HierarchicalNodeParser (Recommended for Research Papers)

Creates a multi-level hierarchy of chunks with parent-child relationships. When combined with `AutoMergingRetriever`, it enables dynamic context expansion.

```python
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)

# Default: [2048, 512, 128] character chunks
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # Section → paragraph → sentence
)
nodes = node_parser.get_nodes_from_documents(documents)

# Get different levels
leaf_nodes = get_leaf_nodes(nodes)  # Smallest (128) - for embedding
root_nodes = get_root_nodes(nodes)  # Largest (2048)
```

#### SentenceWindowNodeParser (Alternative)

Splits documents into individual sentences but stores surrounding context in metadata:

```python
from llama_index.core.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,  # 3 sentences on each side
    window_metadata_key="window",
    original_text_metadata_key="original_sentence",
)
```

#### SemanticSplitterNodeParser (Context-Aware)

Uses embedding similarity to find natural breakpoints:

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model,
)
```

### 1.3 Metadata Extraction

```python
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4")

extractors = [
    TitleExtractor(llm=llm),
    SummaryExtractor(summaries=["prev", "self"], llm=llm),
    QuestionsAnsweredExtractor(questions=3, llm=llm),
    KeywordExtractor(keywords=10, llm=llm),
]

pipeline = IngestionPipeline(
    transformations=[
        node_parser,
        *extractors,
        embed_model,
    ]
)

# Parallel processing for hundreds of documents
nodes = pipeline.run(documents=documents, num_workers=4)
```

### 1.4 Batch Processing Performance

**Benchmarks (from LlamaIndex documentation with ~5,297 nodes):**
- Async + Parallel: ~20 seconds
- Sync + Parallel: ~29 seconds
- Sync + Sequential: ~71 seconds

```python
# Enable caching for incremental updates
from llama_index.core.ingestion import IngestionCache

pipeline = IngestionPipeline(
    transformations=[...],
    cache=IngestionCache(),  # Prevents re-processing
)
```

---

## 2. Indexing Strategies

### 2.1 Vector Store Comparison

| Vector Store | Best For | Pros | Cons |
|-------------|----------|------|------|
| **Qdrant** | Production RAG | Rust performance, powerful filtering, OSS + managed | Requires tier tuning |
| **Chroma** | Prototyping | Simple setup, excellent Python integration | Not for billions of vectors |
| **Weaviate** | Hybrid search | Built-in BM25 + vector, GraphQL API | Higher storage costs |
| **Pinecone** | Enterprise | Fully managed, scalable, guaranteed SLAs | Expensive ($50-$500+/mo) |

**Qdrant Setup (Recommended for Production):**
```python
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

client = QdrantClient(path="./qdrant_db")  # Local persistent
vector_store = QdrantVectorStore(
    client=client,
    collection_name="research_papers",
)
```

### 2.2 Index Types

| Index Type | Best For | Use Case |
|------------|----------|----------|
| **VectorStoreIndex** | Semantic similarity | General Q&A, finding passages |
| **DocumentSummaryIndex** | Document-level retrieval | Paper selection before drilling in |
| **PropertyGraphIndex** | Entity relationships | Citation networks, author connections |

**Recommendation:** Combine DocumentSummaryIndex for high-level selection + VectorStoreIndex for detailed retrieval.

### 2.3 AutoMergingRetriever (Key Technique)

When multiple small chunks from the same parent are retrieved, automatically returns the full parent context:

```python
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

# Store all nodes in docstore
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
storage_context = StorageContext.from_defaults(docstore=docstore)

# Index ONLY leaf nodes
base_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

# Create auto-merging retriever
base_retriever = base_index.as_retriever(similarity_top_k=12)
retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    simple_ratio_thresh=0.5,  # Merge if 50%+ children retrieved
)
```

### 2.4 Persistent Storage

```python
from llama_index.core import StorageContext, load_index_from_storage

# Save
index.storage_context.persist(persist_dir="./storage")

# Load
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

---

## 3. Query Techniques

### 3.1 Hybrid Search (BM25 + Vector)

Essential for research papers where exact terminology matters (e.g., "BERT", "ImageNet", "p < 0.05"):

```python
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# Create both retrievers
vector_retriever = index.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10,
)

# Fuse with Reciprocal Rank Fusion
retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=4,  # Generate 4 query variants
    mode="reciprocal_rerank",
    use_async=True,
)
```

### 3.2 Reranking

Two-stage retrieval: initial high-recall retrieval followed by precision reranking.

**Cohere Reranking:**
```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(
    api_key=api_key,
    top_n=5,
    model="rerank-english-v3.0"
)

query_engine = index.as_query_engine(
    similarity_top_k=25,  # Retrieve more initially
    node_postprocessors=[cohere_rerank],
)
```

**Open-Source Alternative (Sentence Transformers):**
```python
from llama_index.postprocessors.sbert_rerank import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5
)
```

### 3.3 Query Transformations

#### HyDE (Hypothetical Document Embeddings)

Generates a hypothetical document/answer, then uses that for embedding lookup:

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(base_query_engine, hyde)
```

#### Sub-Question Decomposition

Complex queries are broken into targeted sub-questions:

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=ml_engine,
        name="ml_papers",
        description="Research papers on machine learning",
    ),
    QueryEngineTool.from_defaults(
        query_engine=nlp_engine,
        name="nlp_papers",
        description="Research papers on NLP",
    ),
]

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)
```

### 3.4 Response Synthesis Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `compact` (default) | Concatenates chunks before LLM call | Cost efficiency |
| `refine` | Iteratively refines through each chunk | Comprehensive answers |
| `tree_summarize` | Recursively summarizes in tree | Summarization |

```python
query_engine = index.as_query_engine(response_mode="tree_summarize")
```

### 3.5 Router Query Engine

Routes queries to appropriate indexes based on intent:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector

query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        QueryEngineTool.from_defaults(
            query_engine=summary_engine,
            description="For high-level overview questions",
        ),
        QueryEngineTool.from_defaults(
            query_engine=detailed_engine,
            description="For specific technical details",
        ),
    ],
)
```

### 3.6 Citation Tracking

```python
from llama_index.core.query_engine import CitationQueryEngine

citation_engine = CitationQueryEngine.from_args(
    index,
    citation_chunk_size=512,
)

response = citation_engine.query("What are the key findings?")
# Response includes [1], [2] citations
print(response.source_nodes)  # Access cited sources
```

### 3.7 Agentic RAG with ReAct

```python
from llama_index.core.agent.workflow import ReActAgent

agent = ReActAgent(
    tools=query_engine_tools,
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True
)

response = await agent.run("Compare the risk factors between papers", ctx=ctx)
```

### 3.8 Streaming Responses

```python
query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Explain the concepts")
streaming_response.print_response_stream()
```

---

## 4. Model Recommendations

### 4.1 LLM and Embedding Combinations

| Quality | Embedding | LLM | Reranker |
|---------|-----------|-----|----------|
| **Top Accuracy** | `text-embedding-3-large` | GPT-4o | CohereRerank |
| **Open-Source** | BGE-M3 | LLaMA-3.1 70B | bge-reranker-large |
| **Cost-Effective** | BGE-M3 (self-hosted) | LLaMA-3.1 8B | bge-reranker-base |
| **Multilingual** | BGE-M3 (100+ languages) | GPT-4o | CohereRerank |

**Benchmark Results (OpenAI + CohereRerank):**
- Hit Rate: 0.927
- MRR: 0.866

### 4.2 Configuration (Settings vs ServiceContext)

`ServiceContext` is **deprecated** as of v0.10.0. Use `Settings`:

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

---

## 5. Evaluation and Optimization

### 5.1 Built-in Evaluators

| Evaluator | Purpose | Requires Ground Truth |
|-----------|---------|----------------------|
| **FaithfulnessEvaluator** | Detects hallucinations | No |
| **RelevancyEvaluator** | Measures relevance to query | No |
| **CorrectnessEvaluator** | Compares to reference | Yes |

```python
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

faithfulness = FaithfulnessEvaluator(llm=gpt4)
result = faithfulness.evaluate_response(response=response)
print(f"Passing: {result.passing}")  # True if no hallucination
```

### 5.2 Batch Evaluation

```python
from llama_index.core.evaluation import BatchEvalRunner

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
    workers=8,
)

results = await runner.aevaluate_queries(query_engine, queries=eval_questions)
```

### 5.3 External Framework Integration

**RAGAS:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
```

**DeepEval (CI/CD):**
```python
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric

assert_test(test_case, [FaithfulnessMetric(threshold=0.7)])
```

**TruLens (RAG Triad):**
```python
from trulens.apps.llamaindex import TruLlama

tru_query_engine = TruLlama(
    query_engine,
    feedbacks=[context_relevance, groundedness, answer_relevance]
)
```

### 5.4 Hallucination Prevention

1. Use FaithfulnessEvaluator in production
2. Include source citations (CitationQueryEngine)
3. Constrain responses to context with strict prompts
4. Use lower temperature for factual queries

### 5.5 Observability with Phoenix

```python
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
# All LlamaIndex operations now traced
```

---

## 6. Chat Interface with Memory

### 6.1 CondensePlusContextChatEngine (Recommended)

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=8192)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=index.as_retriever(similarity_top_k=5),
    memory=memory,
    llm=Settings.llm,
    node_postprocessors=[reranker],
    system_prompt="""You are a research assistant specializing in academic papers.
    Answer questions based on retrieved context. Cite specific papers when possible.""",
)

# Multi-turn conversation
response1 = chat_engine.chat("What are the main findings on transformers?")
response2 = chat_engine.chat("How does that compare to earlier approaches?")
```

### 6.2 Long-Term Memory

```python
from llama_index.core.memory import Memory
from llama_index.core.memory.memory_blocks import VectorMemoryBlock, FactExtractionMemoryBlock

memory = Memory.from_defaults(
    session_id="user_123",
    memory_blocks=[
        FactExtractionMemoryBlock(name="facts", max_facts=50, llm=llm),
        VectorMemoryBlock(name="long_term", vector_store=vector_store),
    ]
)
```

---

## 7. Complete Production Example

```python
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine, CitationQueryEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
from qdrant_client import QdrantClient

# 1. Configure Settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# 2. Parse PDFs with LlamaParse
parser = LlamaParse(
    api_key="your-key",
    parse_mode="agentic_plus",
    preset="scientific-v-1",
    extract_charts=True,
)
documents = SimpleDirectoryReader(
    "./research_papers/",
    file_extractor={".pdf": parser}
).load_data()

# 3. Hierarchical chunking
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)

# 4. Setup vector store and storage
client = QdrantClient(path="./qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="papers")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
storage_context.docstore.add_documents(nodes)

# 5. Create index with leaf nodes only
index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

# 6. Build hybrid retriever with auto-merging
auto_merging = AutoMergingRetriever(
    index.as_retriever(similarity_top_k=12),
    storage_context,
    simple_ratio_thresh=0.5,
)
bm25 = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=12)
hybrid = QueryFusionRetriever(
    [auto_merging, bm25],
    similarity_top_k=15,
    mode="reciprocal_rerank",
    use_async=True,
)

# 7. Add reranking
reranker = CohereRerank(api_key="your-key", top_n=5)

# 8. Create query engine with citations
query_engine = CitationQueryEngine.from_args(
    index,
    retriever=hybrid,
    node_postprocessors=[reranker],
    response_mode="compact",
    citation_chunk_size=512,
)

# 9. Create chat engine with memory
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=hybrid,
    memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
    node_postprocessors=[reranker],
    system_prompt="You are a research paper assistant. Cite papers when possible.",
)

# Usage
response = query_engine.query("What are the key innovations in attention mechanisms?")
print(response)
print(f"Sources: {[n.metadata.get('title') for n in response.source_nodes]}")
```

---

## 8. Key Takeaways

1. **Use LlamaParse** with `scientific-v-1` preset for research papers - handles tables, figures, equations
2. **HierarchicalNodeParser + AutoMergingRetriever** provides optimal context windows
3. **Hybrid search (BM25 + Vector)** is essential - catches both exact terms and semantic matches
4. **Cohere reranking** significantly improves precision (benchmark: 0.927 hit rate)
5. **CitationQueryEngine** for traceable, verifiable responses
6. **Settings replaces ServiceContext** in modern LlamaIndex (v0.10+)
7. **Evaluate with multiple metrics** - faithfulness, relevancy, groundedness
8. **Add observability** (Phoenix/LlamaTrace) from the start
9. **CondensePlusContextChatEngine** for multi-turn conversations with memory

---

## Sources

### Document Ingestion
- [Loading Data (Ingestion) - LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/loading/loading/)
- [Node Parser Modules](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/)
- [Metadata Extraction](https://docs.llamaindex.ai/en/stable/module_guides/indexing/metadata_extraction/)
- [LlamaParse Modes and Presets](https://developers.llamaindex.ai/python/cloud/llamaparse/presets_and_modes/presets)
- [Auto Merging Retriever](https://developers.llamaindex.ai/python/examples/retrievers/auto_merging_retriever/)

### Indexing Strategies
- [LiquidMetal AI - Vector Database Comparison 2025](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [LlamaIndex Blog - Property Graph Index](https://www.llamaindex.ai/blog/introducing-the-property-graph-index)
- [LlamaIndex Docs - BM25 Retriever](https://developers.llamaindex.ai/python/examples/retrievers/bm25_retriever/)
- [LlamaIndex Docs - Storing](https://developers.llamaindex.ai/python/framework/module_guides/storing/)

### Query Techniques
- [Query Transform Cookbook](https://developers.llamaindex.ai/python/examples/query_transformations/query_transform_cookbook/)
- [Reciprocal Rerank Fusion Retriever](https://developers.llamaindex.ai/python/examples/retrievers/reciprocal_rerank_fusion/)
- [Response Modes](https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine/response_modes/)
- [Router Query Engine](https://developers.llamaindex.ai/python/framework/module_guides/querying/router/)
- [CitationQueryEngine](https://developers.llamaindex.ai/python/examples/query_engine/citation_query_engine/)
- [ReAct Agent with Query Engine Tools](https://developers.llamaindex.ai/python/examples/agent/react_agent_with_query_engine/)

### Evaluation & Optimization
- [LlamaIndex Evaluating Module Guide](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/)
- [Building Performant RAG Applications for Production](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)
- [Evaluating RAG with DeepEval and LlamaIndex](https://www.llamaindex.ai/blog/evaluating-rag-with-deepeval-and-llamaindex)
- [TruLens LlamaIndex Integration](https://www.trulens.org/component_guides/instrumentation/llama_index/)
- [Phoenix LlamaTrace](https://phoenix.arize.com/llamatrace/)

### Architecture & Models
- [ServiceContext to Settings Migration](https://developers.llamaindex.ai/python/framework/module_guides/supporting_modules/service_context_migration/)
- [Boosting RAG: Picking the Best Embedding & Reranker Models](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)
- [Multi-Modal RAG](https://www.llamaindex.ai/blog/multi-modal-rag-621de7525fea)
- [Memory - LlamaIndex Docs](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/)
- [Chat Engine - Condense Plus Context Mode](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/)
