<p align="center">
  <img src="assets/header.png" alt="LangChain Tutorials" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/Anthropic-Claude-D97757?style=for-the-badge&logo=anthropic&logoColor=white" alt="Anthropic"/>
  <img src="https://img.shields.io/badge/Meta-LLaMA-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="Meta"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

---

## Scope & Coverage

A collection of self-contained, code-first tutorials that cover LangChain from foundational concepts to production patterns. Each tutorial is a standalone module with a runnable notebook and a README documenting the core logic.

Built for engineers who want working references — not theory dumps.

---

## Tutorials

| # | Tutorial | What You'll Build | Status |
|:-:|----------|-------------------|:------:|
| 01 | [**LangChain Basics**](./01-langchain-basics/) | Prompts, LLM wrappers, chains, batch & streaming | ✅ |
| 02 | [**LCEL Deep Dive**](./02-lcel-deep-dive/) | RunnableParallel, Lambda, Branch, Fallbacks | ✅ |
| 03 | **Output Parsers** | JSON, Pydantic, Enum, auto-fixing parsers | 🔜 |
| 04 | **Document Loaders** | PDF, CSV, Web, YouTube, GitHub loaders | 🔜 |
| 05 | **Text Splitters** | Recursive, Token, Semantic chunking | 🔜 |
| 06 | **RAG with FAISS** | Embeddings, vector store, retrieval chain | 🔜 |
| 07 | **RAG with ChromaDB** | Persistent store, metadata filtering, MMR | 🔜 |
| 08 | **Conversational Memory** | Buffer, Summary, Window, Entity memory | 🔜 |
| 09 | **Agents & Custom Tools** | ReAct agent, custom tools, tool routing | 🔜 |
| 10 | **Callbacks & Tracing** | Custom handlers, LangSmith, cost tracking | 🔜 |

---

## Learning Path

```mermaid
flowchart TD
    subgraph Foundations
        A["01 · Basics"] --> B["02 · LCEL"] --> C["03 · Output Parsers"]
    end

    subgraph Data_Pipeline["Data Pipeline"]
        D["04 · Doc Loaders"] --> E["05 · Text Splitters"]
    end

    subgraph RAG
        F["06 · RAG + FAISS"] --> G["07 · RAG + Chroma"]
    end

    subgraph Advanced
        H["08 · Memory"] --> I["09 · Agents"] --> J["10 · Callbacks"]
    end

    C --> D
    E --> F
    G --> H

    style A fill:#1e1e2e,stroke:#00d4aa,color:#cdd6f4
    style B fill:#1e1e2e,stroke:#00d4aa,color:#cdd6f4
    style C fill:#1e1e2e,stroke:#00d4aa,color:#cdd6f4
    style D fill:#1e1e2e,stroke:#3b82f6,color:#cdd6f4
    style E fill:#1e1e2e,stroke:#3b82f6,color:#cdd6f4
    style F fill:#1e1e2e,stroke:#f59e0b,color:#cdd6f4
    style G fill:#1e1e2e,stroke:#f59e0b,color:#cdd6f4
    style H fill:#1e1e2e,stroke:#ef4444,color:#cdd6f4
    style I fill:#1e1e2e,stroke:#ef4444,color:#cdd6f4
    style J fill:#1e1e2e,stroke:#06b6d4,color:#cdd6f4
```

Tutorials are grouped by theme but self-contained — jump to any topic that's relevant to you.

---

## Quick Start

```bash
git clone https://github.com/HitPant/langchain-tutorials.git
cd langchain-tutorials
pip install langchain langchain-openai langchain-anthropic langchain-community \
            faiss-cpu chromadb tiktoken
```

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Pick a tutorial folder and open the notebook.

---

## Who This Is For

- **Software Engineers** building LLM-powered products
- **ML/AI Engineers** evaluating LangChain for production
- **Solutions Engineers** needing quick reference implementations

---

<p align="center">
  Built by <a href="https://github.com/HitPant">Hitesh Pant</a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/hitesh-pant/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="https://github.com/HitPant"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://medium.com/@hitpant"><img src="https://img.shields.io/badge/Medium-000000?style=flat-square&logo=medium&logoColor=white" alt="Medium"/></a>
  <a href="https://x.com/hitpant21"><img src="https://img.shields.io/badge/X-000000?style=flat-square&logo=x&logoColor=white" alt="X"/></a>
  <a href="https://hitpant.github.io/"><img src="https://img.shields.io/badge/Portfolio-222222?style=flat-square&logo=githubpages&logoColor=white" alt="Portfolio"/></a>
</p>
