"""
MS-ADS RAG Streamlit App — Milvus/Zilliz + LangChain + Streaming

Local secrets (recommended): create .streamlit/secrets.toml next to app.py:
---------------------------------------------------------------------------
OPENAI_API_KEY = "sk-..."
# If using Zilliz Cloud (remote Milvus):
ZILLIZ_CLOUD_URI = "https://in03-....zilliz.com"
ZILLIZ_CLOUD_API_KEY = "..."
# Optional
MILVUS_COLLECTION = "rag_collection"

If fully local Milvus (milvus-lite), do NOT set ZILLIZ_* and:
pip install "pymilvus[milvus_lite]>=2.4.8"
"""

from __future__ import annotations

# stdlib
import os
import re
import json
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# third-party
import streamlit as st
from pydantic import BaseModel
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()

# LangChain core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangChain integrations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_milvus import Milvus


# ---------------------------------------------------------------------
# 🔐 Secrets / Environment bootstrap (works local & Streamlit Cloud)
# ---------------------------------------------------------------------
load_dotenv()  # pull from .env if present

# bridge st.secrets -> os.environ for any libs that read env vars directly
for k in ("OPENAI_API_KEY", "ZILLIZ_CLOUD_URI", "ZILLIZ_CLOUD_API_KEY", "MILVUS_COLLECTION"):
    try:
        if k in st.secrets and st.secrets[k]:
            os.environ[k] = str(st.secrets[k])
    except Exception:
        # st.secrets exists only inside Streamlit runtime
        pass


# ---------------------------------------------------------------------
# 🧩 Data models for the UI
# ---------------------------------------------------------------------
class Citation(BaseModel):
    title: str
    url: Optional[str] = None
    chunk: str
    score: Optional[float] = None
    doc_id: Optional[str] = None


class RagResponse(BaseModel):
    answer: str
    citations: List[Citation]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    raw_debug: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------
# ⚙️ RAG config + components (Milvus/Zilliz + OpenAI)
# ---------------------------------------------------------------------
@dataclass
class RAGConfig:
    milvus_uri: Optional[str] = None
    milvus_token: Optional[str] = None
    milvus_collection: str = os.environ.get("MILVUS_COLLECTION", "rag_collection")
    embeddings_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    top_k_retrieval: int = 20
    final_k: int = 5
    mmr_diversity: float = 0.3
    temperature: float = 0.0
    streaming: bool = True  # default on; UI can still disable

    def __post_init__(self):
        if self.milvus_uri is None:
            self.milvus_uri = os.environ.get("ZILLIZ_CLOUD_URI", "./milvus_demo.db")
        if self.milvus_token is None:
            self.milvus_token = os.environ.get("ZILLIZ_CLOUD_API_KEY", "")


class RAGRetriever:
    """Simple retriever with MMR diversity and optional metadata filter."""

    def __init__(self, vectorstore: Milvus, config: RAGConfig):
        self.vectorstore = vectorstore
        self.config = config

    def retrieve(self, query: str, exclude_page_types: Optional[List[str]] = None) -> List[Document]:
        candidates = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.config.top_k_retrieval,
            fetch_k=self.config.top_k_retrieval * 2,
            lambda_mult=1 - self.config.mmr_diversity,
        )
        if exclude_page_types:
            candidates = [d for d in candidates if d.metadata.get("page_type") not in exclude_page_types]
        return candidates[: self.config.final_k]


class RAGChain:
    """LCEL chain with retrieval → prompt → LLM → parse. Supports streaming."""

    def __init__(self, retriever: RAGRetriever, config: RAGConfig):
        self.retriever = retriever
        self.config = config
        self.llm = ChatOpenAI(model=config.llm_model, temperature=config.temperature, streaming=config.streaming)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions about the MS in Applied Data Science program at UChicago.

Use the following context to answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        )
        self.chain = self._build_chain()

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n---\n\n".join(
            [
                f"Source: {d.metadata.get('title') or d.metadata.get('section_title') or d.metadata.get('page_title') or 'Unknown'} "
                f"[{d.metadata.get('chunk_type','content')}]\n{d.page_content}"
                for d in docs
            ]
        )

    def _build_chain(self):
        retrieval_chain = RunnablePassthrough.assign(
            context=lambda x: self._format_docs(self.retriever.retrieve(x["question"]))
        )
        return retrieval_chain | self.prompt | self.llm | StrOutputParser()

    def invoke(self, question: str) -> str:
        return self.chain.invoke({"question": question})

    def invoke_with_sources(self, question: str) -> Dict[str, Any]:
        docs = self.retriever.retrieve(question)
        context = self._format_docs(docs)
        answer = (self.prompt | self.llm | StrOutputParser()).invoke({"context": context, "question": question})
        return {"answer": answer, "sources": docs}


# ---------------------------------------------------------------------
# 🗄️ Cached initialization — connect to Milvus (cloud OR local)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def _init_rag() -> Tuple[RAGRetriever, RAGChain]:
    cfg = RAGConfig()

    # If using cloud, require token; if not, expect local milvus-lite installed
    use_cloud = bool(cfg.milvus_uri and str(cfg.milvus_uri).startswith("http"))
    if use_cloud and not cfg.milvus_token:
        raise RuntimeError("ZILLIZ_CLOUD_API_KEY is missing. Add it via secrets or env.")

    embeddings = OpenAIEmbeddings(model=cfg.embeddings_model)  # needs OPENAI_API_KEY
    connection_args: Dict[str, Any] = {"uri": cfg.milvus_uri}
    if use_cloud:
        connection_args["token"] = cfg.milvus_token

    # For fully local: ensure `pip install "pymilvus[milvus_lite]"`
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=cfg.milvus_collection,
        connection_args=connection_args,
        auto_id=True,
        drop_old=False,
    )
    retriever = RAGRetriever(vectorstore, cfg)
    rag_chain = RAGChain(retriever, cfg)
    return retriever, rag_chain


# ---------------------------------------------------------------------
# 🔎 Utility — normalize LangChain Documents → UI Citations
# ---------------------------------------------------------------------
def _to_citations_from_docs(docs: List[Document]) -> List[Citation]:
    out: List[Citation] = []
    for d in docs:
        md = d.metadata or {}
        title = md.get("title") or md.get("section_title") or md.get("section") or md.get("page_title") or "Source"
        url = md.get("url") or md.get("source") or md.get("page_url")
        chunk = d.page_content or md.get("text", "")
        score = md.get("score")
        doc_id = md.get("doc_id") or md.get("id")
        out.append(Citation(title=title, url=url, chunk=chunk, score=score, doc_id=doc_id))
    return out


# ---------------------------------------------------------------------
# 🌀 Streaming builder — always returns (generator, citations) or raises
# ---------------------------------------------------------------------
def _build_streamer(question: str, top_k: int, temperature: float):
    retriever, rag_chain = _init_rag()
    retriever.config.final_k = int(top_k)

    # Retrieve once for citations + context
    docs = retriever.retrieve(question)
    citations = _to_citations_from_docs(docs)

    # Fresh LLM with streaming=True to avoid cache invalidation
    streaming_llm = ChatOpenAI(model=rag_chain.config.llm_model, temperature=float(temperature), streaming=True)
    chain = rag_chain.prompt | streaming_llm | StrOutputParser()
    context = rag_chain._format_docs(docs)

    def gen():
        for token in chain.stream({"context": context, "question": question}):
            yield token

    return gen, citations


# ---------------------------------------------------------------------
# 🚀 Non-streaming entrypoint the UI can call directly
# ---------------------------------------------------------------------
def invoke_rag(
    question: str,
    *,
    top_k: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 800,  # reserved for future use
    retrieve_only: bool = False,
) -> RagResponse:
    retriever, rag_chain = _init_rag()
    retriever.config.final_k = int(top_k)
    rag_chain.config.temperature = float(temperature)

    start = time.time()
    if retrieve_only:
        docs = retriever.retrieve(question)
        answer_text = "🔍 Retrieved top sections only (generation disabled)."
    else:
        res = rag_chain.invoke_with_sources(question)
        answer_text = res["answer"]
        docs = res["sources"]

    citations = _to_citations_from_docs(docs)
    end = time.time()
    return RagResponse(
        answer=answer_text,
        citations=citations,
        latency_ms=int((end - start) * 1000),
        raw_debug={"retrieved": len(citations)},
    )


# ---------------------------------------------------------------------
# 🧰 UI helpers
# ---------------------------------------------------------------------
def _highlight(text: str, terms: List[str]) -> str:
    if not terms:
        return text
    for t in sorted(set([t for t in terms if t]), key=len, reverse=True):
        pat = re.compile(rf"(\b{re.escape(t)}\b)", re.IGNORECASE)
        text = pat.sub(r"<mark>\1</mark>", text)
    return text


def _mk_citation_label(i: int) -> str:
    return f"[{i+1}]"


def _save_feedback(row: Dict[str, Any]):
    try:
        with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------
# 🎨 Streamlit UI
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MS-ADS RAG Assistant", page_icon="🧠", layout="wide")

    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts

    # Sidebar — controls
    with st.sidebar:
        st.title("⚙️ Settings")
        st.caption("Tune retrieval & generation parameters")
        top_k = st.slider("Top-K results", min_value=1, max_value=20, value=5)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
        max_tokens = st.slider("Max tokens", 200, 2000, 800, 50)
        stream_mode = st.toggle("Stream answer", value=True)
        show_chunks = st.toggle("Show retrieved chunks", value=True)
        show_debug = st.toggle("Show debug info", value=False)
        st.markdown("---")
        st.caption("💾 Export")
        if st.button("Download conversation as JSON"):
            data = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
            st.download_button("Save JSON", data=data, file_name="msads_rag_history.json")
        if show_debug:
            st.caption(f"Has OPENAI_API_KEY? {bool(os.getenv('OPENAI_API_KEY'))}")
            st.caption(f"Milvus URI: {os.getenv('ZILLIZ_CLOUD_URI')!r}")
            st.caption(f"Has Zilliz token? {bool(os.getenv('ZILLIZ_CLOUD_API_KEY'))}")

    # ---------- UChicago MS-ADS palette + components ----------
    st.markdown(
        """
        <style>
        :root {
          --uch-maroon:#800000;
          --uch-maroon-dark:#5e0000;
          --uch-gray:#6e6e6e;
          --uch-light:#f6f3f0;
          --uch-border:#e6e2df;
        }
        html, body { background: var(--uch-light); }
        a, .src a { color: var(--uch-maroon) !important; text-decoration: underline; }
        .stButton>button {
          background: var(--uch-maroon);
          border: 1px solid var(--uch-maroon);
          color: #fff;
          border-radius: 8px;
          padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
          background: var(--uch-maroon-dark);
          border-color: var(--uch-maroon-dark);
        }
        div[data-testid="stSidebar"] {
          background: #fff;
          border-right: 1px solid var(--uch-border);
        }
        h1, h2, h3 { font-weight: 700; }
        .topbar {
          background: var(--uch-maroon);
          color: #fff;
          padding: 12px 16px;
          border-radius: 12px;
          margin-bottom: 8px;
        }
        .topbar .brand { display:flex; align-items:center; gap:10px; font-size:1rem; }
        .badge {
          display:inline-block; padding:2px 8px; border-radius:8px;
          background:#f5eaea; border:1px solid var(--uch-border); margin-right:6px; font-size:0.8rem; color:#5a0000;
        }
        .src {font-size:0.9rem;color:#334}
        mark {background:#fff59d;padding:0 2px}
        .chunk {
          border:1px solid var(--uch-border);
          border-left:4px solid var(--uch-maroon);
          border-radius:10px; padding:12px; margin:6px 0; background:#fff;
        }
        .card { border:1px solid var(--uch-border); border-radius:12px; background:#fff; padding:12px; margin:8px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="topbar">
          <div class="brand">🛡️ <strong>University of Chicago · MS-ADS</strong> — RAG Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("# 🧠 MS-ADS RAG Assistant")
    st.write(
        "Ask questions about the University of Chicago MS in Applied Data Science program. "
        "The assistant retrieves relevant sections and generates a grounded answer."
    )

    # Question box
    q = st.text_input("Your question", placeholder="e.g., What are the core courses and can I study part-time online?")
    colA, colB = st.columns([1, 1])
    with colA:
        ask = st.button("Ask", type="primary")
    with colB:
        clear = st.button("Clear")

    if clear:
        st.session_state.history = []
        st.experimental_rerun()

    if ask and q.strip():
        run_id = str(uuid.uuid4())
        already_streamed = False

        if stream_mode:
            # Try streaming path
            try:
                gen, stream_citations = _build_streamer(q.strip(), top_k=top_k, temperature=temperature)
            except Exception as e:
                st.warning(f"Streaming unavailable, falling back to non-streaming: {e}")
                try:
                    resp = invoke_rag(q.strip(), top_k=top_k, temperature=temperature, max_tokens=max_tokens)
                except Exception as e2:
                    st.error(f"RAG pipeline error: {e2}")
                    return
            else:
                # ✅ Answer FIRST (streamed)
                st.write("## 🧠 Answer")
                placeholder = st.empty()
                buf: List[str] = []
                for token in gen():
                    buf.append(token)
                    placeholder.write("".join(buf))
                final_answer = "".join(buf)
                labels_inline = " ".join(f"[{i+1}]" for i in range(len(stream_citations)))
                placeholder.write(final_answer + (" " + labels_inline if labels_inline else ""))
                resp = RagResponse(answer=final_answer, citations=stream_citations)
                already_streamed = True

                # 📚 Sources AFTER the answer
                st.markdown("---")
                st.write("## 📚 Sources")
                for i, c in enumerate(stream_citations):
                    label = f"[{i+1}] {c.title}"
                    if c.url:
                        st.markdown(
                            f"**{label}**  ·  <span class='src'><a href='{c.url}' target='_blank'>{c.url}</a></span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**{label}**", unsafe_allow_html=True)
                    if show_chunks:
                        with st.expander("Show retrieved passage"):
                            query_terms = [t for t in re.split(r"\W+", q) if len(t) > 2]
                            st.markdown(f"<div class='chunk'>{_highlight(c.chunk, query_terms)}</div>", unsafe_allow_html=True)
                    if c.score is not None:
                        st.markdown(f"<span class='badge'>score {c.score:.3f}</span>", unsafe_allow_html=True)

        else:
            # Non-streaming
            try:
                resp = invoke_rag(q.strip(), top_k=top_k, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                st.error(f"RAG pipeline error: {e}")
                return

        # Render answer again only if not already printed via stream
        citation_labels = {i: _mk_citation_label(i) for i in range(len(resp.citations))}
        labels_inline = " ".join(citation_labels.values()) if resp.citations else ""
        if not already_streamed:
            st.write("## 🧠 Answer")
            st.write(resp.answer + (" " + labels_inline if labels_inline else ""))

        # Feedback row — visible for both modes
        fb_cols = st.columns([0.15, 0.15, 0.7])
        if fb_cols[0].button("👍 Helpful", key=f"up_{run_id}"):
            _save_feedback({"run_id": run_id, "question": q, "feedback": "up", "ts": time.time()})
            st.toast("Thanks for the feedback!", icon="👍")
        if fb_cols[1].button("👎 Not helpful", key=f"down_{run_id}"):
            _save_feedback({"run_id": run_id, "question": q, "feedback": "down", "ts": time.time()})
            st.toast("We’ll use this to improve.", icon="🛠️")

        # Sources panel for non-streaming (Answer is already above)
        if not stream_mode and resp.citations:
            st.markdown("---")
            st.write("## 📚 Sources")
            for i, c in enumerate(resp.citations):
                label = f"{_mk_citation_label(i)} {c.title}"
                if c.url:
                    st.markdown(
                        f"**{label}**  ·  <span class='src'><a href='{c.url}' target='_blank'>{c.url}</a></span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**{label}**", unsafe_allow_html=True)
                if show_chunks:
                    with st.expander("Show retrieved passage"):
                        query_terms = [t for t in re.split(r"\W+", q) if len(t) > 2]
                        st.markdown(f"<div class='chunk'>{_highlight(c.chunk, query_terms)}</div>", unsafe_allow_html=True)
                if c.score is not None:
                    st.markdown(f"<span class='badge'>score {c.score:.3f}</span>", unsafe_allow_html=True)

        # Debug box
        if show_debug:
            st.write("## Debug")
            st.json(
                {
                    "latency_ms": getattr(resp, "latency_ms", None),
                    "retrieved": len(resp.citations),
                    "openai": bool(os.getenv("OPENAI_API_KEY")),
                }
            )

        # Save history
        st.session_state.history.append(
            {
                "question": q,
                "response": resp.model_dump(),
                "params": {"top_k": top_k, "temperature": temperature, "max_tokens": max_tokens, "stream": stream_mode},
                "ts": time.time(),
            }
        )

    # Conversation History
    if st.session_state.history:
        st.markdown("---")
        st.write("### Conversation history")
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            st.markdown(f"**Q:** {h['question']}")
            st.markdown(f"**A:** {h['response']['answer']}")
            if h["response"].get("citations"):
                labs = ", ".join(_mk_citation_label(j) for j, _ in enumerate(h["response"]["citations"]))
                st.caption(f"Citations: {labs}")


if __name__ == "__main__":
    main()
