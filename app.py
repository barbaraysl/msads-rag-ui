"""
RAG UI — MS in Applied Data Science (Streamlit)
------------------------------------------------
Single‑file Streamlit app to interact with your RAG system.

🔧 How to use
1) pip install -r requirements (see list below)
2) Modify `invoke_rag()` to call your real retriever/LLM chain.
   - Two examples included: (A) LangChain style, (B) custom pipeline.
3) Run:  streamlit run app.py

📦 Minimal requirements (put these in requirements.txt if you’d like)
streamlit>=1.37
pydantic>=2.7
python-dotenv>=1.0

(Optional if you use them inside invoke_rag):
langchain
openai
sentence-transformers
faiss-cpu or chromadb

"""
import os, json, re, hashlib, requests, numpy as np, faiss
import streamlit as st
import uuid
import time
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------
# 🔌 Adapters & Data Models
# ---------------------------------------------------------------------
class Citation(BaseModel):
    title: str
    url: Optional[str] = None
    chunk: str
    score: Optional[float] = None  # retrieval/rerank score
    doc_id: Optional[str] = None

class RagResponse(BaseModel):
    answer: str
    citations: List[Citation]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    raw_debug: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------
# 
# ---------- Your MS-ADS backend (lifted from your scripts, lightly adapted) ----------
BASE = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
BASE_NETLOC = urlparse(BASE).netloc
BASE_PATH   = urlparse(BASE).path.rstrip("/") + "/"

CHUNKS_FILE = Path("msads_chunks.jsonl")
INDEX_FILE  = Path("msads_test.faiss")
META_FILE   = Path("msads_test_meta.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

USE_OPENAI = bool((os.getenv("OPENAI_API_KEY") or "").strip())
try:
    from openai import OpenAI as _OpenAI
    _openai_client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY") or None) if USE_OPENAI else None
except Exception:
    _openai_client = None

def within_msads(u):
    p = urlparse(u); return p.netloc == BASE_NETLOC and p.path.startswith(BASE_PATH)

def get_soup(url):
    r = requests.get(url, timeout=15); r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def normalize_text(s):
    s = re.sub(r"\s+", " ", s).strip()
    return s

def section_docs(url):
    soup = get_soup(url)
    headers = soup.find_all(["h1","h2","h3"])
    docs = []
    for h in headers:
        cur = []
        for sib in h.next_siblings:
            if getattr(sib, "name", None) in ["h1","h2","h3"]:
                break
            if getattr(sib, "name", None) in ["p","ul","ol","li","table"]:
                cur.append(sib.get_text(" ", strip=True))
        body = normalize_text(" ".join(cur))
        if not body or len(body) < 60:
            continue
        anchor = h.get("id") or re.sub(r"[^a-z0-9\-]+","-", h.get_text(strip=True).lower())
        docs.append({
            "url": url + f"#{anchor}",
            "page_url": url,
            "section_title": h.get_text(strip=True),
            "page_title": soup.title.get_text(strip=True) if soup.title else "",
            "level": h.name,
            "text": body
        })
    return docs

def crawl_sections(start=BASE, max_depth=2):
    from collections import deque
    q, seen_pages, docs = deque([(start,0)]), set(), []
    while q:
        u,d = q.popleft()
        if d>max_depth or u in seen_pages or not within_msads(u):
            continue
        seen_pages.add(u)
        try:
            docs.extend(section_docs(u))
        except Exception:
            pass
        soup = get_soup(u)
        for a in soup.find_all("a", href=True):
            v = urljoin(u, a["href"].split("#")[0])
            if within_msads(v) and v not in seen_pages:
                q.append((v, d+1))
    return docs

FAQ_HINTS = re.compile(r"\bfaq|frequently asked|questions?\b", re.I)

def _chunks(text, max_chars=1400, overlap=250):
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        yield text[i:j].strip()
        i = max(j - overlap, j)

@st.cache_resource(show_spinner=True)
def _load_or_build_index():
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    # Build chunks if missing
    if not CHUNKS_FILE.exists():
        sections = crawl_sections(BASE, max_depth=2)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as g:
            for r in sections:
                txt = r.get("text","") or ""
                is_faq = bool(FAQ_HINTS.search((r.get("section_title","") + " " + txt)[:300]))
                idx = 0
                for ch in _chunks(txt):
                    raw = (r.get("url","") + str(idx) + ch[:60]).encode("utf-8", errors="ignore")
                    cid = hashlib.md5(raw).hexdigest()[:16]
                    obj = {
                        "chunk_id": cid,
                        "text": ch,
                        "url": r.get("url",""),
                        "page_url": r.get("page_url",""),
                        "section_title": r.get("section_title",""),
                        "page_title": r.get("page_title",""),
                        "is_faq": is_faq,
                        "chunk_index": idx
                    }
                    g.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    idx += 1

    # Build FAISS if missing
    if not (INDEX_FILE.exists() and META_FILE.exists()):
        rows = [json.loads(l) for l in open(CHUNKS_FILE, "r", encoding="utf-8", errors="replace")]
        texts = [r["text"] for r in rows]
        batch, vecs = 128, []
        for i in range(0, len(texts), batch):
            embs = model.encode(texts[i:i+batch], convert_to_numpy=True, show_progress_bar=False)
            vecs.append(embs)
        X = np.vstack(vecs)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        index = faiss.IndexFlatIP(dim)
        index.add(X.astype("float32"))
        faiss.write_index(index, str(INDEX_FILE))
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f)

    index = faiss.read_index(str(INDEX_FILE))
    meta  = json.load(open(META_FILE, "r", encoding="utf-8"))
    return model, index, meta

@st.cache_data(show_spinner=False)
def _retrieve(query: str, top_k: int):
    model, index, meta = _load_or_build_index()
    qv = model.encode([query], convert_to_numpy=True)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
    sims, idxs = index.search(qv.astype("float32"), top_k)
    results = []
    for i, s in zip(idxs[0], sims[0]):
        rec = meta[i]
        results.append({
            "score": float(s),
            "text": rec["text"],
            "url": rec.get("url",""),
            "section": rec.get("section_title", rec.get("page_title",""))
        })
    return results

def _generate_answer(query, docs, temperature=0.1, max_tokens=800):
    context = "\n\n".join(f"[{d['section']}] {d['text']}" for d in docs)
    prompt = (
        "You are a helpful assistant answering questions only about the "
        "University of Chicago MS in Applied Data Science program.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    if USE_OPENAI and _openai_client is not None:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content.strip()
    return "🔹 Retrieval only (no LLM). Set OPENAI_API_KEY to enable generation."

# ---------------------------------------------------------------------

def invoke_rag(
    question: str,
    *,
    top_k: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 800,
    retrieve_only: bool = False,
) -> RagResponse:
    start = time.time()
    docs = _retrieve(question, top_k=top_k)
    citations = [
        Citation(
            title=d.get("section") or "Source",
            url=d.get("url"),
            chunk=d.get("text",""),
            score=d.get("score"),
        )
        for d in docs
    ]
    if retrieve_only:
        answer = "🔍 Retrieved top sections only (generation disabled)."
    else:
        answer = _generate_answer(question, docs, temperature=temperature, max_tokens=max_tokens)
    end = time.time()
    return RagResponse(
        answer=answer,
        citations=citations,
        latency_ms=int((end - start) * 1000),
        raw_debug={"retrieved": len(docs), "openai": bool(USE_OPENAI)}
    )

   
    # --------------------------
    # TEMPLATE (A): LangChain
    # --------------------------
    # from langchain.chat_models import ChatOpenAI
    # from langchain.prompts import ChatPromptTemplate
    # retriever = ...  # your retriever, already configured
    # docs = retriever.get_relevant_documents(question)[:top_k]
    # context = "\n\n".join(d.page_content for d in docs)
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "Answer using only the context. Cite sources as [n]. If unsure, say you don't know."),
    #     ("human", f"Question: {question}\n\nContext:\n{context}")
    # ])
    # llm = ChatOpenAI(model="gpt-4o", temperature=temperature, max_tokens=max_tokens)
    # ai_msg = llm.invoke(prompt)
    # answer_text = ai_msg.content
    # citations = [Citation(title=d.metadata.get("title","Source"), url=d.metadata.get("source"), chunk=d.page_content, score=d.metadata.get("score")) for d in docs]
    # return RagResponse(answer=answer_text, citations=citations)

    # --------------------------
    # TEMPLATE (B): Custom pipeline
    # --------------------------
    # pipeline = get_pipeline_somehow()
    # retrieved = pipeline.retrieve(question, top_k=top_k)
    # reranked = pipeline.rerank(retrieved)
    # answer_text, usage = pipeline.generate(question, reranked, temperature=temperature, max_tokens=max_tokens)
    # citations = [Citation(**c) for c in pipeline.to_citations(reranked)]
    # return RagResponse(answer=answer_text, citations=citations, prompt_tokens=usage.get("prompt"), completion_tokens=usage.get("completion"))

    # If neither template is wired, raise with instructions.
    

# ---------------------------------------------------------------------
# 🧰 Utilities
# ---------------------------------------------------------------------

def _highlight(text: str, terms: List[str]) -> str:
    if not terms:
        return text
    # Escape regex and highlight whole words (case-insensitive)
    for t in sorted(set([t for t in terms if t]), key=len, reverse=True):
        pat = re.compile(rf"(\b{re.escape(t)}\b)", re.IGNORECASE)
        text = pat.sub(r"<mark>\1</mark>", text)
    return text


def _mk_citation_label(i: int) -> str:
    return f"[{i+1}]"


def _save_feedback(row: Dict[str, Any]):
    try:
        fn = "feedback_log.jsonl"
        with open(fn, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------
# 🎨 Streamlit UI
# ---------------------------------------------------------------------

def main():
    load_dotenv()
    st.set_page_config(page_title="MS‑ADS RAG Assistant", page_icon="🧠", layout="wide")

    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts

    # Sidebar — controls
    with st.sidebar:
        st.title("⚙️ Settings")
        st.caption("Tune retrieval & generation parameters")
        top_k = st.slider("Top‑K results", min_value=1, max_value=10, value=4)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
        max_tokens = st.slider("Max tokens", 200, 2000, 800, 50)
        show_chunks = st.toggle("Show retrieved chunks", value=True)
        show_debug = st.toggle("Show debug info", value=False)
        st.markdown("---")
        demo = st.toggle("Demo mode (RAG_DEMO_MODE)", value=os.getenv("RAG_DEMO_MODE", "0") == "1")
        if demo:
            os.environ["RAG_DEMO_MODE"] = "1"
        else:
            os.environ["RAG_DEMO_MODE"] = "0"
        st.markdown("---")
        st.caption("💾 Export")
        if st.button("Download conversation as JSON"):
            data = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
            st.download_button("Save JSON", data=data, file_name="msads_rag_history.json")

    # Header
    st.markdown(
        """
        <style>
        .badge {display:inline-block;padding:2px 8px;border-radius:8px;background:#eef;border:1px solid #ccd;margin-right:6px;font-size:0.8rem}
        .src {font-size:0.9rem;color:#334}
        mark {background: #fff59d; padding:0 2px}
        .chunk {border:1px solid #eee;border-radius:10px;padding:12px;margin:6px 0;background:#fafafa}
        .muted {color:#666}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.write("# 🧠 MS‑ADS RAG Assistant")
    st.write(
        "Ask questions about the University of Chicago MS in Applied Data Science program. "
        "The assistant retrieves the most relevant program sections and generates a grounded answer."
    )

    # Question box
    q = st.text_input("Your question", placeholder="e.g., What are the core courses and can I study part‑time online?")
    colA, colB = st.columns([1,1])
    with colA:
        ask = st.button("Ask", type="primary")
    with colB:
        clear = st.button("Clear")

    if clear:
        st.session_state.history = []
        st.experimental_rerun()

    if ask and q.strip():
        run_id = str(uuid.uuid4())
        try:
            resp = invoke_rag(q.strip(), top_k=top_k, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            st.error(f"RAG pipeline error: {e}")
            return

        # Render answer with inline numeric citations
        citation_labels = {i: _mk_citation_label(i) for i in range(len(resp.citations))}
        labels_inline = " ".join(citation_labels.values()) if resp.citations else ""
        st.write("## Answer")
        st.write(resp.answer + (" " + labels_inline if labels_inline else ""))

        # Feedback row
        fb_cols = st.columns([0.15, 0.15, 0.7])
        if fb_cols[0].button("👍 Helpful", key=f"up_{run_id}"):
            _save_feedback({"run_id": run_id, "question": q, "feedback": "up", "ts": time.time()})
            st.toast("Thanks for the feedback!", icon="👍")
        if fb_cols[1].button("👎 Not helpful", key=f"down_{run_id}"):
            _save_feedback({"run_id": run_id, "question": q, "feedback": "down", "ts": time.time()})
            st.toast("We’ll use this to improve.", icon="🛠️")

        # Sources & chunks
        if resp.citations:
            st.write("## Sources")
            for i, c in enumerate(resp.citations):
                label = citation_labels[i]
                meta = f"{label} {c.title}"
                if c.url:
                    st.markdown(f"**{meta}**  ·  <span class='src'><a href='{c.url}' target='_blank'>{c.url}</a></span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{meta}**", unsafe_allow_html=True)
                if show_chunks:
                    with st.expander("Show retrieved passage"):
                        query_terms = [t for t in re.split(r"\W+", q) if len(t) > 2]
                        st.markdown(
                            f"<div class='chunk'>{_highlight(c.chunk, query_terms)}</div>",
                            unsafe_allow_html=True,
                        )
                if c.score is not None:
                    st.markdown(f"<span class='badge'>score {c.score:.3f}</span>", unsafe_allow_html=True)

        if show_debug:
            st.write("## Debug")
            st.json({
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "latency_ms": resp.latency_ms,
                "raw_debug": resp.raw_debug,
            })

        # Save to history
        st.session_state.history.append({
            "question": q,
            "response": resp.model_dump(),
            "params": {
                "top_k": top_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "ts": time.time(),
        })

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
