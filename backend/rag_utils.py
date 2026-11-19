
from langchain_openai import ChatOpenAI
import os
import re

import math
from collections import Counter

# Words ignored by the keyword search
_STOPWORDS = set("""a an and are as at be by can could do does for from has have how i if in is it its
may me my of on or our so than that the their them then there these this to was we what when where
which who will with would you your am about any into not no""".split())

def _keyword_terms(text):
    """Lowercase words for keyword search; 'ash-mia' is also indexed as 'ash' and 'mia'."""
    terms = []
    for w in re.findall(r"[a-z0-9$]+(?:[-:][a-z0-9]+)*", text.lower()):
        if w in _STOPWORDS:
            continue
        terms.append(w)
        if "-" in w:
            terms += [p for p in w.split("-") if p not in _STOPWORDS]
    return terms

_bm25_cache = {}

def _bm25_index(vectorstore):
    """Build (once per vector store) a BM25 keyword index over the chunks stored in FAISS."""
    key = (id(vectorstore), vectorstore.index.ntotal)
    if key not in _bm25_cache:
        docs = [vectorstore.docstore.search(i) for i in vectorstore.index_to_docstore_id.values()]
        terms = [_keyword_terms(d.page_content) for d in docs]
        df = Counter(t for ts in terms for t in set(ts))
        n = len(docs)
        idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        avg_len = sum(len(ts) for ts in terms) / n
        _bm25_cache.clear()
        _bm25_cache[key] = (docs, [Counter(ts) for ts in terms], [len(ts) for ts in terms], idf, avg_len)
    return _bm25_cache[key]

def hybrid_search(query, k, vectorstore, rrf_k=60, k1=1.2, b=0.75, max_per_section=2):
    """Semantic (FAISS) + keyword (BM25) search, merged with Reciprocal Rank Fusion.
    At most `max_per_section` chunks come from the same policy section, so near-identical
    rows (e.g. the property matrix) cannot fill every slot."""
    docs, tfs, lens, idf, avg_len = _bm25_index(vectorstore)
    position = {d.page_content: i for i, d in enumerate(docs)}

    # Keyword ranking (BM25)
    q_terms = set(_keyword_terms(query))
    bm25 = []
    for i, (tf, length) in enumerate(zip(tfs, lens)):
        score = sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * length / avg_len))
                    for t in q_terms if t in tf)
        if score > 0:
            bm25.append((score, i))
    keyword_rank = [i for _, i in sorted(bm25, reverse=True)]

    # Semantic ranking (FAISS, all chunks)
    semantic_rank = [position[d.page_content] for d in vectorstore.similarity_search(query, k=len(docs))]

    # Reciprocal Rank Fusion
    fused = Counter()
    for ranking in (semantic_rank, keyword_rank):
        for rank, i in enumerate(ranking):
            fused[i] += 1 / (rrf_k + rank)

    # Take the best-ranked chunks, at most `max_per_section` from the same section
    results, per_section = [], Counter()
    for i, _ in fused.most_common():
        section = (docs[i].metadata.get("title"), docs[i].metadata.get("section"))
        if per_section[section] < max_per_section:
            results.append(docs[i])
            per_section[section] += 1
        if len(results) == k:
            break
    return results

# ---------------------------------------------------------------------------
# Candidate pool + LLM relevance filter
#  1. The question is rewritten in policy wording ("loud since midnight" -> "noise, quiet hours").
#  2. Hybrid search runs for the original AND the rewritten question (6 chunks each).
#  3. gpt-4o-mini reads the merged candidates and keeps only those needed to answer
#     the question (at most k), most useful first.
# Without an API key (before config.json is loaded), plain hybrid search is used.
# ---------------------------------------------------------------------------
REWRITE_PROMPT = """You turn a hotel guest's question into a search query for the hotel's internal policy documents.
Replace everyday words with the formal terms a written hotel policy would use for the same topic,
and add one or two closely related policy terms. Keep any property names, codes, room types, rate plans and amounts.
Return only the search query, at most 15 words.

Question: {query}"""

FILTER_PROMPT = """You select policy passages for a hotel staff assistant.
Question: {query}

Candidate passages:
{passages}

Return the numbers of the passages that contain information needed to answer the question
(including conditions, exceptions or the property named in the question). Leave out passages
that are only about other properties, other topics, or repeat information already selected.
Return at most {k} numbers, most useful first, as a comma-separated list (for example: 3, 1).
If none are useful, return the single most relevant one."""

CANDIDATES_PER_SEARCH = 6

def _helper_llm(max_tokens):
    """Small, deterministic gpt-4o-mini client for query rewriting and filtering."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=max_tokens,
                      api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_API_BASE"))

def rewrite_query(query):
    """Rewrite the guest question in the wording of the policy documents."""
    return _helper_llm(40).invoke(REWRITE_PROMPT.format(query=query)).content.strip()

def filter_relevant(query, candidates, k):
    """Ask the LLM which candidate passages are needed; return at most k of them."""
    passages = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(candidates))
    reply = _helper_llm(30).invoke(FILTER_PROMPT.format(query=query, passages=passages, k=k)).content
    chosen = []
    for number in re.findall(r"\d+", reply):
        i = int(number) - 1
        if 0 <= i < len(candidates) and i not in chosen:
            chosen.append(i)
    return [candidates[i] for i in chosen[:k]] or candidates[:1]

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 4, vectorstore=None) -> list[str]:
    if os.environ.get("OPENAI_API_KEY"):
        pool, seen = [], set()
        for text in (query, rewrite_query(query)):
            for d in hybrid_search(text, k=CANDIDATES_PER_SEARCH, vectorstore=vectorstore):
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    pool.append(d)
        results = filter_relevant(query, pool, k)
    else:
        results = hybrid_search(query, k=k, vectorstore=vectorstore)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly stated in the retrieved policy information.
    2. Do not use outside knowledge or make assumptions about hotel policies.
    3. If the query names a property (for example ASH-MIA Brickell), use only that property's details and do not mix in values for other properties.
    4. Quote exact figures: fees, times, deadlines and limits, together with the rate plan, room type or condition they apply to.
    5. Include every relevant condition, exception and approval requirement (for example "subject to availability" or "manager approval required").
    6. When the question needs more than one policy (for example a pet fee and the refund rules), combine them into one complete answer.
    7. Answer the specific question first, then add only the supporting details needed to act on it.
    8. Keep the answer concise, clear and in plain language that front-desk staff can repeat to a guest.
    9. Do not invent fees, refunds, exceptions or promises that are not in the retrieved information.
    10. If the retrieved information does not fully answer the question (for example the property or service is not listed),
        say clearly which information is not in the policy documents, give any general policy rule that does apply,
        and tell staff to check the live property profile or contact the property team instead of guessing.
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
