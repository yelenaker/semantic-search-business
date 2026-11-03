"""
Comparative experiment: BM25 vs semantic retrieval with Sentence Transformers.
Run in Google Colab or a Python environment with internet access for model download.

Dataset: small manually curated/synthetic test collection, not real company data.
Gold relevance labels are initial researcher judgments for this pilot.
"""
import pandas as pd
import numpy as np
import time
import re
import math
from collections import Counter
from sentence_transformers import SentenceTransformer

docs = pd.read_csv("corpus.csv")
queries = pd.read_csv("queries.csv")
qrels_df = pd.read_csv("qrels.csv")
qrels = qrels_df.groupby("query_id")["doc_id"].apply(set).to_dict()

STOPWORDS = set("""a an the and or but if then of in on for to with by from as is are was were be can how what which that this these those their its""".split())

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOPWORDS]

corpus = (docs["title"] + ". " + docs["text"]).tolist()
tokenized_corpus = [tokenize(x) for x in corpus]

# BM25 (Okapi) implementation: k1=1.5, b=0.75
N = len(tokenized_corpus)
avgdl = sum(len(d) for d in tokenized_corpus) / N
df = Counter(term for doc in tokenized_corpus for term in set(doc))
def bm25_scores(query):
    q_terms = tokenize(query)
    scores = []
    for doc in tokenized_corpus:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q_terms:
            if term not in df:
                continue
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            f = tf[term]
            score += idf * (f * (1.5 + 1)) / (f + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
        scores.append(score)
    return np.array(scores)

# First run downloads the pretrained model weights.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)

def score_metrics(ranked_ids, relevant, k=5):
    top = ranked_ids[:k]
    precision_at_5 = sum(x in relevant for x in top) / k
    reciprocal_rank = next((1/(i+1) for i,x in enumerate(ranked_ids) if x in relevant), 0.0)
    return precision_at_5, reciprocal_rank

results = []
for _, row in queries.iterrows():
    qid, query = row["query_id"], row["query"]

    t0 = time.perf_counter()
    b_scores = bm25_scores(query)
    b_order = sorted(range(len(docs)), key=lambda i: (-b_scores[i], docs.iloc[i]["doc_id"]))
    bm25_ranked = docs.iloc[b_order]["doc_id"].tolist()
    bm25_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    s_scores = np.dot(doc_embeddings, q_emb[0])
    s_order = sorted(range(len(docs)), key=lambda i: (-s_scores[i], docs.iloc[i]["doc_id"]))
    semantic_ranked = docs.iloc[s_order]["doc_id"].tolist()
    semantic_ms = (time.perf_counter() - t0) * 1000

    relevant = qrels[qid]
    bp, br = score_metrics(bm25_ranked, relevant)
    sp, sr = score_metrics(semantic_ranked, relevant)

    results.append({
        "query_id": qid, "query": query,
        "BM25_top5": ", ".join(bm25_ranked[:5]),
        "Semantic_top5": ", ".join(semantic_ranked[:5]),
        "BM25_P@5": bp, "Semantic_P@5": sp,
        "BM25_MRR": br, "Semantic_MRR": sr,
        "BM25_search_ms": bm25_ms, "Semantic_search_ms": semantic_ms
    })

results = pd.DataFrame(results)
results.to_csv("comparison_results.csv", index=False)
print("\\nPER-QUERY RESULTS")
print(results.to_string(index=False))
metric_cols = ["BM25_P@5","Semantic_P@5","BM25_MRR","Semantic_MRR",
               "BM25_search_ms","Semantic_search_ms"]
print("\\nMEAN RESULTS")
print(results[metric_cols].mean().round(4).to_string())
