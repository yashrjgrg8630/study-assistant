# helper.py
import os
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain.text_splitter import CharacterTextSplitter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import random
from typing import List, Tuple

# Try to ensure punkt & tagger available; calling code should also ensure this
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    try:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    except Exception:
        nltk.download("averaged_perceptron_tagger", quiet=True)

# Embedding model (fast & small)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def extract_text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            # add page marker (optional)
            text += f"\n\n[PAGE {page_idx}]\n" + page_text + "\n"
    return text

def split_text(text: str, chunk_size=800, chunk_overlap=100) -> List[str]:
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

class SimpleVectorStore:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None
        self.texts = []

    def build(self, texts: List[str], batch_size=64):
        self.texts = texts
        # encode in batches to reduce memory spikes
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_emb.append(emb)
        emb = np.vstack(all_emb).astype("float32")
        faiss.normalize_L2(emb)
        self.embeddings = emb
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(emb)
        self.index = index

    def query(self, query_text: str, top_k=3):
        q_emb = self.model.encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({"score": float(score), "text": self.texts[int(idx)], "idx": int(idx)})
        return results

def build_vectorstore_from_text(text: str, chunk_size=800, chunk_overlap=100) -> Tuple[SimpleVectorStore, List[str]]:
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = SimpleVectorStore()
    store.build(chunks)
    return store, chunks

def get_extractive_answer(store: SimpleVectorStore, query: str, top_k=3, max_chars=1200):
    results = store.query(query, top_k=top_k)
    passages = []
    for r in results:
        cleaned = remove_noise_from_chunk(r["text"])
        passages.append(f"(score: {r['score']:.3f})\n{cleaned}\n")
    answer = "\n---\n".join(passages)
    if len(answer) > max_chars:
        answer = answer[:max_chars] + "... (truncated)"
    return answer, results

# ------------------ Noise removal / heuristic cleaner ------------------
def remove_noise_from_chunk(text: str) -> str:
    """
    Heuristic cleaner: remove repeated headers/footers, page markers, multiple spaces,
    common short phrases like author names, and numeric page references.
    """
    t = text
    # remove bracketed page markers like [PAGE 3]
    t = re.sub(r"\[PAGE\s*\d+\]", " ", t, flags=re.IGNORECASE)
    # remove sequences like "Dr. Name ... Assistant Professor" (common in headers) -> remove short title blocks
    t = re.sub(r"\bDr\.[^\n]{0,80}\n", " ", t)
    t = re.sub(r"\bAssistant Professor[^\n]{0,80}", " ", t)
    # remove long numeric references like "12 Cryptography and Network Security"
    t = re.sub(r"\d+\s+Cryptography and Network Security", " ", t, flags=re.IGNORECASE)
    # remove multiple spaces and newlines
    t = re.sub(r"\s{2,}", " ", t)
    t = t.strip()
    # optionally, remove lines that are too short and look like headings (<=3 words)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    filtered = []
    for ln in lines:
        if len(ln.split()) <= 3 and re.search(r"[A-Za-z]{2,}", ln):
            # treat as heading: skip unless includes colon or keyword
            if ":" in ln or any(k in ln.lower() for k in ["definition", "theory", "purpose"]):
                filtered.append(ln)
            else:
                continue
        else:
            filtered.append(ln)
    return " ".join(filtered)

# ------------------ Extractive summarizer (centroid sentence pick) ------------------
def summarize_passages_with_embeddings(store: SimpleVectorStore, results, num_sentences=3):
    combined = "\n".join([remove_noise_from_chunk(r["text"]) for r in results])
    sentences = sent_tokenize(combined)
    if len(sentences) <= num_sentences:
        return combined
    sent_emb = store.model.encode(sentences, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(sent_emb)
    centroid = np.mean(sent_emb, axis=0, keepdims=True)
    sims = (sent_emb @ centroid.T).squeeze()
    top_idx = np.argsort(-sims)[:num_sentences]
    top_idx_sorted = sorted(top_idx.tolist())
    summary = " ".join([sentences[i] for i in top_idx_sorted])
    # small cleanup
    summary = re.sub(r"\s{2,}", " ", summary).strip()
    return summary

# ------------------ Simple MCQ Generator (heuristic) ------------------
def generate_mcqs_from_passages(results, num_questions=5, max_options=4):
    combined = " ".join([remove_noise_from_chunk(r["text"]) for r in results])
    sentences = sent_tokenize(combined)
    candidate_sentences = []
    for s in sentences:
        tokens = word_tokenize(s)
        tags = pos_tag(tokens, lang="eng")
        nouns = [w for (w, t) in tags if t.startswith("NN")]
        if len(nouns) >= 1 and len(s.split()) >= 6:
            candidate_sentences.append((s, nouns))
    if not candidate_sentences:
        return []
    mcqs = []
    used = set()
    idx = 0
    while len(mcqs) < num_questions and idx < len(candidate_sentences):
        s, nouns = candidate_sentences[idx]
        idx += 1
        nouns_sorted = sorted(nouns, key=lambda x: len(x), reverse=True)
        answer = nouns_sorted[0]
        if answer.lower() in used:
            continue
        used.add(answer.lower())
        if answer in s:
            question_text = s.replace(answer, "_____", 1)
        else:
            words = s.split()
            question_text = " ".join(words[:-1]) + " _____."
        all_other_nouns = []
        for (ss, ns) in candidate_sentences:
            for n in ns:
                if n.lower() != answer.lower():
                    all_other_nouns.append(n)
        distractors = []
        pool = list(set([d for d in all_other_nouns if d.lower() != answer.lower() and len(d) >= 2]))
        random.shuffle(pool)
        for d in pool:
            if len(distractors) >= max_options - 1:
                break
            distractors.append(d)
        i_fallback = 0
        while len(distractors) < max_options - 1 and i_fallback < len(s.split()):
            w = s.split()[i_fallback]
            if w.lower() != answer.lower() and w.isalpha() and w not in distractors:
                distractors.append(w)
            i_fallback += 1
        options = [answer] + distractors[: max_options - 1]
        random.shuffle(options)
        mcqs.append({
            "question": question_text,
            "options": options,
            "answer": answer,
            "source": s
        })
    return mcqs

# ------------------ Generative polishing (safe fallback) ------------------
def generate_polished_summary(text: str, num_sentences=3, model_name="google/flan-t5-small", device=-1):
    """
    Try to load a small local text2text model and paraphrase the extractive summary.
    If model load or generation fails (memory/other), return the input extractive text plus a notice.
    device=-1 uses CPU. If you have a GPU and torch can use it, set device to 0 manually.
    """
    # small prompt template
    prompt = f"Summarize the following text into {num_sentences} concise exam-style bullet points or a short paragraph:\n\n{text}\n\nSummary:"
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        # load tokenizer + model (may consume memory)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
        out = gen(prompt, max_length=256, do_sample=False)
        if isinstance(out, list) and len(out) > 0:
            return out[0].get("generated_text", out[0].get("summary_text", str(out[0])))
        else:
            return text
    except Exception as e:
        # fallback: return the input text but cleaned
        fallback = re.sub(r"\s{2,}", " ", text).strip()
        notice = "(polishing model unavailable or memory-limited — returning cleaned extractive summary)\n\n"
        return notice + fallback


# ------------------ Deduplicate sentences ------------------
def deduplicate_sentences(text: str) -> str:
    """
    Remove repeated sentences across passages.
    Keeps first occurrence, drops duplicates.
    """
    sentences = sent_tokenize(text)
    seen = set()
    unique = []
    for s in sentences:
        s_clean = re.sub(r"\s+", " ", s.strip().lower())
        if s_clean not in seen and len(s_clean) > 0:
            seen.add(s_clean)
            unique.append(s.strip())
    return " ".join(unique)


# ------------------ NEW HELPERS: deduplicate + build short final answer ------------------
# Updated per your request: replaced deduplicate_sentences_from_text implementation
# and changed build_final_short_answer tail to keep only first ~6 sentences.

def deduplicate_sentences_from_text(text: str) -> str:
    """
    Remove repeated or noisy sentences across passages.
    - Drops duplicates (case-insensitive)
    - Removes boilerplate course references (e.g., "Already Covered in Unit")
    - Skips very short or near-duplicate sentences
    """
    sentences = sent_tokenize(text)
    seen = set()
    unique = []
    for s in sentences:
        s_clean = re.sub(r"\s+", " ", s).strip()

        # Drop noisy headers/footers / common boilerplate
        if re.search(r"(already covered in unit|assistant professor|upes|cryptography and network security)", s_clean, re.I):
            continue
        if len(s_clean.split()) < 4:  # skip too short
            continue

        # Normalize for deduplication
        norm = s_clean.lower()
        norm = re.sub(r"shannon.?s theory", "shannon theory", norm)  # normalize repeated phrasing
        norm = re.sub(r"\W+", " ", norm).strip()

        if norm in seen:
            continue
        seen.add(norm)
        unique.append(s_clean)
    return " ".join(unique)


def build_final_short_answer(store: SimpleVectorStore, results, max_words=160, bullets=True):
    """
    Build a short exam-ready answer:
    1. Collect top passages -> clean -> dedupe sentences
    2. Score sentences by embedding similarity to centroid of top passages
       (so we pick the most representative sentences)
    3. Keep sentences until `max_words` reached
    4. Return as short paragraph or bullet list
    """
    # combine cleaned passages (use remove_noise_from_chunk for initial cleaning)
    cleaned_passages = []
    for r in results:
        try:
            cleaned = remove_noise_from_chunk(r["text"])
        except Exception:
            # fallback: use raw text but remove page markers
            cleaned = re.sub(r'\[PAGE\s*\d+\]', ' ', r.get("text", ""))
            cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        cleaned_passages.append(cleaned)
    combined = " ".join(cleaned_passages)

    # dedupe first to remove exact repeats
    deduped = deduplicate_sentences_from_text(combined)

    # split into sentences
    sents = sent_tokenize(deduped)
    if not sents:
        return ""

    # compute embeddings for sentences and centroid to pick representative ones
    order = list(range(len(sents)))
    try:
        emb = store.model.encode(sents, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        centroid = emb.mean(axis=0, keepdims=True)
        sims = (emb @ centroid.T).squeeze()
        # sort sentences by similarity to centroid (higher = more central)
        order = list(reversed(sims.argsort().tolist()))  # highest first
    except Exception:
        # fallback: keep sentence order
        order = list(range(len(sents)))

    # select sentences until max_words hit, prefer highest scoring sentences but preserve original order
    chosen = []
    chosen_set = set()
    words = 0
    for idx in order:
        s = sents[idx].strip()
        s_norm = re.sub(r'\s+', ' ', s).strip()
        if s_norm.lower() in chosen_set:
            continue
        s_words = len(s_norm.split())
        # skip tiny headings
        if s_words <= 3:
            continue
        # add if it doesn't blow the budget (but allow one longer if nothing chosen)
        if words + s_words > max_words and len(chosen) > 0:
            continue
        chosen.append((idx, s_norm))
        chosen_set.add(s_norm.lower())
        words += s_words
        if words >= max_words:
            break

    # sort chosen by their original order in document
    chosen.sort(key=lambda x: x[0])
    final_sentences = [s for (_, s) in chosen]

    # keep only first 5-6 sentences for conciseness
    final_sentences = final_sentences[:6]

    if bullets:
        bullets_text = "\n".join([f"- {s}" for s in final_sentences])
        return bullets_text
    else:
        return " ".join(final_sentences)
