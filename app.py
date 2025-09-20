# app.py
import streamlit as st
import nltk

def ensure_nltk():
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
ensure_nltk()

from helper import (
    extract_text_from_pdf,
    build_vectorstore_from_text,
    get_extractive_answer,
    summarize_passages_with_embeddings,
    generate_mcqs_from_passages,
    remove_noise_from_chunk,
    generate_polished_summary,
    deduplicate_sentences,
)

st.set_page_config(page_title="Study Assistant - Polished + Heuristic", layout="wide")
st.title("📘 Study Assistant — Polished Answers + MCQs (Safe)")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded)

    if not text.strip():
        st.error("❌ Could not extract text from this PDF. Try another file.")
    else:
        st.success("✅ PDF loaded successfully")
        try:
            with st.spinner("Building vector store..."):
                store, chunks = build_vectorstore_from_text(text)
            st.info(f"Document processed into {len(chunks)} chunks.")
        except Exception as e:
            st.error("Error while building index. Try a smaller PDF.")
            st.exception(e)
            st.stop()

        query = st.text_input("Ask a question from the PDF (example: 'Explain AES SubBytes')")

        col1, col2 = st.columns([3,1])
        with col2:
            top_k = st.number_input("Top passages (k)", min_value=1, max_value=8, value=3, step=1)
            max_chars = st.number_input("Max chars in answer", min_value=200, max_value=5000, value=1200, step=100)
            summary_sentences = st.number_input("Summary sentences", min_value=1, max_value=6, value=3, step=1)
            num_mcqs = st.number_input("Generate MCQs", min_value=0, max_value=10, value=3, step=1)
            try_gen = st.checkbox("Try local generative polish (may need ~2+GB RAM)", value=False)

        if query.strip():
            with st.spinner("Searching..."):
                answer, results = get_extractive_answer(store, query, top_k=top_k, max_chars=max_chars)

            # show raw extractive reference but collapsed (user can expand)
            with st.expander("🔎 Extractive Answer (Top passages) — expand for full passages"):
                st.markdown(answer)

            # === Final concise non-repetitive answer ===
            from helper import build_final_short_answer  # local import (helper already loaded)
            final_answer = build_final_short_answer(store, results, max_words=160, bullets=True)
            if not final_answer.strip():
                st.info("Could not build a short answer. Try increasing top_k or uploading a richer PDF.")
            else:
                st.subheader("✅ Final Exam-Ready Answer (concise, deduplicated)")
                st.write(final_answer)   # already bullet-formatted if bullets=True

            # Optional: try generative polish (small model)
            if try_gen:
                with st.spinner("Polishing answer with local model (may fail on low RAM)..."):
                    # Use the final_answer (converted to paragraph) as base text
                    base_text = final_answer.replace("\n- ", " ").replace("\n", " ")
                    polished = generate_polished_summary(base_text, num_sentences=summary_sentences)
                st.subheader("✨ Polished Answer (model)")
                st.write(polished)

            # MCQs
            if num_mcqs > 0:
                with st.spinner("Generating MCQs (heuristic)..."):
                    mcqs = generate_mcqs_from_passages(results, num_questions=num_mcqs)
                st.subheader("📝 Auto-generated MCQs (check and edit)")
                if not mcqs:
                    st.info("No suitable sentences found to create MCQs. Try increasing top_k or uploading a more content-rich PDF.")
                else:
                    for i, q in enumerate(mcqs, 1):
                        st.markdown(f"**Q{i}.** {q['question']}")
                        for opt_idx, opt in enumerate(q["options"]):
                            label = chr(ord("A") + opt_idx)
                            st.write(f"{label}. {opt}")
                        with st.expander("Show answer and source sentence"):
                            st.write(f"**Answer:** {q['answer']}")
                            st.write(f"**Source sentence:** {q['source']}")
