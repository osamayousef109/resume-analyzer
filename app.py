# app.py (robust, lazy-load embeddings, shows errors)
import streamlit as st
import pdfplumber
from io import BytesIO
import textwrap
import json
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# Do NOT import sentence-transformers at top-level; lazy-load it when needed
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Analyzer (robust)", layout="wide")

st.title("📄 Resume Analyzer — Robust (lazy model load + error display)")
st.write("This version lazy-loads the embedding model only when needed and shows any import/runtime errors in the UI.")

# Display any unexpected runtime errors in the UI so you don't get a blank page
try:
    # -------------------------
    # Helpers / Initialization
    # -------------------------

    # Load skills DB
    try:
        with open("skills.json", "r", encoding="utf-8") as fh:
            SKILLS_DB = json.load(fh)
    except Exception:
        SKILLS_DB = [
            "python","java","c++","sql","javascript","react","django","flask",
            "pytorch","tensorflow","nlp","machine learning","data structures","algorithms",
            "git","rest api","docker","kubernetes","aws","mysql","postgresql","html","css"
        ]

    # PDF text extraction
    def extract_text_from_pdf_bytes(raw_bytes):
        try:
            text_chunks = []
            with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_chunks.append(page_text)
            full_text = "\n\n".join(text_chunks).strip()
            return full_text
        except Exception as e:
            return f"[ERROR] Could not extract PDF text: {e}"

    # Skill extraction (exact + fuzzy)
    def extract_skills(text, skills_db=SKILLS_DB, fuzzy_threshold=75):
        text_lower = (text or "").lower()
        found = set()
        for s in skills_db:
            s_low = s.lower()
            if s_low in text_lower:
                found.add(s)
        remaining = [s for s in skills_db if s not in found]
        tokens = list(set([tok for tok in text_lower.replace("\n", " ").split() if len(tok) > 2]))
        detailed = []
        for s in remaining:
            score = fuzz.partial_ratio(s.lower(), text_lower)
            if score >= fuzzy_threshold:
                found.add(s)
                detailed.append((s, None, score))
            else:
                best = 0
                best_tok = None
                for tok in tokens:
                    sc = fuzz.ratio(s.lower(), tok)
                    if sc > best:
                        best = sc
                        best_tok = tok
                if best >= fuzzy_threshold:
                    found.add(s)
                    detailed.append((s, best_tok, best))
        missing = set(skills_db) - found
        return sorted(found), sorted(missing), detailed

    # Lazy model helpers
    @st.cache_resource
    def load_embedding_model(name="all-MiniLM-L6-v2"):
        # Import here to avoid heavy import at module load time
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)

    def embed_texts_with_model(model, texts):
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()))
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb

    def compute_match_score_lazy(resume_text, job_desc_text):
        # Called only when we need embeddings
        if not job_desc_text or not resume_text:
            return 0.0
        model = load_embedding_model()  # cached by streamlit
        embs = embed_texts_with_model(model, [resume_text, job_desc_text])
        if embs.shape[0] < 2:
            return 0.0
        sim = cosine_similarity(embs[0].reshape(1, -1), embs[1].reshape(1, -1))[0][0]
        sim = max(-1.0, min(1.0, float(sim)))
        return round((sim + 1) / 2 * 100, 2)

    # --------------------
    # UI: Form for input
    # --------------------
    with st.form("upload_form", clear_on_submit=False):
        st.subheader("Upload resumes and job description")
        uploaded_files = st.file_uploader(
            "Upload resume PDFs (you can upload multiple)", type=["pdf"], accept_multiple_files=True
        )
        job_desc = st.text_area(
            "Paste the job description here (or type bullet points):",
            height=220,
            placeholder="e.g. Seeking Python developer with experience in Django, REST APIs, SQL..."
        )
        submitted = st.form_submit_button("Extract & Analyze")

    # Process on submit
    if submitted:
        if not uploaded_files:
            st.warning("Please upload at least one PDF resume before clicking Extract & Analyze.")
        else:
            with st.spinner("Extracting text from PDFs..."):
                parsed = []
                for f in uploaded_files:
                    raw = f.read()
                    text = extract_text_from_pdf_bytes(raw)
                    parsed.append({"filename": f.name, "text": text})

            # perform analysis: skills extraction + matching score
            results = []
            job_text = job_desc or ""
            # If job_text is non-empty, we will load the embedding model (this may take time)
            will_compute_embeddings = bool(job_text.strip())
            if will_compute_embeddings:
                st.info("A sentence-transformers model will be loaded to compute match scores. This may take ~30-120s on first run.")
            for r in parsed:
                text = r.get("text", "") or ""
                found, missing, details = extract_skills(text)
                score = compute_match_score_lazy(text, job_text) if will_compute_embeddings else 0.0
                results.append({
                    "filename": r.get("filename"),
                    "match_score": score,
                    "found_skills": ", ".join(found) if found else "",
                    "missing_skills": ", ".join(missing) if missing else "",
                    "char_count": len(text)
                })

            # Save to session_state
            st.session_state["parsed_resumes"] = parsed
            st.session_state["analysis_results"] = results
            st.session_state["job_desc"] = job_text

    # Load from session_state if exists
    parsed = st.session_state.get("parsed_resumes", None)
    analysis = st.session_state.get("analysis_results", None)
    saved_job_desc = st.session_state.get("job_desc", "")

    # Layout: left resumes, right job + results
    left_col, right_col = st.columns([2, 3])
    with left_col:
        st.subheader("Parsed Resumes")
        if parsed:
            for i, r in enumerate(parsed):
                fname = r["filename"]
                st.markdown(f"**{i+1}. {fname}**")
                if r["text"].startswith("[ERROR]"):
                    st.error(r["text"])
                else:
                    preview = r["text"][:2000] + ("..." if len(r["text"]) > 2000 else "")
                    st.code(preview, language="text")
                    st.write("")
                    with st.expander(f"Show full text — {fname}", expanded=False):
                        st.text_area(f"Full text — {fname}", value=r["text"], height=400)
        else:
            st.info("No parsed resumes yet. Upload files and click Extract & Analyze to parse them.")

    with right_col:
        st.subheader("Job Description (Preview)")
        preview_jd = saved_job_desc if saved_job_desc else job_desc
        if not preview_jd:
            st.info("Paste a job description in the form and click Extract & Analyze.")
        else:
            st.write(preview_jd)
            st.markdown("---")
            st.write("Next steps: consider tuning the skills DB or adding NER and wordcloud visualizations.")

        # Ranked results
        if analysis:
            df = pd.DataFrame(analysis)
            df_sorted = df.sort_values(by="match_score", ascending=False).reset_index(drop=True)
            st.subheader("Ranked Resume Matches")
            st.dataframe(df_sorted[["filename", "match_score", "found_skills", "missing_skills"]], use_container_width=True)

            # Per-resume highlights
            st.subheader("Per-resume highlights")
            for row in analysis:
                st.markdown(f"**{row['filename']}** — score: {row['match_score']}%")
                st.write(f"**Found:** {row['found_skills'] or '—'}")
                missing_preview = ', '.join(row['missing_skills'].split(',')[:8]) if row['missing_skills'] else '—'
                st.write(f"**Missing (sample):** {missing_preview}")
                st.write("---")

            # CSV download
            csv = df_sorted.to_csv(index=False).encode("utf-8")
            st.download_button("Download analysis CSV", csv, file_name="resume_match_results.csv", mime="text/csv")

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit, pdfplumber, rapidfuzz, and sentence-transformers (lazy-loaded).")

except Exception as e:
    # If anything unexpected happens at import/run time, show the error in the app UI.
    st.error("An unexpected error occurred while running the app.")
    st.exception(e)
    # Also print to console for convenience
    import traceback
    traceback.print_exc()
