# app.py (version 2: with visualizations and enhanced NLP logic)
import streamlit as st
import pdfplumber
from io import BytesIO
import json
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt

# Do NOT import sentence-transformers at top-level; lazy-load it when needed
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Analyzer", layout="wide")

st.title("📄 Resume Analyzer & Job Match Tool")
st.write("This tool parses resumes, compares them against a job description, and visualizes the results.")

# --- Constants for Hybrid Score ---
SEMANTIC_SCORE_WEIGHT = 0.60
SKILL_SCORE_WEIGHT = 0.40

# Display any unexpected runtime errors in the UI
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
            if s.lower() in text_lower:
                found.add(s)
        # Note: The more complex fuzzy logic from the original was removed for clarity
        # but can be re-added if needed. This version focuses on direct matches.
        missing = set(skills_db) - found
        return sorted(list(found)), sorted(list(missing))

    # Word Cloud Generation
    def generate_wordcloud(text, title):
        if not text or not text.strip():
            return None
        try:
            wc = WordCloud(
                background_color="white", width=800, height=400,
                max_words=150, contour_width=3, contour_color='steelblue'
            ).generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(title, fontsize=16)
            ax.axis("off")
            return fig
        except ValueError:
            # Handles cases where the text has no words to plot
            return None


    # Lazy model helpers
    @st.cache_resource
    def load_embedding_model(name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)

    def embed_texts_with_model(model, texts):
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()))
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def compute_semantic_score(resume_text, job_desc_text):
        if not job_desc_text or not resume_text:
            return 0.0
        model = load_embedding_model()
        embs = embed_texts_with_model(model, [resume_text, job_desc_text])
        if embs.shape[0] < 2:
            return 0.0
        sim = cosine_similarity(embs[0].reshape(1, -1), embs[1].reshape(1, -1))[0][0]
        return max(0.0, float(sim)) # Return score between 0 and 1

    # --------------------
    # UI: Form for input
    # --------------------
    with st.form("upload_form", clear_on_submit=False):
        st.subheader("1. Upload Resumes and Job Description")
        uploaded_files = st.file_uploader(
            "Upload resume PDFs (you can upload multiple)", type=["pdf"], accept_multiple_files=True
        )
        job_desc = st.text_area(
            "Paste the job description here:", height=220,
            placeholder="e.g. Seeking Python developer with experience in Django, REST APIs, SQL..."
        )
        submitted = st.form_submit_button("Analyze & Match")

    # Process on submit
    if submitted:
        if not uploaded_files:
            st.warning("Please upload at least one PDF resume.")
        elif not job_desc.strip():
            st.warning("Please paste a job description.")
        else:
            with st.spinner("Analyzing... This may take a moment."):
                # 1. Parse all uploaded resumes
                parsed_resumes = []
                for f in uploaded_files:
                    raw_bytes = f.read()
                    text = extract_text_from_pdf_bytes(raw_bytes)
                    parsed_resumes.append({"filename": f.name, "text": text})

                # 2. Extract required skills from the job description
                job_text = job_desc or ""
                required_skills, _ = extract_skills(job_text, SKILLS_DB)
                if not required_skills:
                    st.warning("Could not identify any key skills from the job description. Skill matching will be based on the full skill database.")
                    # Fallback to full DB if JD is vague
                    required_skills = SKILLS_DB

                # 3. Analyze each resume
                st.info("A sentence-transformers model will be loaded to compute match scores. This may take ~30-120s on first run.")
                analysis_results = []
                for resume in parsed_resumes:
                    resume_text = resume.get("text", "") or ""
                    
                    # A. Skill-based analysis
                    found_skills, missing_skills = extract_skills(resume_text, required_skills)
                    skill_score = (len(found_skills) / len(required_skills) * 100) if required_skills else 0.0

                    # B. Semantic analysis
                    semantic_score = compute_semantic_score(resume_text, job_text) * 100 # as percentage

                    # C. Hybrid score calculation
                    hybrid_score = (semantic_score * SEMANTIC_SCORE_WEIGHT) + (skill_score * SKILL_SCORE_WEIGHT)

                    analysis_results.append({
                        "filename": resume.get("filename"),
                        "match_score": round(hybrid_score, 2),
                        "semantic_similarity (%)": round(semantic_score, 2),
                        "skill_overlap (%)": round(skill_score, 2),
                        "found_skills": ", ".join(found_skills) if found_skills else "None",
                        "missing_skills": ", ".join(missing_skills) if missing_skills else "None",
                    })

                # Save to session_state
                st.session_state["parsed_resumes"] = parsed_resumes
                st.session_state["analysis_results"] = analysis_results
                st.session_state["job_desc"] = job_text
                st.session_state["required_skills"] = required_skills


    # --------------------
    # UI: Display Results
    # --------------------
    if "analysis_results" in st.session_state:
        st.header("📊 Analysis Results")
        
        analysis = st.session_state["analysis_results"]
        job_text = st.session_state["job_desc"]
        required_skills = st.session_state["required_skills"]
        parsed = st.session_state["parsed_resumes"]

        df = pd.DataFrame(analysis)
        df_sorted = df.sort_values(by="match_score", ascending=False).reset_index(drop=True)

        # --- Summary View ---
        st.subheader("Ranked Resume Matches")
        
        # Display ranked chart
        chart = (
            alt.Chart(df_sorted)
            .mark_bar()
            .encode(
                x=alt.X("match_score:Q", title="Overall Match Score (%)"),
                y=alt.Y("filename:N", sort="-x", title="Resume"),
                tooltip=["filename", "match_score", "semantic_similarity (%)", "skill_overlap (%)"]
            ).properties(title="Resume Match Score Ranking")
        )
        st.altair_chart(chart, use_container_width=True)

        # Display ranked dataframe
        st.dataframe(
            df_sorted[[
                "filename", "match_score", "semantic_similarity (%)",
                "skill_overlap (%)", "found_skills", "missing_skills"
            ]],
            use_container_width=True
        )
        csv = df_sorted.to_csv(index=False).encode("utf-8")
        st.download_button("Download analysis as CSV", csv, "resume_match_results.csv", "text/csv")

        st.markdown("---")

        # --- Detailed View using Tabs ---
        st.subheader("Detailed Breakdown")
        
        tab_titles = [f"📄 {row['filename']}" for _, row in df_sorted.iterrows()]
        tabs = st.tabs(["📌 **Job Description**"] + tab_titles)

        # Job Description Tab
        with tabs[0]:
            st.markdown(f"**Skills required based on analysis:** `{', '.join(required_skills)}`")
            st.info("The word cloud below highlights the most frequent terms in the job description.")
            jd_wc_fig = generate_wordcloud(job_text, "Job Description Key Terms")
            if jd_wc_fig:
                st.pyplot(jd_wc_fig)
            with st.expander("Show full job description text"):
                st.text_area("", value=job_text, height=300)

        # Per-Resume Tabs
        for i, tab in enumerate(tabs[1:]):
            with tab:
                row = df_sorted.iloc[i]
                resume_data = next((r for r in parsed if r["filename"] == row["filename"]), None)
                
                st.markdown(f"### Results for: **{row['filename']}**")
                
                # Metrics
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Overall Match Score", f"{row['match_score']}%")
                mcol2.metric("Semantic Similarity", f"{row['semantic_similarity (%)']}%")
                mcol3.metric("Skill Overlap", f"{row['skill_overlap (%)']}%")

                # Word Cloud
                st.write("**Resume Word Cloud**")
                resume_wc_fig = generate_wordcloud(resume_data['text'], "Resume Key Terms")
                if resume_wc_fig:
                    st.pyplot(resume_wc_fig)
                else:
                    st.info("Not enough text to generate a word cloud for this resume.")
                
                # Skills Expander
                with st.expander("View Skill Analysis Details"):
                    st.success(f"**Found Skills:** {row['found_skills']}")
                    st.error(f"**Missing Skills:** {row['missing_skills']}")

                # Full Text Expander
                with st.expander("View Full Parsed Resume Text"):
                    st.text_area("", value=resume_data['text'], height=400)

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit, pdfplumber, rapidfuzz, sentence-transformers, and Altair.")

except Exception as e:
    # If anything unexpected happens, show the error in the app UI.
    st.error("An unexpected error occurred.")
    st.exception(e)