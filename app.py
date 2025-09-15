# app.py
import streamlit as st
import pdfplumber
from io import BytesIO
import textwrap

st.set_page_config(page_title="Resume Analyzer (Skeleton)", layout="wide")

st.title("📄 Resume Analyzer — Skeleton")
st.write("Upload one or more resume PDFs and paste a job description. This skeleton extracts text from PDFs and displays it for further processing.")

# Sidebar instructions / options
st.sidebar.header("Instructions")
st.sidebar.write(
    textwrap.dedent(
        """
        1. Upload PDF resumes (.pdf).
        2. Paste the job description on the right.
        3. Click 'Extract' to parse PDFs and preview extracted text.
        Next: add skill extraction, embeddings, matching, visualizations.
        """
    )
)

# ---- Upload resumes ----
uploaded_files = st.file_uploader(
    "Upload resume PDFs (you can upload multiple)", type=["pdf"], accept_multiple_files=True
)

# ---- Job description input ----
st.subheader("Job Description")
job_desc = st.text_area(
    "Paste the job description here (or type a few bullet points):",
    height=240,
    placeholder="e.g. Seeking Python developer with experience in Django, REST APIs, SQL..."
)

# helper: extract text from a single PDF file-like object
def extract_text_from_pdf(file_like):
    try:
        text_chunks = []
        with pdfplumber.open(file_like) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
        full_text = "\n\n".join(text_chunks).strip()
        return full_text
    except Exception as e:
        return f"[ERROR] Could not extract PDF text: {e}"

# Cache extracted texts so reuploads / reruns are faster
@st.cache_data(show_spinner=False)
def parse_all_pdfs(file_list):
    results = []
    for f in file_list:
        # streamlit file_uploader returns an UploadedFile with .read()
        # convert to BytesIO so pdfplumber can reopen it later if cached
        raw = f.read()
        text = extract_text_from_pdf(BytesIO(raw))
        results.append({"filename": f.name, "text": text})
    return results

# ---- Extract button ----
if st.button("Extract"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        with st.spinner("Extracting text from PDFs..."):
            parsed = parse_all_pdfs(uploaded_files)

        # layout: left column for resumes, right column for job description + actions
        left_col, right_col = st.columns([2, 3])
        with left_col:
            st.subheader("Parsed Resumes")
            for i, r in enumerate(parsed):
                st.markdown(f"**{i+1}. {r['filename']}**")
                if r["text"].startswith("[ERROR]"):
                    st.error(r["text"])
                else:
                    # show first N characters and allow expand
                    preview = r["text"][:2000] + ("..." if len(r["text"]) > 2000 else "")
                    st.code(preview, language="text")
                    st.write("")  # spacing
                    if st.button(f"Show full text — {r['filename']}", key=f"full_{i}"):
                        st.text_area(f"Full text — {r['filename']}", value=r["text"], height=400)

        with right_col:
            st.subheader("Job Description (Preview)")
            if job_desc.strip() == "":
                st.info("Paste a job description above to compare against the resumes.")
            else:
                st.write(job_desc)
                st.markdown("---")
                st.write("Next steps (examples):")
                st.write(
                    "- Extract skills from resumes and JD using keyword matching or NER.\n"
                    "- Compute sentence embeddings for semantic matching (sentence-transformers).\n"
                    "- Show a match score and list missing skills."
                )

        # store parsed results in session state for later steps
        st.session_state["parsed_resumes"] = parsed

# If already parsed (from earlier run), show a smaller preview and an option to continue
elif uploaded_files and "parsed_resumes" in st.session_state:
    st.info("Resumes parsed in previous run. Click Extract to re-parse if you uploaded new files.")
    parsed = st.session_state["parsed_resumes"]
    for r in parsed:
        st.markdown(f"**{r['filename']}** — {len(r['text'])} characters")

# Footer / dev notes
st.markdown("---")
st.caption("Skeleton built with Streamlit + pdfplumber. Next: add skills DB, embeddings, and matching logic.")
