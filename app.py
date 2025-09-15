# app.py
import streamlit as st
import pdfplumber
from io import BytesIO
import textwrap

st.set_page_config(page_title="Resume Analyzer (Fixed Form)", layout="wide")

st.title("📄 Resume Analyzer — Form + Expander Fix")
st.write("Upload one or more resume PDFs and paste a job description. Use the Extract button in the form to parse PDFs.")

# Sidebar instructions / options
st.sidebar.header("Instructions")
st.sidebar.write(
    textwrap.dedent(
        """
        1. Use the form to upload PDF resumes and paste the job description.
        2. Click 'Extract' (the form submit button) to parse PDFs and preview extracted text.
        3. Use expanders to view full resume text.
        """
    )
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
def parse_all_pdfs(file_bytes_list):
    results = []
    for name, raw in file_bytes_list:
        text = extract_text_from_pdf(BytesIO(raw))
        results.append({"filename": name, "text": text})
    return results

# -----------------------
# Form: uploads + job desc
# -----------------------
with st.form("upload_form", clear_on_submit=False):
    st.subheader("Upload resumes and job description")
    uploaded_files = st.file_uploader(
        "Upload resume PDFs (you can upload multiple)", type=["pdf"], accept_multiple_files=True
    )
    job_desc = st.text_area(
        "Paste the job description here (or type a few bullet points):",
        height=200,
        placeholder="e.g. Seeking Python developer with experience in Django, REST APIs, SQL..."
    )
    submitted = st.form_submit_button("Extract")

# When the form is submitted, parse and store results
if submitted:
    if not uploaded_files:
        st.warning("Please upload at least one PDF resume before clicking Extract.")
    else:
        with st.spinner("Extracting text from PDFs..."):
            # read bytes for each file so we can cache by bytes
            file_bytes_list = []
            for f in uploaded_files:
                raw = f.read()
                file_bytes_list.append((f.name, raw))
            parsed = parse_all_pdfs(file_bytes_list)

        # store parsed results in session state
        st.session_state["parsed_resumes"] = parsed
        st.session_state["job_desc"] = job_desc

# If parsed results already exist in session state, load them
parsed = st.session_state.get("parsed_resumes", None)
saved_job_desc = st.session_state.get("job_desc", "")

# Layout for parsed results and job description
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
                st.write("")  # spacing
                # Use expander to show full text reliably
                with st.expander(f"Show full text — {fname}", expanded=False):
                    st.text_area(f"Full text — {fname}", value=r["text"], height=400)
    else:
        st.info("No parsed resumes yet. Upload files and click Extract to parse them.")

with right_col:
    st.subheader("Job Description (Preview)")
    if parsed:
        # show job description from session state if available, otherwise from form
        preview_jd = saved_job_desc if saved_job_desc else job_desc
        if not preview_jd:
            st.info("Paste a job description in the form (above) and click Extract.")
        else:
            st.write(preview_jd)
            st.markdown("---")
            st.write("Next steps (examples):")
            st.write(
                "- Extract skills from resumes and JD using keyword matching or NER.\n"
                "- Compute sentence embeddings for semantic matching (sentence-transformers).\n"
                "- Show a match score and list missing skills."
            )
    else:
        st.info("After extracting resumes, the job description preview will appear here.")

# Footer / dev notes
st.markdown("---")
st.caption("Skeleton built with Streamlit + pdfplumber. Next: add skills DB, embeddings, and matching logic.")
