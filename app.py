import hashlib
import json
import os

import streamlit as st
from dotenv import load_dotenv

from manual_agent import (
    describe_image_for_search,
    create_vector_index,
    answer_from_manual,
    extract_manual,
    rerank_passages,
    search_vector_index,
)


load_dotenv()
st.set_page_config(page_title="Bike Manual Troubleshooter", layout="wide")
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --ink: #e8efe9;
        --leaf: #4e926c;
        --paper: #101714;
        --panel: #19231e;
        --line: #3b5145;
        --copper: #e18a65;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(16, 23, 20, 0.92); }
    [data-testid="stSidebar"] { background: #17211c; }
    h1, h2, h3, p, label, [data-testid="stCaptionContainer"] { color: var(--ink); }
    input, textarea, [data-baseweb="select"] > div {
        background-color: var(--panel) !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }
    [data-testid="stFileUploader"] section {
        background: var(--panel);
        border-color: var(--line);
    }
    [data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
    }
    div.stButton > button { background: var(--leaf); color: white; border: 0; }
    div.stButton > button:hover { background: #397553; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Bike Manual Troubleshooter")
st.caption("Royal Enfield, TVS, and other makes | Manual-grounded answers | Cited sources")

with st.sidebar:
    st.subheader("Manuals")
    manual_files = st.file_uploader(
        "Add owner or service manuals",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    st.caption("PDF and plain-text manuals are supported. Scanned PDFs need selectable text.")

    api_key = os.getenv("SARVAM_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Sarvam API key", type="password")
    st.caption("The key is used for this session and is not saved by the app.")

if not manual_files:
    st.info("Add a bike owner or service manual to begin.")
    st.stop()

try:
    passages = extract_manual(manual_files)
except Exception as error:
    st.error(f"Could not read a manual: {error}")
    st.stop()

if not passages:
    st.warning("No selectable text was found in these manuals. Try a text-based PDF or TXT file.")
    st.stop()

passages_signature = json.dumps(passages, ensure_ascii=False, sort_keys=True)
index_fingerprint = hashlib.sha256(passages_signature.encode("utf-8")).hexdigest()
if st.session_state.get("manual_index_fingerprint") != index_fingerprint:
    try:
        with st.spinner("Embedding manual sections and building the vector index..."):
            st.session_state["manual_vector_index"] = create_vector_index(passages)
            st.session_state["manual_index_fingerprint"] = index_fingerprint
    except Exception as error:
        st.error(f"Could not build the semantic search index: {error}")
        st.stop()

vector_index = st.session_state["manual_vector_index"]
st.write(f"Loaded {len(manual_files)} manual(s) with {len(passages)} searchable sections.")

with st.form("troubleshooting_question"):
    question = st.text_area(
        "Describe the bike issue",
        placeholder="For example: White smoke is coming from the exhaust. What does the manual say?",
        height=100,
    )
    symptom_image = st.file_uploader(
        "Optional bike photo",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )
    submitted = st.form_submit_button("Find manual sections", type="primary")

if submitted:
    if not question.strip():
        st.warning("Describe the issue in text so the manual can be searched.")
        st.stop()

    search_query = question.strip()
    if symptom_image and api_key:
        try:
            image_notes = describe_image_for_search(
                api_key, symptom_image.getvalue(), question.strip()
            )
            search_query = f"{search_query}\nVisible details for manual search: {image_notes}"
        except Exception:
            st.warning("The photo could not be analyzed; searching from your text instead.")
    elif symptom_image:
        st.warning("Add a Sarvam API key to use the photo for manual search. Text search is still available.")

    candidates = search_vector_index(vector_index, search_query)
    if not candidates:
        st.warning("No sufficiently similar section was found in the uploaded manual. No answer was generated.")
        st.stop()

    selected = []
    if api_key:
        try:
            selected = rerank_passages(api_key, search_query, candidates)
        except Exception:
            st.warning("Sarvam could not rank the matches; showing the closest vector-search results.")

    if not selected:
        selected = candidates[:5]

    answer = None
    if api_key:
        try:
            answer = answer_from_manual(api_key, question.strip(), selected)
        except Exception:
            st.warning("Sarvam could not prepare a manual-grounded response. The source excerpts are available below.")

    if answer:
        st.subheader("Manual-based response")
        st.markdown(answer["answer"])
        passages_by_id = {passage["id"]: passage for passage in selected}
        citations = [
            passages_by_id[passage_id]["reference"]
            for passage_id in answer["passage_ids"]
        ]
        st.caption("Sources: " + "; ".join(citations))
    elif api_key:
        st.warning("The selected manual sections did not support a cited response. Review the source text below.")

    st.subheader("Source text from the manual")
    st.caption("These excerpts are copied from the uploaded manual for verification.")
    for passage in selected:
        with st.expander(passage["reference"], expanded=True):
            st.write(passage["text"])