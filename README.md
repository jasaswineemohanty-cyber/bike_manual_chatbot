# Bike Manual Troubleshooter

A small Streamlit assignment that searches uploaded bike owner/service manuals and answers from the retrieved passages. A multilingual FastEmbed model creates passage and query embeddings, Chroma performs cosine-similarity search, and Gemma 4 31B served through Sarvam re-ranks the closest passages and rephrases supported information with passage citations. Original manual excerpts remain visible for verification. If no sufficiently similar or supporting passage is found, it abstains.

Optional bike photos are analyzed with Sarvam's `gemma4` image input only to add visible terms to the manual search. Image analysis is not presented as a diagnosis or answer.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set `SARVAM_API_KEY`, or enter the key in the sidebar.
4. Start the app with `streamlit run app.py`.

The first run downloads the multilingual embedding model. The Chroma vector index is held in memory for the current Streamlit session and is rebuilt when the uploaded manual set changes; it is not persisted to disk. The Sarvam key needs access to the v2 beta chat completion endpoint and Gemma 4. Without a key, semantic text search still returns literal manual passages; Sarvam reranking and image-assisted search are unavailable.

## Supported documents

Text-based PDF and UTF-8 TXT files are supported. Scanned/image-only PDFs have no selectable text and are not OCR'd in this assignment. Page numbers are included in passage references.

## Grounding behavior

The model may select only IDs from locally retrieved manual sections. The answer-generation response must cite IDs from the selected passages; malformed responses or unsupported citations are rejected. The model is instructed to use only those passages and avoid outside knowledge. Since generated paraphrases cannot be proven factually entailed by citation IDs alone, the source excerpts remain visible for verification. Image description is used only as an additional search query and is not shown as an answer.