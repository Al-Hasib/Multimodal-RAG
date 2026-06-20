import streamlit as st
import requests
import json
import uuid

API_URL = "http://localhost:8000"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG")

with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Ingest"):
        with st.spinner("Processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            resp = requests.post(f"{API_URL}/ingest", files=files)
            if resp.ok:
                data = resp.json()
                st.success(f"Done! {data.get('texts', 0)} texts, {data.get('tables', 0)} tables, {data.get('images', 0)} images")
            else:
                st.error(f"Error: {resp.text}")

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            with st.expander(f"Images ({len(msg['images'])})"):
                for i, img in enumerate(msg["images"]):
                    st.image(f"data:image/jpeg;base64,{img}", caption=f"Image {i+1}")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            params = {"question": prompt, "session_id": st.session_state.session_id}
            with requests.get(f"{API_URL}/query/stream", params=params, stream=True) as resp:
                for line in resp.iter_lines():
                    if line:
                        data = line.decode("utf-8").replace("data: ", "")
                        if data == "[DONE]":
                            break
                        try:
                            token = json.loads(data)["token"]
                            full += token
                            placeholder.markdown(full + "▌")
                        except json.JSONDecodeError:
                            pass
            placeholder.markdown(full)
        except Exception as e:
            placeholder.error(f"Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full})
