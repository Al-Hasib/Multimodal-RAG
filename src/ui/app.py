import streamlit as st
import requests
import json
import uuid
from datetime import datetime

API_URL = "http://localhost:8000"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "chat"

st.set_page_config(page_title="Multimodal RAG", layout="wide")


def auth_header():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def api_get(path, params=None):
    return requests.get(f"{API_URL}{path}", headers=auth_header(), params=params)


def api_post(path, json_data=None, files=None):
    return requests.post(f"{API_URL}{path}", headers=auth_header(), json=json_data, files=files)


def api_delete(path):
    return requests.delete(f"{API_URL}{path}", headers=auth_header())


# ── Auth pages ──────────────────────────────────────────────────────

def auth_page():
    st.title("Multimodal RAG")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                resp = api_post("/auth/login", json_data={"email": email, "password": password})
                if resp.ok:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    me = api_get("/auth/me")
                    if me.ok:
                        st.session_state.user = me.json()
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", {}).get("message", "Login failed"))

    with tab2:
        with st.form("register"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Register", use_container_width=True):
                resp = api_post("/auth/register", json_data={"email": email, "password": password, "name": name or None})
                if resp.ok:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    me = api_get("/auth/me")
                    if me.ok:
                        st.session_state.user = me.json()
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", {}).get("message", "Registration failed"))


# ── Chat page ───────────────────────────────────────────────────────

def chat_page():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Multimodal RAG")
    with col2:
        user = st.session_state.user
        if user:
            st.caption(f"👤 {user.get('name') or user.get('email')}")

    # Display messages with feedback
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "images" in msg and msg["images"]:
                with st.expander(f"Images ({len(msg['images'])})"):
                    for img in msg["images"]:
                        st.image(f"data:image/jpeg;base64,{img}")
            if msg["role"] == "assistant" and "message_id" in msg:
                fb_key = f"feedback_{msg['message_id']}"
                col_a, col_b, col_c = st.columns([1, 1, 6])
                existing_fb = st.session_state.get(fb_key)
                with col_a:
                    if st.button("👍", key=f"up_{msg['message_id']}", type="secondary" if existing_fb != 1 else "primary"):
                        api_post("/feedback", json_data={
                            "session_id": st.session_state.session_id,
                            "message_id": msg["message_id"],
                            "rating": 1,
                        })
                        st.session_state[fb_key] = 1
                        st.rerun()
                with col_b:
                    if st.button("👎", key=f"down_{msg['message_id']}", type="secondary" if existing_fb != -1 else "primary"):
                        api_post("/feedback", json_data={
                            "session_id": st.session_state.session_id,
                            "message_id": msg["message_id"],
                            "rating": -1,
                        })
                        st.session_state[fb_key] = -1
                        st.rerun()

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            try:
                params = {"question": prompt, "session_id": st.session_state.session_id}
                with api_get("/query/stream", params=params) as resp:
                    if resp.status_code == 401:
                        placeholder.error("Session expired. Please log in again.")
                        st.session_state.token = None
                        st.session_state.user = None
                        st.rerun()
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

        mid = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": full, "message_id": mid})


# ── Documents page ──────────────────────────────────────────────────

def documents_page():
    st.header("My Documents")

    col1, col2 = st.columns([3, 1])
    with col2:
        uploaded = st.file_uploader("Upload", type=["pdf", "png", "jpg", "jpeg", "docx", "html", "mp3", "wav", "m4a"])
        if uploaded and st.button("Ingest", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                resp = api_post("/ingest", files={"file": (uploaded.name, uploaded.getvalue())})
                if resp.ok:
                    st.success(f"Ingested: {uploaded.name}")
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", {}).get("message", "Ingestion failed"))

    resp = api_get("/documents", params={"skip": 0, "limit": 50})
    if resp.ok:
        data = resp.json()
        docs = data.get("documents", [])
        if not docs:
            st.info("No documents yet. Upload one to get started.")
        else:
            for doc in docs:
                with st.expander(f"{doc['filename']} ({doc['file_format']})"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Texts", doc.get("num_texts", 0))
                    c2.metric("Tables", doc.get("num_tables", 0))
                    c3.metric("Images", doc.get("num_images", 0))
                    st.caption(f"Uploaded: {doc.get('created_at')}")
                    if st.button("🗑️ Delete", key=f"del_{doc['id']}"):
                        del_resp = api_delete(f"/documents/{doc['id']}")
                        if del_resp.ok:
                            st.success("Deleted")
                            st.rerun()
                        else:
                            st.error("Delete failed")


# ── Sidebar ─────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        user = st.session_state.user
        if user:
            st.caption(f"👤 {user.get('name') or user.get('email')}")
            if st.button("Logout", use_container_width=True):
                st.session_state.token = None
                st.session_state.user = None
                st.rerun()
        else:
            if st.button("🔑 Login / Register", use_container_width=True):
                st.session_state.page = "auth"
                st.rerun()

        st.divider()

        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.page == "chat" else "secondary"):
            st.session_state.page = "chat"
            st.rerun()
        if st.button("📁 Documents", use_container_width=True, type="primary" if st.session_state.page == "documents" else "secondary"):
            st.session_state.page = "documents"
            st.rerun()

        st.divider()
        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


# ── Router ──────────────────────────────────────────────────────────

sidebar()

if not st.session_state.token:
    auth_page()
elif st.session_state.page == "documents":
    documents_page()
else:
    chat_page()
