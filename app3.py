import os
import re
import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient

# ====== HF Token from ENV ======
HF_API_KEY = "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"

# ====== Mistral-7B Inference Client ======
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
hf_client = InferenceClient(provider="hf-inference", api_key=HF_API_KEY)

# ====== Load CSV Files (Text & Video Data) ======
try:
    algo_df = pd.read_csv("algorithm_info.csv")
except Exception:
    algo_df = pd.DataFrame({
        "Title": ["Radix Sort"],
        "Description": ["Sorts numbers by processing individual digits."],
        "Steps": ["1. Group elements", "2. Sort each group"],
        "Code Example": ["def radix_sort(arr):\n    pass"],
        "Time Complexity (Worst)": ["O(nk)"],
        "Space Complexity": ["O(n+k)"]
    })

try:
    video_df = pd.read_csv("videos.csv")
except Exception:
    video_df = pd.DataFrame({
        "Topic": ["Radix Sort"],
        "VideoLink": ["https://www.youtube.com/embed/dQw4w9WgXcQ"]
    })

content_mode = "text"  # Can be switched to "images" or "videos"

# ====== Inference & Retrieval Logic ======

def extract_keywords(query):
    prompt = (
        f"Extract relevant keywords from the following algorithm query: '{query}'. "
        "Return a comma-separated list of keywords."
    )
    keywords_text = hf_client.text_generation(
        prompt,
        model=MODEL_ID,
        max_new_tokens=64,
        temperature=0.3,
        do_sample=False
    ).strip()
    return [kw.strip() for kw in keywords_text.split(",") if kw.strip()]

def retrieve_text_data(topic):
    keywords = extract_keywords(topic)
    best_row, best_score = None, 0
    for _, row in algo_df.iterrows():
        score = sum(1 for kw in keywords if kw.lower() in row["Title"].lower())
        if score > best_score:
            best_score, best_row = score, row
    if best_row is not None and best_score > 0:
        return (
            f"**Description:** {best_row['Description']}\n\n"
            f"**Steps:** {' | '.join(best_row['Steps'].splitlines())}\n\n"
            f"**Code Example:**\n```\n{best_row['Code Example']}\n```\n\n"
            f"**Time Complexity (Worst):** {best_row['Time Complexity (Worst)']}\n\n"
            f"**Space Complexity:** {best_row['Space Complexity']}"
        )
    return "No matching algorithm details found."

def retrieve_image_data(topic):
    folder = "images"
    if not os.path.exists(folder): return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if topic.lower() in f.lower()])

def retrieve_video_data(topic):
    keywords = extract_keywords(topic)
    best_row, best_score = None, 0
    for _, row in video_df.iterrows():
        score = sum(1 for kw in keywords if kw.lower() in row["Topic"].lower())
        if score > best_score:
            best_score, best_row = score, row
    return best_row["VideoLink"] if best_row and best_score > 0 else None

def update_dynamic_box(topic):
    if content_mode == "text":
        st.session_state.dynamic_box.markdown(retrieve_text_data(topic))
    elif content_mode == "images":
        imgs = retrieve_image_data(topic)
        if imgs:
            st.session_state.dynamic_box.image(imgs, width=300)
        else:
            st.session_state.dynamic_box.markdown("No images found for this topic.")
    elif content_mode == "videos":
        link = retrieve_video_data(topic)
        if link:
            st.session_state.dynamic_box.video(link)
        else:
            st.session_state.dynamic_box.markdown("No videos found for this topic.")
    else:
        st.session_state.dynamic_box.markdown("Invalid content mode selected.")

def generate_response(prompt):
    full_prompt = f"You are a helpful assistant for algorithms.\nUser: {prompt}\nAssistant:"
    response_text = hf_client.text_generation(
        full_prompt,
        model=MODEL_ID,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    ).strip()
    return response_text

# ====== Streamlit Layout ======
st.set_page_config(page_title="Engagement + Assistant", layout="wide")
col1, col2 = st.columns([1, 1])

# ==== Engagement Detection (Placeholder) ====
with col1:
    st.header("Real-time Engagement Detection")
    st.write("Engagement stream will be here...")

# ==== Dynamic Box + LLM Assistant ====
with col2:
    st.header("Dynamic Content & Assistant")

    if "dynamic_box" not in st.session_state:
        st.session_state.dynamic_box = st.empty()

    topic = st.text_input("Enter topic for dynamic content", value="Radix Sort")
    update_dynamic_box(topic)

    st.markdown("---")

    st.subheader("LLM Chat Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['role']}**: {msg['content']}")

    user_msg = st.text_input("Ask your question...", key="chat_input")
    if user_msg:
        st.session_state.chat_history.append({"role": "User", "content": user_msg})
        with st.spinner("Generating response..."):
            response = generate_response(user_msg)
        st.session_state.chat_history.append({"role": "Assistant", "content": response})
        st.rerun()
