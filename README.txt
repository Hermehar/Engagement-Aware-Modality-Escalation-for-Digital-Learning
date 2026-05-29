# Engagement-Aware-Modality-Escalation-for-Digital-Learning

A multi-modal digital learning assistant that combines real-time engagement detection with intelligent content retrieval and dialogue. This project demonstrates how webcam-based student engagement sensing can be paired with an LLM-powered learning companion to dynamically support learners of sorting algorithms.

## 🚀 Project Overview

This repository implements an engagement-aware digital learning system that:
- Detects student engagement in real time using webcam video and a YOLO-based model pipeline.
- Provides a Streamlit chatbot interface powered by Google Gemini (`gemini-2.0-flash`).
- Uses retrieval-augmented generation (RAG) for contextual answers from educational content.
- Supports text, image, and video learning modalities for algorithm education.

## ✨ Key Features

- **Real-time Engagement Monitoring**
  - Person detection with `YOLOv8`
  - Custom engagement classification via `best.pt`
  - Rolling engagement score visualization on the live feed

- **Contextual Learning Assistant**
  - Uses `SentenceTransformer` embeddings and FAISS for fast semantic retrieval
  - Retrieves relevant content chunks from PDF course material
  - Answers user questions with an LLM-guided assistant

- **Multi-modal Educational Support**
  - Built-in algorithm information dataset from `algorithm_info.csv`
  - Linked video resources via `video_links.csv`
  - Algorithm visuals in `sorting_images/`

- **Preprocessing Pipeline**
  - PDF text extraction and chunking with `preprocess.py`
  - Generates `chunks.pkl`, `embeddings.npy`, and `faiss.index`

## 📁 Repository Structure

- `app.py` — Main Streamlit application combining live engagement detection and chat assistant
- `Final_Code.py` — Enhanced Streamlit interface with algorithm selection, content retrieval, and camera management
- `preprocess.py` — Preprocessing script for PDF text extraction, embedding generation, and FAISS indexing
- `algorithm_info.csv` — Algorithm metadata for curated learning content
- `video_links.csv` — Linked educational video resources
- `sorting_images/` — Visualization assets for sorting algorithms
- `chunks.pkl` — Serialized content chunks for retrieval
- `embeddings.npy` — Saved sentence embeddings
- `faiss.index` — FAISS vector store index
- `best.pt` — Custom engagement classifier model
- `yolov8n.pt`, `yolov5s.pt` — YOLO detection models

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd HCAI_Project
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required Python packages:
   ```bash
   pip install streamlit opencv-python av streamlit-webrtc ultralytics sentence-transformers faiss-cpu google-generativeai python-dotenv pymupdf pandas pillow
   ```

4. Create a `.env` file in the project root and set your Google API key:
   ```bash
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

## ▶️ Usage

### 1. Preprocess the Educational Content

This step extracts text from the source PDF, creates text chunks, generates embeddings, and builds the FAISS index.

```bash
python preprocess.py
```

### 2. Start the Streamlit Application

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

### 3. Interact with the System

- Allow webcam access for engagement detection
- Ask questions in the chat panel about algorithm topics
- Monitor live engagement scores and modality escalation behavior

## 🧠 Requirements

- Python 3.9+
- Webcam access for real-time engagement tracking
- `sorting_algorithms.pdf` present in the repository for preprocessing
- Google Generative AI API key in `.env`

## 💡 Notes

- The system is designed as a proof-of-concept for engagement-aware digital learning.
- `app.py` demonstrates the combined webcam + chatbot experience.
- `Final_Code.py` offers a more feature-rich interface for algorithm-specific learning.
- Ensure the model files `best.pt` and `yolov8n.pt` are available in the repository root.

## � Study Design & Evaluation

This project also reflects a controlled user study comparing two instructional conditions:

- **Adaptive condition**: a dynamic interface escalates instructional modality from text to image to video when disengagement is detected.
- **Static condition**: a conventional text-only interface delivers identical educational content without modality changes.

### Learning task

- Participants studied seven sorting algorithms: Bubble, Selection, Insertion, Merge, Quick, Heap, and Radix Sort.
- Each module began with text explanations and steps, with images and videos introduced in the adaptive condition only.

### Engagement inference

- The system captures visual behavioral cues through a webcam and estimates learner engagement in real time.
- A lightweight YOLOv8n model was fine-tuned on student engagement data to trigger modality escalation.
- The detection pipeline is used to adapt instruction rather than to evaluate perception accuracy.

### Participants and measures

- 20 university students participated in the study (10 dynamic, 10 static).
- Measures included NASA-TLX cognitive workload dimensions, user experience questionnaire ratings, and qualitative feedback.

### Key findings

- Adaptive modality escalation reduced mental demand, effort, and frustration compared to static text-only delivery.
- Participants reported higher engagement, satisfaction, and concentration with the adaptive interface.
- Adaptive modality changes were generally perceived as helpful rather than distracting.
- Reported limitations focused on responsiveness and smoother disengagement detection.

##�📌 Recommended Enhancements

- Add more algorithm topics and video resources
- Improve engagement prediction robustness with additional training data
- Expand modality escalation logic for adaptive interventions
- Add user authentication and session logging for classroom evaluation

##📄 License

This repository is provided for research and demonstration purposes. Add a license file if you want to publish this on GitHub.
