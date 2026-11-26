import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="IMDB Sentiment Analyzer (DistilBERT)",
    page_icon="🎬",
    layout="centered"
)

MAX_LEN = 250
MODEL_DIR = "bert_sentiment_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_transformer_artifacts():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    word_count = len(text.split())
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoded)
        prob = torch.sigmoid(outputs.logits.squeeze(-1)).item()

    label = "Positive" if prob >= 0.5 else "Negative"
    return label, prob, word_count

st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.caption("Fine-Tuned DistilBERT Transformer (PyTorch + Hugging Face)")

if not os.path.exists(MODEL_DIR):
    st.error(f"'{MODEL_DIR}' folder nahi mila. Pehle `train.py` run karein.")
    st.stop()

tokenizer, model = load_transformer_artifacts()

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

col1, col2, col3 = st.columns(3)
if col1.button("Positive Sample"):
    st.session_state.review_text = "The movie was not bad at all, I really enjoyed it."
if col2.button("Negative Sample"):
    st.session_state.review_text = "Not worth watching, terrible script and weak direction."
if col3.button("Clear"):
    st.session_state.review_text = ""

review_input = st.text_area(
    "Apna movie review yahan likhein:",
    value=st.session_state.review_text,
    height=130
)

if st.button("Analyze Sentiment", type="primary"):
    if not review_input.strip():
        st.warning("Kripya pehle koi review likhein.")
    else:
        label, prob, word_count = predict_sentiment(review_input, tokenizer, model)
        st.divider()
        st.caption(f"Word Count: {word_count}")
        if label == "Positive":
            st.success("**Prediction: Positive 😊**")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Positive Confidence Score: `{prob:.4f}`")
        else:
            st.error("**Prediction: Negative 😞**")
            st.progress(min(max(1.0 - prob, 0.0), 1.0))
            st.caption(f"Positive Score: `{prob:.4f}` | Negative Confidence: `{1.0 - prob:.4f}`")

if os.path.exists("training_metrics.png"):
    with st.expander("📊 Training Metrics Graph Dekhein"):
        st.image("training_metrics.png", use_container_width=True)