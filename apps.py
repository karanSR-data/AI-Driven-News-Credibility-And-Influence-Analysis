import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "ai_news_models"

# Load model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict_news(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
        )
    
    inputs.pop("token_type_ids", None)


    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][label].item()

    return label, confidence


def analyze_news(text):
    label, confidence = predict_news(text)

    credibility = "FAKE ❌" if label == 1 else "REAL ✅"

    if confidence > 0.85:
        influence = "HIGH 🔥"
    elif confidence > 0.60:
        influence = "MEDIUM ⚠️"
    else:
        influence = "LOW ℹ️"

    return credibility, influence, round(confidence * 100, 2)


# Streamlit UI
st.set_page_config(page_title="AI News Analyzer", layout="centered")

st.title("📰 AI News Credibility Analyzer")

user_input = st.text_area("Enter news text", height=150)

if st.button("Analyze"):
    if user_input.strip():
        credibility, influence, confidence = analyze_news(user_input)

        st.subheader("Result")
        st.write(f"Credibility: {credibility}")
        st.write(f"Influence: {influence}")
        st.write(f"Confidence: {confidence}%")
    else:
        st.warning("Please enter some text.")