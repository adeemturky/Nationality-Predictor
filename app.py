import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/nationality_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Country label mapping (اختياري: أضف أعلام أو تعريب)
country_mapping = {
    "Saudi": "🇸🇦 Saudi",
    "Egyptian": "🇪🇬 Egyptian",
    "Emirati": "🇦🇪 Emirati",
    "Palestinian": "🇵🇸 Palestinian",
    "Jordanian": "🇯🇴 Jordanian",
    "Iraqi": "🇮🇶 Iraqi",
    "Persian": "🇮🇷 Persian",
    "Turkish": "🇹🇷 Turkish",
    "Indian": "🇮🇳 Indian",
    "Pakistani": "🇵🇰 Pakistani",
    "Indonesian": "🇮🇩 Indonesian",
    "Malaysian": "🇲🇾 Malaysian",
    "French": "🇫🇷 French",
    "British": "🇬🇧 British",
    "American": "🇺🇸 American",
    "Chinese": "🇨🇳 Chinese",
    "Japanese": "🇯🇵 Japanese",
    "German": "🇩🇪 German",
    "Italian": "🇮🇹 Italian",
    "Spanish": "🇪🇸 Spanish"
}

# Streamlit UI
st.set_page_config(page_title="Nationality Predictor", page_icon="🌍")
st.title("🌍 Predict Nationality from Name")

# Input field
name_input = st.text_input("Enter your name:")

if name_input:
    name = name_input.strip().lower()
    name_vectorized = vectorizer.transform([name])
    prediction = model.predict(name_vectorized)[0]

    label = country_mapping.get(prediction, "🌐 Unknown")
    st.success(f"Predicted Origin: **{label}**")
