import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)

MODEL_PATH = Path("ser_final_svm_model.pkl")
SCALER_PATH = Path("ser_scaler.pkl")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def extract_features(filepath):
    y, sr = librosa.load(filepath, duration=3, offset=0.5)

    mfcc = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
        axis=0
    )
    chroma = np.mean(
        librosa.feature.chroma_stft(y=y, sr=sr).T,
        axis=0
    )
    mel = np.mean(
        librosa.feature.melspectrogram(y=y, sr=sr).T,
        axis=0
    )

    return np.concatenate([mfcc, chroma, mel])

st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Upload a speech audio file and the trained SVM model will "
    "predict the emotion category from acoustic features."
)

st.info("Supported formats: WAV, MP3, OGG, M4A")

try:
    model, scaler = load_model()
except Exception as e:
    st.error("The model files could not be loaded.")
    st.code(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload your speech audio",
    type=["wav", "mp3", "ogg", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🔍 Predict Emotion", use_container_width=True):
        temp_path = None

        try:
            suffix = Path(uploaded_file.name).suffix or ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            features = extract_features(temp_path)

            if features.shape[0] != 180:
                st.error(
                    f"Unexpected feature count: {features.shape[0]}. "
                    "The model expects 180 features."
                )
                st.stop()

            features_scaled = scaler.transform(
                features.reshape(1, -1)
            )

            prediction = model.predict(features_scaled)[0]

            st.success(f"Predicted Emotion: **{str(prediction).upper()}**")

            st.caption(
                "The model predicts an emotion category from speech "
                "acoustic features; it does not determine a person's "
                "actual internal emotional state."
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

st.divider()
st.caption("Speech Emotion Recognition • SVM + MFCC + Chroma + Mel features")
