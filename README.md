# Speech Emotion Recognition

A Streamlit deployment of the Speech Emotion Recognition machine-learning project.

## Model pipeline

Audio → MFCC (40) + Chroma (12) + Mel (128) → 180 features → StandardScaler → SVM → Emotion

The deployment uses the saved trained model and scaler:

- `ser_final_svm_model.pkl`
- `ser_scaler.pkl`

## Deployment

1. Create a public GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `ser_final_svm_model.pkl`
   - `ser_scaler.pkl`
3. Go to Streamlit Community Cloud.
4. Create an app from the GitHub repository.
5. Select `app.py` as the main file.
