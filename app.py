import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("voice_sentiment_cnn_model.h5")

# Load LabelEncoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_sentiment(audio_file):
    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    emotion = le.classes_[predicted_class]
    return f"Predicted Emotion: {emotion}"

interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="🎤 Voice Sentiment Analysis",
    description="Upload a .wav audio file to predict emotional sentiment."
)

interface.launch()