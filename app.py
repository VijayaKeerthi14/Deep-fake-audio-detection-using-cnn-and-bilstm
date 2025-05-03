import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from werkzeug.utils import secure_filename

# Load the trained model
model_path = "C:/Users/dasar/Downloads/deepfaek_audio_detection.h5"
model = load_model(model_path)

# Flask app setup
app = Flask(__name__)

# Define Upload Folder inside static/
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Convert non-WAV files to WAV
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# Extract MFCC features
def audio_to_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc, axis=1)  # Compute mean across time-axis
    return mfcc.reshape(1, 40, 1, 1)

# Generate MFCC and Mel Spectrogram Images
def save_mfcc_mel_images(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    
    # MFCC Plot
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    mfcc_path = os.path.join(app.config['UPLOAD_FOLDER'], "mfcc.png")
    plt.savefig(mfcc_path)
    plt.close()
    
    # Mel Spectrogram Plot
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title('Mel Spectrogram')
    mel_path = os.path.join(app.config['UPLOAD_FOLDER'], "mel_spectrogram.png")
    plt.savefig(mel_path)
    plt.close()
    
    # Return relative paths for Flask to serve
    return "uploads/mfcc.png", "uploads/mel_spectrogram.png"

# Predict function
def predict_audio(file_path):
    if not file_path.lower().endswith(".wav"):
        file_path = convert_to_wav(file_path)
    
    mfcc = audio_to_mfcc(file_path)
    prediction = model.predict(mfcc)
    confidence = float(prediction[0][0])  # Convert to float for better JSON handling
    result = "Fake Audio" if confidence > 0.85 else "Real Audio"
    
    mfcc_img, mel_img = save_mfcc_mel_images(file_path)
    return result, confidence, mfcc_img, mel_img

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result, confidence, mfcc_img, mel_img = predict_audio(file_path)
            return render_template('result.html', result=result, confidence=confidence, mfcc_img=mfcc_img, mel_img=mel_img)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
