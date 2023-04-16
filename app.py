from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('emotion_recognition_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the file from the request
    file = request.files['image']

    # Read the image and preprocess it
    img = Image.open(file).convert('L').resize((48, 48))
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape((1, 48, 48, 1))

    # Predict the emotion using the loaded model
    emotion = model.predict(img_arr)[0]
    emotion_label = np.argmax(emotion)

    # Return the emotion as a string
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[emotion_label]

if __name__ == '__main__':
    app.run(debug=True)