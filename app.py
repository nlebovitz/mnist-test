from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)
try:
    # Replace with the correct path to your model
    mnist_model = tf.keras.models.load_model('./models')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

@app.route('/')
def index():
    return render_template('index.html')  # HTML for the drawing canvas

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return 'Image data not found', 400

    image_data = data['image']
    # Decode the base64 string
    image_data = base64.b64decode(image_data.split(',')[1])
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_data))
    image.save("processed_image.png")

    # Convert to grayscale, resize and invert colors for MNIST
    image = image.convert('L').resize((28, 28))
    image = np.invert(image)

    # Normalize and reshape for the model
    image = np.array(image) / 255.0
    image = image.reshape((1, 28, 28, 1))

    # Predicting with the model
    predictions = mnist_model.predict(image)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    predicted_digit = np.argmax(predictions[0])
    print(probabilities)  # To see the raw output

    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)