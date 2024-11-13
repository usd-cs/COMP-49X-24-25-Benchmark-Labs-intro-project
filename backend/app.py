from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from flask_cors import CORS
import time
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Launch Flask app
app = Flask(__name__, template_folder='../frontend')
CORS(app)  # Enable CORS for all routes

# Load the model at startup
model = load_model('./final_model.h5')

# Function to process image for model prediction
def process_image(image_data):
    
    # convert uploaded image to bytes
    img_bytes = BytesIO(image_data)
    
	# load the image
    img = load_img(img_bytes, color_mode='grayscale', target_size=(28, 28))
    
	# convert to array
    img_array = img_to_array(img)
    
	# reshape into a single sample with 1 channel
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # prepare pixel data
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0
    return img_array

# Send home page HTML
@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

# Route to predict digit from image
@app.route('/predict', methods=['POST'])
def predict():
    try:
                
        # Ensure image is sent in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Read in image data
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Process the image for model
        processed_image = process_image(image_data)
        
        # Make prediction from processed image
        prediction = model.predict(processed_image)
        
        # Get prediction results and confidence
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit])
                        
        # Return results
        return jsonify({
            'digit': int(predicted_digit),
            'confidence': confidence
        })
        
    # If error, return 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)