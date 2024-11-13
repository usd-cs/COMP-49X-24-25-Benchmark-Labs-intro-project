import pytest
from app import app, process_image
import io
from PIL import Image
import numpy as np

# Set up testing client
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Test the home route returns 200 response"""
    response = client.get('/home')
    assert response.status_code == 200

def test_predict_no_image(client):
    """Test prediction route without image returns 400 response"""
    response = client.post('/predict')
    assert response.status_code == 400
    assert b'No image file provided' in response.data

def test_process_image():
    """Test image processing function returns correct shape and type"""
    
    # Create a test image
    img = Image.new('L', (28, 28), color=255)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Process the image
    processed = process_image(img_byte_arr)
    
    # Verify output shape and type
    assert processed.shape == (1, 28, 28, 1)
    assert processed.dtype == 'float32'