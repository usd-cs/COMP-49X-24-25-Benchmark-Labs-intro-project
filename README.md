# COMP-49X-24-25-Benchmark-Labs-intro-project

Intro Project For COMP 491

Team: Benchmark Labs

Description: This project focuses on the use of machine learning to classify written characters.
The application allows the user to input an image to be classified by the model with atleast 95% accuracy.

Setup Instructions:
1. Download python version 3.9.20
2. Run pip install -r requirements.txt

Use Instructions: 

In backend directory run app.py to start website, then go to http://127.0.0.1:5000/home

To train on alternate datasets, use the config_helper directory.
Run main.py to create a subset of the MNIST dataset. Then use config_trainer.py to train on that dataset.
Next, you can import that file (final_model.h5) into the backend directory by dragging and dropping and this new model will be used instead.
