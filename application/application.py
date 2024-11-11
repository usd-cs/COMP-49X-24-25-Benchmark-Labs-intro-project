from numpy import argmax
from keras.utils import load_img  
from keras.utils import img_to_array
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

class ImproperFileSelection(Exception):
    pass

class Application:
    def __init__(self):
        # load model
        self.model = load_model('final_model.h5')
        self.img = self.load_image('sample_image.png')

        self.window = tk.Tk()
        self.window.title("MNIST Character Predictor")
        self.frame = tk.Frame(self.window)
        self.frame.pack(padx=10, pady=10)

        self.upload_button = tk.Button(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(self.frame)
        self.image_label.pack()

        self.prediction_label = tk.Label(self.frame, text="Predicted Digit: ")
        self.prediction_label.pack(pady=10)

        tk.mainloop()

    def upload_image(self):
        image = tk.filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;")])
        img = self.load_image(image)
        start = time.time()
        # self.predict_char(img)
        predict_value = self.model.predict(img)
        end = time.time()
        self.digit = argmax(predict_value)
        self.time_length = end - start
        self.output_prediction()
        # if file is selected
        if image != None:
            pass
        else:
            raise ImproperFileSelection

    # load and prepare the image
    def load_image(self,filename):
        # load the image
        img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 1 channel
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img

    def predict_char(self,img):
        # predict the char
        predict_value = self.model.predict(img)
        self.digit = argmax(predict_value)

    def output_prediction(self):
        self.prediction_label.config(text=f"Predicted Digit: {self.digit}\n Took {self.time_length * 1000} milliseconds to classify")
        print(self.digit)

Application()