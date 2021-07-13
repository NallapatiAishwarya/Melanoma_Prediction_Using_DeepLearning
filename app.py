from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image as pil_image
import cv2


# Keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


Model= load_model(r'.\models\model.h5')     
Model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
classes = ['Melanoma','Not Melanoma']



def model_predict(img_path,fname, Model):

    img = pil_image.open(img_path)
    #cv2.imwrite(fname, img)
    #window_name='image'
    #cv2.imshow(window_name,img)
    img = img.resize((128, 128))
    img=np.asarray(img).reshape(-1,128,128,3)
    img=img.astype('float32')
    img /=255.0
    result = Model.predict_classes(img)
    #print(type(result))
    print(result[0])
    #max_prob = max(result[0])
    #max_prob=1.0
    #class_ind = list(result[0]).index(max_prob)
    #class_name = classes[class_ind]
    return classes[int(result[0])]

@app.route('/', methods=['GET','POST'])
def about():
    # Main page
    return render_template('about.html')


@app.route('/about.html', methods=['GET','POST'])
def about1():
        return render_template('about.html')

@app.route('/index.html', methods=['GET','POST'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        #basepath = r'C:\Users\DELL\OneDrive\Documents\melanoma dataset'
        file_path = os.path.join(
            './uploads/',secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path ,secure_filename(f.filename), Model)

        # Process your result for human
        

        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        #pr = lesion_classes_dict[pred_class[0]]
        #result =str(pr)         
        return str(preds)
    return None


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
