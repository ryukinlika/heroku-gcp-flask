# from operator import contains, methodcaller
import os
from app import app, bootstrap
from predict_classes import mix_classes, local_classes
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
# import tensorflow as tf
import cv2
import numpy as np
from google.cloud import aiplatform
from google.oauth2 import service_account
# import keras
# import pandas as pd
# import urllib.request
# import joblib
# from keras_cv_attention_models import coatnet
# import json
# from json import JSONEncoder


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PROJECT_NAME = "nema-online-classifier"
PROJECT_LOCATION = "asia-northeast1"
PROJECT_ENDPOINT_ID = "7422108107767021568"
PROJECT_API_ENDPOINT = "asia-northeast1-aiplatform.googleapis.com"

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload_form():
	return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()

@app.route('/inference', methods=['POST'])
def inference(): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    filename = request.form.get('file_name')
    if filename == '':
        flash('File name empty')
        return redirect("/")

    print(filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    modelType = request.form.get('model')

    # prepare image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (224, 224))
    resized_img = (resized_img - 127.5) / 127.5  # value to -1:1 for efficientnet
    resized_img = np.expand_dims(resized_img, axis=0)
    resized_img = np.expand_dims(resized_img, axis=-1)
    # print("============Values=============")
    # print("min values / max values")  
    # print(np.amin(resized_img))
    # print(np.amax(resized_img))
    # print("shapes")
    # print(resized_img.shape)

    data = resized_img.tolist()

    prediction = endpoint_predict_sample(PROJECT_NAME, PROJECT_LOCATION, PROJECT_ENDPOINT_ID, 
        instances = data
    )

    predict_label = local_classes[np.argmax(prediction.predictions[0])]
    
    return render_template('inference.html', filename=filename, prediction=predict_label, model=modelType)


def endpoint_predict_sample(
    project: str, location: str, endpoint: str, instances: list,
):
    creds = service_account.Credentials.from_service_account_file("json/nema-online-classifier-e5d23692a03d.json")
    aiplatform.init(project=project, location=location, credentials=creds)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction

