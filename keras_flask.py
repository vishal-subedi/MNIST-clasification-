# Import flask which will handle our communication with the frontend
# Also import few other libraries
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request
from scipy.misc import imread, imresize, imsave
import numpy as np
import re
import keras.models
import tensorflow as tf
from keras.models import model_from_json
import sys
import base64
import os

# Path to our saved model
sys.path.append(os.path.abspath("./model"))

from load import *
# Initialize flask app
app = Flask(__name__)

def init(): 
	json_file = open('C:/Users/visha/flask_deploy/model/model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load weights into new model
	loaded_model.load_weights('C:/Users/visha/flask_deploy/model/model.h5')
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model,graph

#Initialize some global variables
global model, graph
model, graph = init()

def convertImage(imgData1):
  imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
  with open('output.png', 'wb') as output:
    output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
  return render_template("index.html")
  
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
  # Predict method is called when we push the 'Predict' button 
  # on the webpage. We will feed the user drawn image to the model
  # perform inference, and return the classification
  imgData = request.get_data()
  convertImage(imgData)
  # read the image into memory
  x = imread('output.png', mode = 'L')
  # make it the right size
  x = imresize(x, (28, 28))/255
  #You can save the image (optional) to view later
  imsave('final_image.jpg', x)
  x = x.reshape(1, 28, 28, 1)  
  
  with graph.as_default():
    out = model.predict(x)
    response = np.argmax(out, axis = 1)
    return str(response[0])
	
if __name__ == "__main__":
	# run the app locally on the given port
	app.run(host = '0.0.0.0', port = 5000)