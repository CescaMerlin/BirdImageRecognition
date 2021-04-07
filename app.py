import tensorflow as tf
import numpy as np
import requests
import imageio
from PIL import Image
from tensorflow import keras
from flask import Flask,request
from keras.models import load_model

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
"""

model = load_model('owl_recognition_model.h5')

# create the flask object
app = Flask(__name__)

@app.route('/')
def index():
    return "Index Page"

@app.route('/predict',methods=['GET','POST'])
def makePrediction():
    imgpath = request.form.get('data')
    class_names = ['Asio otus', 'Bubo blakistoni', 'Ninox japonica', 'Otus elegans', 'Strix uralensis']
    if imgpath == None:
        return 'Got None'
    else:
        im = imageio.imread(imgpath)
        im = Image.fromarray(im).resize((160,160))
        img_array = keras.preprocessing.image.img_to_array(im)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

        return result
    

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)