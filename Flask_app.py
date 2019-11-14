
import base64
import numpy as np
import io
import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from PIL import Image
from flask import Flask
from flask import request
from flask import jsonify

# initializing FLASK

app = Flask(__name__)
def get_model ():
    global  model
    new_model = load_model('run1.h5')
    print('model loaded !!')
    return new_model

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(100,100)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    return image

print('loading model')
get_model()

@app.route('/predict',methods=['POST'])

def predict():
    prediction = request.get_json(force=True)
    encoded = prediction['image']
    decoded =base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image=image)
    prediction = model.predict(processed_image).tolist()
    return jsonify(prediction)

if __name__== '__main__':
    app.run(host='192.168.0.105')


