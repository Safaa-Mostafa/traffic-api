from flask import Flask, request, jsonify
import numpy as np 
from keras.models import load_model
from keras.preprocessing import image
import os
from tensorflow import keras


app = Flask(__name__)
model = load_model('pothole_classifire.h5')
labels = ['Accident image', 'normal', 'potholes']
def model_predict(img_path, model):
    img = keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255  
    # preds = model.predict(img)
    predictions = model.predict(img)[0]

    # convert the NumPy array to a Python list
    predictions_list = predictions.tolist()
    print(predictions_list)
    
    # format predictions
    results = []
    for i, prediction in enumerate(predictions_list):
        result = {
            'label': labels[i],
            'probability': float(prediction)
        }
        results.append(result)

    # return predictions as JSON
    return results

@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['file']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    preds = model_predict(image_path, model)
    result = {"prediction":preds}
    return jsonify(result)
 

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
