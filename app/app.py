from flask import Flask, render_template, request, send_file, send_from_directory
from flask import jsonify
import json
import mysql.connector as mysql
import numpy as np
import random
from src.db import *
from waitress import serve
from src.model import ModelWrapper


# Setup 
mydb = mysql.connect(
    host="127.0.0.1",
    port=49153,
    user="root",
    password="datascience",
    database="racer")

dbcursor = mydb.cursor()

model = ModelWrapper("../models/alpha.5layer.3-20pctdrop.rmsprop.mse.unnormalized.40krows.100epochs.model")

app=Flask(__name__)

@app.route('/')
def index():
    batchList = getBatches(mydb)
    return render_template('index.html', batchList=batchList)

@app.route('/train')
def root():
    return app.send_static_file('game.html')

@app.route('/hello/<name>/')
def hello(name):
    """ Displays the page greats who ever comes to visit it.
    """
    return render_template('hello.html', name=name)

@app.route('/predict/', methods=["POST"])
def predict():
    if request.method == 'POST':
        time = request.get_json().get('t')
        position = request.get_json().get('p')
        velocity = request.get_json().get('v')
        rotation = request.get_json().get('r')
        batchid  = request.get_json().get('b')
        image = request.get_json().get('i')
        imagearray = uriToNP(image)
        prediction = model.drive(imagearray, time)
        #saveImage(mydb, batchid, 0, time, position, velocity, rotation, image)
        return jsonify( 
            position=churnAnswer,
            time=time
        )

@app.route('/putdata/', methods=['POST'])
def data():
    if request.method == 'POST':
        #print(" received POST")
        #print(request.get_json().get('i'))
        time = request.get_json().get('t')
        position = request.get_json().get('p')
        velocity = request.get_json().get('v')
        rotation = request.get_json().get('r')
        batchid  = request.get_json().get('b')
        image = request.get_json().get('i')
        saveImage(mydb, batchid, 0, time, position, velocity, rotation, image)
        return f"200 {position} + {velocity} + {image}"

@app.route('/model/')
def model():
    inputs = request.args
    featurevalues = []
    
    for key,val in enumerate(inputs):
        print(f"shape of inputs: {key} {val}")
        featurevalues.append(float(inputs[val]))
        #inputs['input_1']
    
    featurevalues = np.array(featurevalues)
    featurevalues = np.reshape(featurevalues,(1,-1))
    #featurevalues = featurevalues.astype(np.float64)

    print("----------------------------------------")
    print("----------------------------------------")
    print("----------------------------------------")
    print(f"featurevalues.shape={featurevalues.shape}")
    print(featurevalues)
    nn = load_model("static/model.joblib")
    predictions = nn.predict(featurevalues)
    print(predictions.shape)
    #predictions = [ random.random() for _ in range(0,17) ]
    churnAnswer = ""
    if predictions[0] > .5:
        churnAnswer = "Churn"
    else:
        churnAnswer = "No Churn"
    return jsonify( 
        input_1_result=churnAnswer,
        input_2_result=str(predictions[0])
        )


if __name__ == '__main__':
    #app.debug=False
    #app.run()
    serve(app, host='0.0.0.0', port=5000)