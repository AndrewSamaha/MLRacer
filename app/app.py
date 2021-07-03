from flask import Flask, render_template, request, send_file, send_from_directory
from flask import jsonify
import json
import mysql.connector as mysql
import numpy as np
import random

mydb = mysql.connect(
    host="127.0.0.1",
    port=49157,
    user="root",
    password="datascience",
    database="racer")


  #host="localhost",
  ###port=49153,
  #user="root",
  #password="datascience",
  #auth_plugin="mysql_native_password")
dbcursor = mydb.cursor()

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def root():
    return app.send_static_file('game.html')

@app.route('/hello/<name>/')
def hello(name):
    """ Displays the page greats who ever comes to visit it.
    """
    return render_template('hello.html', name=name)

@app.route('/putdata/', methods = ['POST', 'GET'])
def data():
    if request.method == 'POST':
        #print(" received POST")
        #print(request.get_json().get('i'))
        time = request.get_json().get('t')
        position = request.get_json().get('p')
        velocity = request.get_json().get('v')
        rotation = request.get_json().get('r')
        image = request.get_json().get('i')
        insertCmd = "INSERT into images(time, position, velocity, rotation, image) VALUES ('"+str(time)+"', '"+str(position)+"','"+str(velocity)+"','"+str(rotation)+"','"+image+"');"
        
        try:
            dbcursor.execute(insertCmd)
            mydb.commit()
        except Exception as e:
            print(f"Exception: {e}")
            return f"Exception: {e}"
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
    app.debug=True
    app.run()