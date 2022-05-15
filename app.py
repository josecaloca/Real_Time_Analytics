import numpy as np
from joblib import load
from flask import (
        Flask, 
        request, 
        jsonify
)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_prediction():
    model = load('model.joblib')
    
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))

    features = [[sepal_length, petal_length]]
    
    predicted_class = model.predict(features)
    
    predicted_class = float(predicted_class)
    
    return jsonify(features=str(features), predicted_class= str(predicted_class))


if __name__ == '__main__':
    app.run(debug=True)
    
