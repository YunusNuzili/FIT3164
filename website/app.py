from flask import Flask, request, jsonify
from flask_cors import CORS  
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
CORS(app)  



linear_model = LinearRegression()
poly_model = LinearRegression()

def linear_model_predict(sales_quantity, number_of_days):
    X = np.array([[1, 10], [2, 20], [3, 30]])
    y = np.array([100, 200, 300])
    linear_model.fit(X, y)
    return linear_model.predict(np.array([[sales_quantity, number_of_days]]))[0]

def polynomial_model_predict(sales_quantity, number_of_days):
    X = np.array([[1, 10], [2, 20], [3, 30]])
    y = np.array([100, 200, 300])
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model.fit(X_poly, y)
    return poly_model.predict(poly.transform(np.array([[sales_quantity, number_of_days]])))[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  
    sales_quantity = data.get('salesQuantity', 0)
    number_of_days = data.get('numberOfDays', 0)
    model_type = data.get('modelType', 'linear')

    if model_type == 'linear':
        prediction = linear_model_predict(sales_quantity, number_of_days)
    elif model_type == 'polynomial':
        prediction = polynomial_model_predict(sales_quantity, number_of_days)
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)