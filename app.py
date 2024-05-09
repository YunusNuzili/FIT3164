from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        item_id = data['itemId']
        store_id = data['storeId']
        sales = data.get('sales', 0)  # 默认销售量为0，如果没有提供
        target_sales_date = data['targetSalesDate']
        current_date = data.get('currentDate', '')
        year = int(data['year'])

        # 调用 optimise_price 函数进行价格优化计算
        prediction = optimise_price(item_id, store_id, sales, target_sales_date, current_date, year)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



#polynomial_model(item_id, store_id, year, target_sales_date, current_date)
   
   
def optimise_price(item_id, store_id, target_sales, target_sales_date, current_date, year):
    file_name = f"percentage_changes_decr_price/{year-1}_percentage_changes.csv"

    df = pd.read_csv(file_name)

    base_price1 =  base_price(df, item_id, store_id)
    base_demand1 = base_demand(df, item_id, store_id)

    sales_change = identify_level(df, item_id, store_id, target_sales, year, 30)[0]
    print(sales_change)

    # Initial settings
    max_iterations = 100
    tolerance = 100  # Adjust this based on the acceptable error in target sales
    learning_rate = 0.01  # This should be fine-tuned based on the responsiveness of the model

    current_price = base_price1
    current_demand = base_demand1

    for _ in range(max_iterations):
        # Calculate price change percentage
        price_change_percentage = (current_price - base_price) / base_price * 100
        # Predict the percentage change in sales
        predicted_sales_change_percentage = identify_level(df, item_id, store_id, current_demand, year, price_change_percentage)
        # Calculate predicted sales
        predicted_sales = current_demand * (1 + predicted_sales_change_percentage / 100)

        # Check if the predicted sales are within tolerance of the target sales
        if abs(predicted_sales - target_sales) <= tolerance:
            print(f"Optimal price found: ${current_price} with predicted sales of {predicted_sales}")
            return current_price
        # Update current demand for the next iteration based on predicted sales
        current_demand = predicted_sales
        # Adjust the price based on the error
        error = predicted_sales - target_sales
        current_price -= learning_rate * error  # Adjust the price accordingly

    print("Max iterations reached without finding optimal price.")
    return current_price

        

def base_price(df, item_id, store):
    dept_id = item_id[:-4]
    if len(df[(df['item_id'] == item_id) & (df['store_id'] == store)]) > 0:
        temp_df = df[(df['item_id'] == item_id) & (df['store_id'] == store)]
        max_price =  max(temp_df['sell_price'])
    elif len(df[df['item_id'] == item_id]) > 0:
        temp_df = df[df['item_id'] == item_id]
        max_price = max(temp_df["sell_price"])
    elif len(df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]) > 0:
        temp_df = df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]
        max_price = max(temp_df["sell_price"])
    else:
        temp_df = df[df['dept_id'] == dept_id]
        max_price = max(temp_df["sell_price"])
    return max_price

def base_demand(df, item_id, store):
    dept_id = item_id[:-4]
    if len(df[(df['item_id'] == item_id) & (df['store_id'] == store)]) > 0:
        temp_df = df[(df['item_id'] == item_id) & (df['store_id'] == store)]
        max_sales =  max(temp_df['sales'])
    elif len(df[df['item_id'] == item_id]) > 0:
        temp_df = df[df['item_id'] == item_id]
        max_sales = max(temp_df["sales"])
    elif len(df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]) > 0:
        temp_df = df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]
        max_sales = max(temp_df["sales"])
    else:
        temp_df = df[df['dept_id'] == dept_id]
        max_sales = max(temp_df["sales"])
    return max_sales


def identify_level(df, item_id, store, sales, year, price_change):

    if not df[df['item_id'] == item_id].empty:
        if len(df[(df['item_id'] == item_id) & (df['store_id'] == store)]) < 20:
            if len(df[df['item_id'] == item_id]) < 20:
                dept_id = item_id[:-4]
                if len(df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]) < 20:
                    return department_level_model(item_id, store, sales, year-1, df, price_change)
                else:
                    return department_store_level_model(item_id, store, sales, year-1, df, price_change)
            else:
                return item_level_model(item_id, store, sales, year-1, df, price_change)
        else:
            return item_store_level_model(item_id, store, sales, year-1, df, price_change)
    else:
        return department_store_level_model(item_id, store, sales, year-1, df, price_change)


def item_level_model(item_id, store, sales, year, df, price_change):
    df = df[df['item_id'] == item_id]
    df['price_change'] = df['price_change'].abs()
    return fit_polynomial_model(df, price_change)

def item_store_level_model(item_id, store, sales, year, df, price_change):
    df = df[(df['item_id'] == item_id) & (df['store_id'] == store)]
    df['price_change'] = df['price_change'].abs()
    return fit_polynomial_model(df, price_change)

def department_level_model(item_id, store, sales, year, df, price_change):
    dept_id = item_id[:-4]
    df = df[df['dept_id'] == dept_id]
    df['price_change'] = df['price_change'].abs()
    return fit_polynomial_model(df, price_change)

def department_store_level_model(item_id, store, sales, year, df, price_change):
    dept_id = item_id[:-4]
    df = df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]
    df['price_change'] = df['price_change'].abs()
    return fit_polynomial_model(df, price_change)


def fit_polynomial_model(df, price_change):

    df = df.dropna()

    # Extract features (price_change) and target variable (sales_change)
    X = df[['price_change']].values
    X = np.clip(X, -100, 100)
    y = df['sales_change'].values
    y = np.clip(y, -100, 2000)

    # Define polynomial degree
    degree = 3  # You can change this to any degree you want

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X.reshape(-1, 1))

    # Create polynomial regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_poly, y)

    # Assuming X_new contains new data points for price_change
    X_new = np.arange(101).reshape(-1, 1)  # Example new data
    # Predict sales change

    X_new_poly = poly_features.transform(X_new)
    # Transform new data using polynomial features

    y_pred = model.predict(X_new_poly)
    print(y_pred)

    # Plot actual vs predicted sales change
    plt.scatter(X, y, color='blue', label='Actual Sales Change')
    plt.plot(X_new, y_pred, color='red', label='Predicted Sales Change')
    plt.xlabel('Price Change')
    plt.ylabel('Sales Change')
    dep = df['dept_id'].unique()
    plt.title(f'Price Elasticity for {dep}')
    plt.legend()
    plt.grid(True)

    plt.show()

    price_change_new = np.array([[price_change]])
    price_change_new_poly = poly_features.transform(price_change_new)
    return model.predict(price_change_new_poly)



if __name__ == '__main__':
    app.run(debug=True)
