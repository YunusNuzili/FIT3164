from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from flask_cors import CORS
import boto3

import json

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        item_id = data['itemId']
        store_id = data['storeId']
        target_sales_date = data['targetSalesDate']
        current_date = data.get('currentDate', '')
        year = int(data['year'])

        # 调用 optimise_price 函数进行价格优化计算
        prediction = optimise_price(item_id, store_id, target_sales_date, current_date, year)
        plot = "/temp_plot.png"

        
        
        

        optimal_price = str(round(prediction[0],2))
        print("Prediction:", optimal_price)
        x_pred = [i for i in range(1, 101)]

        y_pred = prediction[2].tolist()
        print(prediction[2].tolist())
        x_values = prediction[3].tolist()
        y_values = prediction[4].tolist()
        price_change_percentage = prediction[5]


        return jsonify({'x_values': x_values, 'y_values': y_values, 'x_pred': x_pred, 'y_pred': y_pred,'optimized_price': optimal_price, "discount": price_change_percentage})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



#polynomial_model(item_id, store_id, year, target_sales_date, current_date)
   
   
def optimise_price(item_id, store_id, target_sales_date, current_date, year):
    file_name = f"percentage_changes_decr_price/{year-1}_percentage_changes.csv"

    df = pd.read_csv(file_name)


    # Get base price and average weekly demand at that price
    base_price = get_base_price_and_demand(df, item_id, store_id)[0]
    base_weekly_demand = get_base_price_and_demand(df, item_id, store_id)[1]

    if base_weekly_demand < 1:
        base_weekly_demand = 1
    
    # Convert date strings to datetime objects
    date_format = "%d/%m/%Y"
    date1 = datetime.strptime(current_date, date_format)
    date2 = datetime.strptime(target_sales_date, date_format)
    
    # Calculate the difference between the dates
    delta = date2 - date1
    
    # Extract the number of days from the timedelta object
    num_weeks = delta.days//7

    # Initial settings
    max_iterations = 100
    margin = 0.05
    # Cost price is generated with a 5% margin
    cost_price = base_price * (1 - margin)

    current_price = base_price
    current_demand = base_weekly_demand
    best_profit = 0
    best_loss = float("-inf")
    best_price = base_price
    temp = [1, 2, 3, 4]
    price_change_percentage1 = 0

    X_pred = []
    Y_pred = []
    X = []
    Y = []

    for i in range(1, max_iterations):
        price_change_percentage = i
        current_price = base_price - ((price_change_percentage*base_price)/100)
        temp_value = identify_level(df, item_id, store_id, current_demand, year, price_change_percentage)
        predicted_sales_change_percentage = temp_value[0]
        X_pred = temp_value[1]
        Y_pred = temp_value[2]
        X = temp_value[3]
        Y = temp_value[4]
        predicted_sales = num_weeks * (base_weekly_demand * (1 + predicted_sales_change_percentage / 100))
        if i == 81:
            temp[0] = (current_price - cost_price) * predicted_sales
            temp[1] = predicted_sales
            temp[2] = current_price
            temp[3] = predicted_sales_change_percentage

        # Calculate profit and check against the best seen so far
        if predicted_sales < 200:
            profit = (current_price - cost_price) * predicted_sales  # Ensure we do not exceed stock
            if (profit <  0) and (profit > best_loss):
                best_loss = profit
                best_price = current_price
                price_change_percentage1 = price_change_percentage
            elif profit > best_profit:
                best_profit = profit
                best_price = current_price
                price_change_percentage1 = price_change_percentage
    
    print(temp)
    print(best_price, price_change_percentage1, best_profit, best_loss)
    print(base_price, cost_price, base_weekly_demand)
    return [best_price, X_pred, Y_pred, X, Y, price_change_percentage1]

        

def get_base_price_and_demand(df, item_id, store):
    dept_id = item_id[:-4]
    selected_df = None

    # Prioritize data specificity from most specific (item and store match) to least specific (department level)
    if not df[(df['item_id'] == item_id) & (df['store_id'] == store)].empty:
        selected_df = df[(df['item_id'] == item_id) & (df['store_id'] == store)]
    elif not df[df['item_id'] == item_id].empty:
        selected_df = df[df['item_id'] == item_id]
    elif not df[(df['dept_id'] == dept_id) & (df['store_id'] == store)].empty:
        selected_df = df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]
    elif not df[df['dept_id'] == dept_id].empty:
        selected_df = df[df['dept_id'] == dept_id]

    if selected_df is not None:
        # Calculate the base price as the maximum price observed
        base_price = max(selected_df['sell_price'])
        # Filter the DataFrame to rows with this base price
        base_price_df = selected_df[selected_df['sell_price'] == base_price]
        # Calculate the average weekly demand (sales) at this base price
        if not base_price_df.empty:
            average_demand = base_price_df['sales'].mean()
            return base_price, average_demand
        else:
            # In case there's no sales data at the base price, fall back to overall average
            average_demand = selected_df['sales'].mean()
            return base_price, average_demand

    # Fallback if no relevant data found; perhaps raise an error or return a default
    return None, None

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
    #print(y_pred)

    # if price_change == 99:
    #     # Plot actual vs predicted sales change
    #     plt.scatter(X, y, color='blue', label='Actual Sales Change')
    #     plt.plot(X_new, y_pred, color='red', label='Predicted Sales Change')
    #     plt.xlabel('Price Change')
    #     plt.ylabel('Sales Change')
    #     dep = df['dept_id'].unique()
    #     plt.title(f'Price Elasticity for {dep}')
    #     plt.legend()
    #     plt.grid(True)

    #     filename = "website/temp_plot.png"

    #     plt.savefig(filename)

    #     plt.close()

    price_change_new = np.array([[price_change]])
    price_change_new_poly = poly_features.transform(price_change_new)
    return [model.predict(price_change_new_poly), X_new, y_pred, X, y]





if __name__ == '__main__':
    app.run(debug=True)
