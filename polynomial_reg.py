import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def optimise_price(df, item_id, store_id, target_sales, target_sales_date, current_date, year):
    file_name = f"percentage_changes_decr_price/{year-1}_percentage_changes.csv"

    df = pd.read_csv(file_name)

    base_price =  base_price(df, item_id, store_id)
    base_demand = base_demand(df, item_id, store_id)
    

    

def base_price(df, item_id, store):
    dept_id = item_id[:-4]
    if len(df[(df['item_id'] == item_id) & (df['store_id'] == store)]) > 0:
        temp_df = df[(df['item_id'] == item_id) & (df['store_id'] == store)]
        max_price =  max(temp_df['sell_price'])
    elif len(df[df['item_id'] == item_id]) > 0:
        temp_df = df[df['item_id'] == item_id]
        max_price = max(temp_df["sell-price"])
    elif len(df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]) > 0:
        temp_df = df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]
        max_price = max(temp_df["sell-price"])
    else:
        temp_df = df[df['dept_id'] == dept_id]
        max_price = max(temp_df["sell-price"])
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


def identify_level(df, item_id, store, sales, year, target_sales_date, price_change):

    if not df[df['item_id'] == item_id].empty:
        if len(df[(df['item_id'] == item_id) & (df['store_id'] == store)]) < 20:
            if len(df[df['item_id'] == item_id]) < 20:
                dept_id = item_id[:-4]
                if len(df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]) < 20:
                    department_level_model(item_id, store, sales, year-1, target_sales_date, df, price_change)
                else:
                    department_store_level_model(item_id, store, sales, year-1, target_sales_date, df, price_change)
            else:
                item_level_model(item_id, store, sales, year-1, target_sales_date, df, price_change)
        else:
            item_store_level_model(item_id, store, sales, year-1, target_sales_date, df, price_change)
    else:
        department_store_level_model(item_id, store, sales, year-1, target_sales_date, df, price_change)


def item_level_model(item_id, store, sales, year, target_sales_date, df, price_change):
    df = df[df['item_id'] == item_id]
    df['price_change'] = df['price_change'].abs()
    fit_polynomial_model(df, price_change)

def item_store_level_model(item_id, store, sales, year, target_sales_date, df, price_change):
    df = df[(df['item_id'] == item_id) & (df['store_id'] == store)]
    df['price_change'] = df['price_change'].abs()
    fit_polynomial_model(df, price_change)

def department_level_model(item_id, store, sales, year, target_sales_date, df, price_change):
    dept_id = item_id[:-4]
    df = df[df['dept_id'] == dept_id]
    df['price_change'] = df['price_change'].abs()
    fit_polynomial_model(df, price_change)

def department_store_level_model(item_id, store, sales, year, target_sales_date, df, price_change):
    dept_id = item_id[:-4]
    df = df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]
    df['price_change'] = df['price_change'].abs()
    fit_polynomial_model(df, price_change)


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
    X_new = np.arange(0,101).reshape(-1, 1)  # Example new data
    # Predict sales change

    X_new_poly = poly_features.transform(X_new)
    # Transform new data using polynomial features

    y_pred = model.predict(X_new_poly)

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
    return model.predict(price_change)


identify_level('HOBBIES_1_002', 'CA_1', 500000, 2014, '12/08/2014')