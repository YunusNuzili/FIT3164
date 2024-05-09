import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
#test

def identify_level(item_id, store, sales, year, target_sales_date):
    file_name = f"percentage_changes_decr_price/{year-1}_percentage_changes.csv"

    df = pd.read_csv(file_name)

    if not df[df['item_id'] == item_id].empty:
        if len(df[(df['item_id'] == item_id) & (df['store_id'] == store)]) < 200:
            if len(df[df['item_id'] == item_id]) < 200:
                dept_id = item_id[:-4]
                if len(df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]) < 200:
                    department_level_model(item_id, store, sales, year-1, target_sales_date, df)
                else:
                    department_store_level_model(item_id, store, sales, year-1, target_sales_date, df)
            else:
                item_level_model(item_id, store, sales, year-1, target_sales_date, df)
        else:
            item_store_level_model(item_id, store, sales, year-1, target_sales_date, df)
    else:
        department_store_level_model(item_id, store, sales, year-1, target_sales_date, df)


def item_level_model(item_id, store, sales, year, target_sales_date, df):
    df_filtered = df[df['item_id'] == item_id]
    df_filtered['price_change'] = df_filtered['price_change'].abs()
    fit_polynomial_model(df_filtered)

def item_store_level_model(item_id, store, sales, year, target_sales_date, df):
    df_filtered = df[(df['item_id'] == item_id) & (df['store_id'] == store)]
    df_filtered['price_change'] = df_filtered['price_change'].abs()
    fit_polynomial_model(df_filtered)

def department_level_model(item_id, store, sales, year, target_sales_date, df):
    dept_id = item_id[:-4]
    df_filtered = df[df['dept_id'] == dept_id]
    df_filtered['price_change'] = df_filtered['price_change'].abs()
    fit_polynomial_model(df_filtered)

def department_store_level_model(item_id, store, sales, year, target_sales_date, df):
    dept_id = item_id[:-4]
    df_filtered = df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]
    df_filtered['price_change'] = df_filtered['price_change'].abs()
    fit_polynomial_model(df_filtered)


def fit_polynomial_model(df):

    df = df.dropna()

    # Extract features (price_change) and target variable (sales_change)
    X = df[['price_change']].values
    X = np.clip(X, -100, 100)
    y = df['sales_change'].values
    y = np.clip(y, -100, 2000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define polynomial degree
    degree = 3  # You can change this to any degree you want

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
    #X_test_poly = poly_features.fit_transform(X_test.reshape(-1, 1))

    # Create polynomial regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train_poly, y_train)

    # Assuming X_new contains new data points for price_change

    # Predict sales change

    X_new_poly = poly_features.transform(X_test)
    # Transform new data using polynomial features

    y_pred = model.predict(X_new_poly)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Square Error (RMSE):", rmse)

    # Plot actual vs predicted sales change
    plt.scatter(X_train, y_train, color='blue', label='Actual Sales Change')
    plt.plot(X_test, y_pred, color='red', label='Predicted Sales Change')
    plt.xlabel('Price Change')
    plt.ylabel('Sales Change')
    dep = df['dept_id'].unique()
    plt.title(f'Price Elasticity for {dep}')
    plt.legend()
    plt.grid(True)

    plt.show()


identify_level('HOUSEHOLD_1_007', 'CA_2', 500000, 2014, '12/08/2014')