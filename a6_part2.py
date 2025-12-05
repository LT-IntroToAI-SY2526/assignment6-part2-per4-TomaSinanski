"""
Assignment 6 Part 2: House Price Prediction (Multivariable Regression)

This assignment predicts house prices using MULTIPLE features.
Complete all the functions below following the in-class car price example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    data = pd.read_csv(filename)
    print("=== House Price Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"\nBasic statistics:")
    print(data.describe())
    print(f"\nColumn names: {list(data.columns)}")
    return data


def visualize_features(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('House Features vs Price', fontsize=16, fontweight='bold')
    axes[0, 0].scatter(data['SquareFeet'], data['Price'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('SquareFeet')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('SquareFeet vs Price')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].scatter(data['Bedrooms'], data['Price'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Bedrooms')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Bedrooms vs Price')
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].scatter(data['Bathrooms'], data['Price'], color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Bathrooms')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Bathrooms vs Price')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].scatter(data['Age'], data['Price'], color='orange', alpha=0.6)
    axes[1, 1].set_xlabel('Age (years)')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].set_title('Age vs Price')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_plots.png', dpi=300, bbox_inches='tight')
    plt.show()


def prepare_features(data):
    feature_columns = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    X = data[feature_columns]
    y = data['Price']
    print(f"\n== Feature Prep ==")
    print(f"Features (X) shape: {X.shape}")
    print("\n== Feature Prep ==")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Feature columns: {feature_columns}")
    return X, y



def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    print(f"\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set:  {len(X_test)} samples")
    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train, feature_names):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n=== Model Training ===")
    print(f"Intercept (b0): {model.intercept_}")
    print("\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef}")
    print("\nModel Equation:")
    equation = "Price = {:.2f}".format(model.intercept_)
    for name, coef in zip(feature_names, model.coef_):
        equation += f" + ({coef:.2f} * {name})"
    print(equation)
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print("\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2 * 100:.2f}% of the variation in house prices")
    print(f"\nRMSE: ${rmse:,.2f}")
    print(f"  → On average, predictions are off by about ${rmse:,.2f}")
    print("\n=== Feature Importance ===")
    importance = np.abs(model.coef_)
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(feature_importance, start=1):
        print(f"{i}. {name}: {score:.4f}")
    return predictions


def compare_predictions(y_test, predictions, num_examples=5):
    print("\n=== Prediction Examples ===")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)
    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100
        print(f"${actual:>13.2f}   ${predicted:>13.2f}   ${error:>10.2f}   {pct_error:>6.2f}%")

def make_prediction(model, sqft, bedrooms, bathrooms, age):
    house_features = pd.DataFrame(
        [[sqft, bedrooms, bathrooms, age]],
        columns=['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age'])
    predicted_price = model.predict(house_features)[0]
    print("\n=== New House Prediction ===")
    print(f"House specs: {sqft} sqft, {bedrooms} bed, {bathrooms} bath, {age} years old")
    print(f"Predicted price: ${predicted_price:,.2f}")
    return predicted_price



if __name__ == "__main__":
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)
    data = load_and_explore_data("house_prices.csv")
    visualize_features(data)
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, X.columns)
    predictions = evaluate_model(model, X_test, y_test, X.columns)
    compare_predictions(y_test, predictions, num_examples=10)
    make_prediction(model, sqft=1800, bedrooms=3, bathrooms=2, age=12)
    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part2_writeup.md!")
    print("=" * 70)