# Linear Regression
from sklearn.linear_model import LinearRegression

# Random Forest
from sklearn.ensemble import RandomForestRegressor

# XGBoost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_linear_regression(X_train, y_train):
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def train_random_forest(X_train, y_train):
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    return model

def train_xgboost(X_train, y_train):
    
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_name = model.__class__.__name__
    
    print(f"Model: {model_name}")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    print("-"*30)
    
    return model_name, mse, mae, r2