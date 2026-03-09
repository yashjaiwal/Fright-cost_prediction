import joblib
import pandas as pd
import os

# Import preprocessing functions
from data_preprocessing import load_vendore_invoice, prepare_feature, split_data

# Import model functions
from model_eval import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model
)

def main():

    # 1️⃣ Database path
    data_path = (r"C:\Users\User\Desktop\ML project\Data\inventory.db")

    # 2️⃣ Load data
    df = load_vendore_invoice(data_path)

    # 3️⃣ Prepare features
    X, y = prepare_feature(df)

    # 4️⃣ Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5️⃣ Train models
    lr = train_linear_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    # 6️⃣ Evaluate models
    lr_name, lr_mse, lr_mae, lr_r2 = evaluate_model(lr, X_test, y_test)
    rf_name, rf_mse, rf_mae, rf_r2 = evaluate_model(rf, X_test, y_test)
    xgb_name, xgb_mse, xgb_mae, xgb_r2 = evaluate_model(xgb, X_test, y_test)

    # 7️⃣ Store results
    results = [
        {"Model": lr_name, "MSE": lr_mse, "MAE": lr_mae, "R2": lr_r2, "Model_Obj": lr},
        {"Model": rf_name, "MSE": rf_mse, "MAE": rf_mae, "R2": rf_r2, "Model_Obj": rf},
        {"Model": xgb_name, "MSE": xgb_mse, "MAE": xgb_mae, "R2": xgb_r2, "Model_Obj": xgb},
    ]

    results_df = pd.DataFrame(results)

    # 8️⃣ Select best model (lowest MAE)
    best_model_row = results_df.loc[results_df["MAE"].idxmin()]

    print("\n==========================")
    print("BEST MODEL")
    print("==========================")

    print("Model:", best_model_row["Model"])
    print("MAE:", best_model_row["MAE"])
    print("R2 Score:", best_model_row["R2"])

    best_model = best_model_row["Model_Obj"]

    # 9️⃣ Save best model
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/best_freight_model.pkl")

    print("\nBest model saved as: best_freight_model.pkl")

    return best_model


if __name__ == "__main__":
    best_model = main()
    