from data_preprocessing import load_invoice_data, apply_label, split_data, scaled_fetures
from model_eval import train_XGB, evaluation
import joblib

features= ['invioce_quantity',
            'invoice_dollar',
            'Freight',
            'total_item_quantity',
            'total_item_dollars',
            'avg_receiving_delay']

target = "flag_invoice"

def main():
    df = load_invoice_data()
    if df.empty:
        raise ValueError("Invoice data is empty! Check your database or query.")
    apply_label(df)
    X_train, X_test, y_train, y_test = split_data(df,features,target)
    X_train_scaled,X_test_scaled = scaled_fetures(X_train,X_test)
    grid = train_XGB(X_train_scaled,y_train)
    evaluation(grid.best_estimator_,X_test_scaled,y_test)
    joblib.dump(grid.best_estimator_,"C:/Users/User/Desktop/ML project/model/predict_flag_invoice.pkl")

if __name__ == "__main__":
    main()
    

