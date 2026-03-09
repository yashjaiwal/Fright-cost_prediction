from data_preprocessing import load_invoice_data, apply_label, split_data, scaled_fetures
from model_evalfile import train_XGB, evaluation

fetures= ['invioce_quantity',
         'invoice_dollar',
         'Freight',
         'days_po_to_invoice',
         'total_item_quantity',
         'total_item_dollars',
         'avg_receiving_delay']

target = "flag_invoice"

def main():
    df = load_invoice_data()
    df = apply_label(df)
    X_train, X_test, y_train, y_test = split_data(df,fetures,target)
    X_train_scaled,X_test_scaled = scaled_fetures(X_train,X_test)
    grid = train_XGB(X_train_scaled,y_train)
    evaluation(grid.best_estimator_,X_test_scaled,y_test)
    joblib.dump(grid.best_estimator_,"C:/Users/User/Desktop/ML project/fright_cost_prediction/model/predict_flag_invoice.pkl")

if __name__ == "__main__":
    main()
    

