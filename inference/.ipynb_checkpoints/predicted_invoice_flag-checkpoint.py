import joblib
import pandas as pd

model_path = "C:/Users/User/Desktop/ML project/model/predict_flag_invoice.pkl"

def load_model(model_path:str = model_path):
    with open (model_path,"rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    model = load_model()
    df = pd.DataFrame(input_dat)
    df["Predict_Flag"] = model.predict(df).round()
    return df

if __name__ == "__main__":
    
    