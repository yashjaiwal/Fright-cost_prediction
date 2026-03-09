import joblib
import pandas as pd

model_path = "C:/Users/User/Desktop/ML project/model/best_freight_model.pkl"

def load_model(model_path:str = model_path):
    with open (model_path,"rb") as f:
        model = joblib.load(f)
    return model

def predict_fright_cost(input_dat):
    model = load_model()
    df = pd.DataFrame(input_dat)
    df["Predict_Fright"] = model.predict(df).round()
    return df

if __name__ == "__main__":
    sample_data = {
    "Dollars": [1000, 200, 390]
    }
    
    predict = predict_fright_cost(sample_data)
    
    print(predict)

