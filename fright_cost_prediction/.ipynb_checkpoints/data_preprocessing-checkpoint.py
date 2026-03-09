import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
def load_vendore_invoice(data_path: str):
    conn = sqlite3.connect(data_path)
    
    query = "select * from vendor_invoice"
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    return df

def prepare_feature(df:pd.DataFrame):
 
    # Features
    X = df[["Dollars"]]
    
    # Target
    y = df["Freight"]
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test