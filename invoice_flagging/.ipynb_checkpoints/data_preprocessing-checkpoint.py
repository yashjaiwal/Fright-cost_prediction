import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import sqlite3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_invoice_data():
    conn = sqlite3.connect(r"C:\Users\User\Desktop\ML project\Data\inventory.db")
    query = """WITH purchase_agg_df as(
    select
    p.PONumber,
    count(distinct p.Brand) as total_brand,
    sum(p.Quantity) as total_item_quantity,
    sum(p.Dollars) as total_item_dollars,
    avg(julianday(p.ReceivingDate) - julianday(p.PODate)) as avg_receiving_delay
    from purchases p
    group by p.PONumber
    )
    select
    vi.Quantity as invioce_quantity,
    vi.Dollars as invoice_dollar,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate)) as days_po_to_invoice,
    (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) as days_to_pay,
    pa.total_brand,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay
    from vendor_invoice vi
    left join purchase_agg_df pa
    on vi.PONumber = pa.PONumber"""
    df = pd.read_sql_query(query,conn)
    conn.close()
    return df


def create_invoice_rsk(row):
    if (abs(row["invoice_dollar"] - row["total_item_dollars"])>5):
        return 1

    if row["avg_receiving_delay"]>10:
        return 1
    else:
        return 0

def apply_label(df):
    df["flag_invoice"] = df.apply(create_invoice_rsk,axis = 1)

def split_data(df,features,target):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scaled_fetures(X_train,X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler,"C:/Users/User/Desktop/ML project/fright_cost_prediction/model/scaler.pkl")
    return X_train_scaled,X_test_scaled



    