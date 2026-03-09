from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import GridSearchCV

def train_XGB(X_train,y_train):
    xgb = XGBClassifier()
    param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1]
    }
    grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid

def evaluation(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    print(f"Model : {model.__class__.__name__}")
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred)
    recc = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    con = confusion_matrix(y_test,y_pred)
    print(f"accuracy:{acc}")
    print(f"precision_score:{prec}")
    print(f"recall_score:{recc}")
    print(f"f1_score:{f1}")
    print(f"confusion_matrix:{con}")

