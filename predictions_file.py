from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib

#we are loading the model using joblib

loaded_model = joblib.load("lrmodel.sav")
def predict_bills(df):
    dfcopy = df.copy()
    if 'id' in df.columns:
       df = df.drop(["id"], axis = 1)
    predictions=loaded_model.predict(df)
    proba = loaded_model.predict_proba(df)
    proba = pd.DataFrame(data = proba).round(3)
    conditions = [
    (proba[1] > 0.50) & (proba[0] < 0.50),
    (proba[0] > 0.50) & (proba[1] < 0.50)
    ]
    values = ['Vrai', 'Faux']
    proba['vrai/faux'] = np.select(conditions, values)
    df = pd.merge(proba, dfcopy, left_index=True, right_index=True)
    df['predictions']=predictions
    df = df[["id", "diagonal", "height_left", "height_right", "margin_low", "margin_up", "length", 
             0, 1, "predictions", "vrai/faux"]]
    return(df)

#This function import the machine learning model saved with joblib and call it in a variable loaded_model.
#The function predict_bills which take the dataframe as parameters, put in a variable the prediction results,
#that the model made, with the help of the .predict() function.
#Then we create a nex columns inside the dataframe that contain the prediction of the model.
#And at the end return the dataframe.