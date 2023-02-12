import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

st.write("""
# Simple Air flight Prediction App
This app predicts the **Airplane flight** Taxi-out!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    n_estimators = st.sidebar.slider('Number of estimators', 5, 1000, 50)
    random_state = st.sidebar.slider('Random state', 1, 64, 32)
    test_size = st.sidebar.slider('Random state', 0.0, 0.6, 0.2)
    data = {'n_estimators': n_estimators,
            'random_state': random_state,
            'test_size': test_size}
    df = pd.read_csv('M1_final.csv')
    X = df.iloc[:, [0,1,2,6,7,8,9,10,11,12,14,16,17,18,20,21]]   
    Y = df.iloc[:, 22]

    ## Train, test, split - training with 80%
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=32) 

    rf_reg = RandomForestRegressor(n_estimators=n_estimators, random_state=32) 
    rf_reg.fit(X_train, y_train)

    
    rf_reg_pred = rf_reg.predict(X_test)
    pd.DataFrame({'Actual': y_test, 'Predicted': rf_reg_pred})

    MAE = metrics.mean_absolute_error(y_test, rf_reg_pred)
    MSE = metrics.mean_squared_error(y_test, rf_reg_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, rf_reg_pred))
    R2 = metrics.r2_score(y_test, rf_reg_pred)

    features = pd.DataFrame(data, index=[])
    return features

sf = user_input_features()

st.subheader('User Input parameters')
st.write(sf)



st.subheader('Class labels and their corresponding index number')
st.write(sf.metrics.r2_score(y_test, rf_reg_pred))


st.subheader('Prediction')
#st.write(df.target_names[rf_reg_pred])
#st.write(prediction)
