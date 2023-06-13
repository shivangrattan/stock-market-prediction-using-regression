import pickle
import pandas as pd
import random
import streamlit as st

features = ['Volume', 'mom', 'mom1', 'mom2', 'mom3', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200', 'DTB4WK', 'DTB3', 'DTB6', 'DGS5', 'DGS10', 'Oil', 'Gold', 'DAAA', 'DBAA', 'GBP', 'JPY', 'CAD', 'CNY', 'AAPL', 'AMZN', 'GE', 'JNJ', 'JPM', 'MSFT', 'WFC', 'XOM', 'FCHI', 'FTSE', 'GDAXI', 'HSI', 'SSEC', 'TE1', 'TE2', 'TE3', 'TE5', 'TE6', 'DE1', 'DE2', 'DE4', 'DE5', 'DE6', 'CTB3M', 'CTB6M', 'CTB1Y', 'AUD', 'Brent', 'CAC-F', 'copper-F', 'WIT-oil', 'DAX-F', 'DJI-F', 'EUR', 'FTSE-F', 'gold-F', 'HSI-F', 'KOSPI-F', 'NASDAQ-F', 'GAS-F', 'Nikkei-F', 'NZD', 'silver-F', 'RUSSELL-F', 'S&P-F', 'CHF', 'Dollar index-F', 'Dollar index', 'wheat-F', 'XAG', 'XAU']

tree = pickle.load(open("tree.pickle", "rb"))
forest = pickle.load(open("forest.pickle", "rb"))

st.title("Stock Closing Price Predictor")
placeholders = [None]*len(features)

data = dict.fromkeys(features)

for i in range(len(features)):
    placeholders[i] = st.empty()
    if features[i] not in st.session_state:
        st.session_state[features[i]] = 0

if st.button("Randomize"):
    for i in range(len(features)):
        st.session_state[features[i]] = random.randint(1,100000)

for i in range(len(features)):
    data[features[i]] = placeholders[i].number_input(features[i], key=features[i])

st.write("Model")
model = st.radio("Select the model to use:",
                    ("Linear", "Decision Tree", "Random Forest"))

def predict(data=data):
    df = pd.DataFrame(data)
    imputer = pickle.load(open("impupter.pickle", "rb"))
    scaler = pickle.load(open("scaler.pickle", "rb"))
    x = df[features]
    x = imputer.transform(x)
    x = scaler.transform(x)
    
    if model == "Linear":
        linear = pickle.load(open("linear.pickle", "rb"))
        pred = linear.predict(x)

    elif model == "Decision Tree":
        tree = pickle.load(open("tree.pickle", "rb"))
        pred = tree.predict(x)

    else:
        forest = pickle.load(open("forest.pickle", "rb"))
        pred = forest.predict(x)

    st.success(pred)

if st.button("Predict closing price"):
    predict()





