import pickle
import pandas as pd
import random
import streamlit as st

features = ['Volume', 'mom', 'mom1', 'mom2', 'mom3', 'ROC_5', 'ROC_10', 'ROC_15', 'ROC_20', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200', 'DTB4WK', 'DTB3', 'DTB6', 'DGS5', 'DGS10', 'Oil', 'Gold', 'DAAA', 'DBAA', 'GBP', 'JPY', 'CAD', 'CNY', 'AAPL', 'AMZN', 'GE', 'JNJ', 'JPM', 'MSFT', 'WFC', 'XOM', 'FCHI', 'FTSE', 'GDAXI', 'HSI', 'SSEC', 'TE1', 'TE2', 'TE3', 'TE5', 'TE6', 'DE1', 'DE2', 'DE4', 'DE5', 'DE6', 'CTB3M', 'CTB6M', 'CTB1Y', 'AUD', 'Brent', 'CAC-F', 'copper-F', 'WIT-oil', 'DAX-F', 'DJI-F', 'EUR', 'FTSE-F', 'gold-F', 'HSI-F', 'KOSPI-F', 'NASDAQ-F', 'GAS-F', 'Nikkei-F', 'NZD', 'silver-F', 'RUSSELL-F', 'S&P-F', 'CHF', 'Dollar index-F', 'Dollar index', 'wheat-F', 'XAG', 'XAU']
minmax = {'Volume': [-1407.48353119, 864.815580543], 'ROC_5': [-17.8961920029, 12.3726452013], 'ROC_10': [-21.7031509251, 13.1896691147], 'ROC_15': [-20.6802507889, 20.7616271106], 'ROC_20': [-22.5843996839, 21.5869021901], 'EMA_10': [601.084229913, 23462.4825344], 'EMA_20': [612.036534382, 23335.9117003], 'EMA_50': [630.026731262, 22932.2257313], 'EMA_200': [654.039747615, 21605.5256269], 'DTB4WK': [-0.03, 1.28], 'DTB3': [-0.02, 1.24], 'DTB6': [0.0, 1.37], 'DGS5': [0.0, 2.75], 'DGS10': [0.0, 4.01], 'Oil': [-1.0, 0.1195112638], 'Gold': [-1.0, 0.0495772483], 'DAAA': [0.0, 5.49], 'DBAA': [0.0, 6.51], 'GBP': [-0.0276571724, 0.028889031], 'JPY': [-0.0314668863, 0.0381960835], 'CAD': [-0.0189046942, 0.0265820844], 'CNY': [-0.0118374747, 0.0185815443], 'AAPL': [-0.1235579463, 0.0887413657], 'AMZN': [-0.1265683503, 0.1574570142], 'GE': [-0.0717423133, 0.1080450836], 'JNJ': [-0.032695647, 0.0538205639], 'JPM': [-0.0941488614, 0.0843837565], 'MSFT': [-0.1139954603, 0.1045223581], 'WFC': [-0.0904402654, 0.0806803751], 'XOM': [-0.061881823, 0.0551593808], 'FCHI': [-0.0804249826, 0.0965928516], 'FTSE': [-0.0466732847, 0.0516103654], 'GDAXI': [-0.0682332116, 0.0534850873], 'HSI': [-0.0660100027, 0.0660220655], 'SSEC': [-0.0849089545, 0.062260405], 'TE1': [0.0, 3.85], 'TE2': [0.0, 3.83], 'TE3': [0.0, 3.74], 'TE5': [-0.27, 0.26], 'TE6': [-0.19, 0.41], 'DE1': [0.0, 1.54], 'DE2': [-2.4, 5.97], 'DE4': [-1.34, 6.33], 'DE5': [-1.21, 6.41], 'DE6': [-1.03, 6.44], 'CTB3M': [-1.0, 1.5], 'CTB6M': [-1.0, 0.3055555556], 'CTB1Y': [-1.0, 0.1588785047], 'AUD': [-3.79, 3.75], 'Brent': [-8.57, 10.98], 'CAC-F': [-8.08, 9.64], 'copper-F': [-7.25, 7.08], 'WIT-oil': [-8.67, 12.32], 'DAX-F': [-6.63, 5.52], 'DJI-F': [-5.93, 4.36], 'EUR': [-2.62, 3.07], 'FTSE-F': [-4.83, 4.91], 'gold-F': [-9.35, 7.14], 'HSI-F': [-7.39, 5.79], 'KOSPI-F': [-6.18, 5.75], 'NASDAQ-F': [-6.81, 5.74], 'GAS-F': [-11.25, 18.17], 'Nikkei-F': [-8.67, 7.41], 'NZD': [-4.12, 3.43], 'silver-F': [-17.75, 12.81], 'RUSSELL-F': [-7.54, 7.39], 'S&P-F': [-7.22, 5.44], 'CHF': [-15.76, 9.68], 'Dollar index-F': [-2.39, 2.18], 'Dollar index': [-4.28, 3.98], 'wheat-F': [-11.06, 12.31], 'XAG': [-12.93, 6.53], 'XAU': [-8.49, 4.8]}
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
        st.session_state[features[i]] = random.randint(minmax[features[i]][0], minmax[features[i]][1])

for i in range(len(features)):
    data[features[i]] = placeholders[i].number_input(features[i], min_value=minmax[features[i]][0], max_value=minmax[features[i]][1], key=features[i])

st.write("Model")
model = st.radio("Select the model to use:",
                    ("Linear", "Decision Tree", "Random Forest"))

def predict(data=data):
    df = pd.DataFrame(data, index=[0])
    imputer = pickle.load(open("imputer.pickle", "rb"))
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

    st.success('%.7f' % pred)

if st.button("Predict closing price"):
    predict()





