import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pypickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing

loaded_model = pypickle.load('model.pkl')

def prediction(data):
    
     expresso_df = pd.DataFrame(data)
     expresso_df.iloc[2].replace({"D 3-6 month": 3, "E 6-9 month": 6, "K > 24 month": 24,
                        "I 18-21 month": 18, "H 15-18 month": 15, "G 12-15 month": 12,
                        "J 21-24 month": 21, "F 9-12 month": 9}, inplace=True)

     label = preprocessing.LabelEncoder()

     expresso_df.iloc[1] = label.fit_transform(expresso_df.iloc[1])
     expresso_df.iloc[16] = label.fit_transform(expresso_df.iloc[16])

     num_data = expresso_df.drop([0, 14]).values.reshape(1, -1)

     pred = loaded_model.predict(num_data)

     if pred[0] == 0 :
          return "Customer is not going, but staying with Expresso"
     else:
          return "The Customer will leave Expresso"
     

def main():
    st.title(" Expresso Churn Prediction Challenge")
    user_id = st.text_input(" ID of Client")
    REGION = st.text_input("Region of Each Client")
    TENURE = st.text_input(" Duration in the Network")
    MONTANT = st.number_input("Top-Up Amount")
    FREQUENCE_RECH = st.number_input("Number of times the Client Refilled")
    REVENUE = st.number_input("Monthly Income of Each Client")
    ARPU_SEGMENT = st.number_input("Income Over 90 days/3")
    FREQUENCE = st.number_input("Number of times the Client has made an Income")
    DATA_VOLUME = st.number_input("Number of Connections")
    ON_NET = st.number_input("Inter Expresso Call")
    ORANGE = st.number_input("Call to Orange")
    TIGO = st.number_input("Call to Tigo")
    ZONE1 = st.number_input("Call to Zones1")
    ZONE2 = st.number_input("Call to Zones2")
    MRG= st.text_input("A Client Who is Going")
    REGULARITY = st.number_input("Number of times the Client is Active for 90 days")
    TOP_PACK = st.text_input("The Most Active Packs")
    FREQ_TOP_PACK = st.number_input("Number of times the Client has Activated the Top Pack Packages")

    Churn  = ""

    if st.button("Result"):
          Churn = prediction([user_id, REGION, TENURE, MONTANT, FREQUENCE_RECH, REVENUE,
          ARPU_SEGMENT, FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO,
          ZONE1, ZONE2, MRG, REGULARITY, TOP_PACK, FREQ_TOP_PACK])
          

    st.success(Churn)


if __name__ == "__main__":
     main()

 