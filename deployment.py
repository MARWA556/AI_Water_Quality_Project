import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import joblib
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title='water quality',
    page_icon=':gem:',
    initial_sidebar_state='collapsed'  # Collapsed sidebar
)

def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
model=joblib.load(open("water_quality",'rb'))

def predict(ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity):

    features = np.array([ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction
with st.sidebar:
    choose = option_menu(None, ["Home", "Graphs"],
                         icons=['house', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose=='Home':
       st.write('# water quality')
       st.write('---')
       st.subheader('Enter your water quality details')
       # User input
       ph = st.number_input("Enter ph value: ",min_value=0,max_value=14)
       Hardness = st.number_input("Enter Hardness value: ",min_value=47,max_value=323)
       Solids = st.number_input("Enter Solids value: ",min_value=320,max_value=61227)
       Chloramines= st.number_input("Enter Chloramines value: ",min_value=0,max_value=13)
       Sulfate = st.number_input("Enter Sulfate value: ",min_value=129,max_value=481)
       Conductivity = st.number_input("Enter Conductivity value: ",min_value=181,max_value=753)
       Organic_carbon = st.number_input("Enter Organic_carbon value: ",min_value=2,max_value=28)
       Trihalomethanes = st.number_input("Enter Trihalomethanes value: ",min_value=0,max_value=124)
       Turbidity = st.number_input("Enter Turbidity value: ",min_value=1,max_value=6)
       # Predict the cluster
       sample_prediction = predict(ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity)

       if st.button("Predict"):
        if sample_prediction == 0:
            st.warning("Predicted Salary: Low")
            st.write("This indicates a low salary.")
        elif sample_prediction == 1:
            st.success("Predicted Salary: High")
            st.write("This indicates a high salary.")
            st.balloons()
              
 



elif choose == 'Graphs':
    st.write('# water quality Graphs')
    st.write('---')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("## pairplot Graph:")
    st.image("pairplot.png")
    st.write("## boxplot Graph:")
    st.image("boxplot.png")
    st.write("## correlation heatmap Graph:")
    st.image("correlation heatmap.png")
    st.write("## scatterplot Graph")
    st.image("scatterplot.png")
    st.write("## Age period Vs Gender Graph")
    st.image("output5.png")
    st.write("## Age Period Vs Income Graph")
    st.image("output6.png")
    st.write("## Workclass Vs Income Graph")
    
    data = pd.read_csv('water_potability.csv')
    # Create a DataFrame
    df = pd.DataFrame(data)
    
