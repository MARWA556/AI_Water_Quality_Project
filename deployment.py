import json
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

model_image_matrix = {
    "Logistic Regression": "assets\\LR_matrix.png",
    "Decision Tree": "assets\\DT_matrix.png",
    "Random Forest": "assets\\RF_matrix.png",
    "Hist Gradient Boost": "assets\\HIST_matrix.png",
    "KNN": "assets\\KNN_matrix.png",
    "SVM": "assets\\SVM_matrix.png",
    "ada Boost classifier": "assets\\ADA_matrix.png",
    "XGBOOST": "assets\\XGB_matrix.png",
    "Naive": "",
}

accuracies = {
    "None": 0.0,
    "Logistic Regression": 0.62*100 ,
    "Decision Tree": 0.72*100,
    "Random Forest": 0.8*100,
    "Hist Gradient Boost": 0.78*100,
    "KNN": 0.6*100,
    "SVM": 0.62*100,
    "ada Boost classifier": 0.74*100,
    "XGBOOST": 0.82*100,
    "Naive" :0.61*100
}
    

st.set_page_config(
    page_title='water quality',
    page_icon=':gem:',
    initial_sidebar_state='auto'  # Collapsed sidebar
)

def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
model=joblib.load(open("water_quality",'rb'))

def predict(selected_model, model, ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    A = np.array([ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]).reshape(1, -1)
    
    if selected_model == "Logistic Regression":
        prediction = model['log'].predict(A)
        s=prediction
    elif selected_model == "Decision Tree":
        prediction = model['dt'].predict(A)
        b=prediction
    elif selected_model == "Random Forest":
        prediction = model['rf'].predict(A)
    elif selected_model == "Hist Gradient Boost":
        prediction = model['hgbc'].predict(A)
    elif selected_model == "KNN":
        prediction = model['knn'].predict(A)
    elif selected_model == "SVM":
        prediction = model['svm_classifier'].predict(A)
    elif selected_model == "ada Boost classifier":
        prediction = model['model_ada'].predict(A)
    elif selected_model == "XGBOOST":
        prediction = model['model_xgboost'].predict(A)
    elif selected_model == "Naive":
        prediction = model['naive_bayes_model'].predict(A)
    else:
        prediction = None
    #print(s)
    #print(b)
    return prediction

def get_accuracy(selected_model):
    return accuracies.get(selected_model)

with st.sidebar:
    choose = option_menu(None, ["Home", "Graphs"],
                         icons=['house', 'kanban','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose=='Home':
       st.write('# Water Quality')
       st.write('---')
       st.subheader("Select a Model")
       selected_model = st.selectbox("Select Model", list(accuracies.keys()))
       st.write('---')

       accuracy = get_accuracy(selected_model)
       if accuracy != "Model not found":
        st.write(f"### The accuracy of {selected_model} is {accuracy}%")
        st.write('_________________')
       else:
        st.write("Model not found")
        st.write('_________________')

       model_image = model_image_matrix.get(selected_model)
       if model_image:
           st.image(f"{model_image}", caption=selected_model, use_column_width=True)
       
       st.write('---')

       # User input
       st.subheader('Enter your water quality details')
       ph = st.number_input("Enter ph value: ",min_value=0.0,max_value=14.0, format="%.6f")
       Hardness = st.number_input("Enter Hardness value: ",min_value=0.0,max_value=323.0, format="%.6f")
       Solids = st.number_input("Enter Solids value: ",min_value=0.0,max_value=61227.0, format="%.6f")
       Chloramines= st.number_input("Enter Chloramines value: ",min_value=0.0,max_value=13.0, format="%.6f")
       Sulfate = st.number_input("Enter Sulfate value: ",min_value=0.0,max_value=481.0, format="%.6f")
       Conductivity = st.number_input("Enter Conductivity value: ",min_value=0.0,max_value=753.0, format="%.6f")
       Organic_carbon = st.number_input("Enter Organic_carbon value: ",min_value=0.0,max_value=28.0, format="%.6f")
       Trihalomethanes = st.number_input("Enter Trihalomethanes value: ",min_value=0.0,max_value=124.0, format="%.6f")
       Turbidity = st.number_input("Enter Turbidity value: ",min_value=0.0,max_value=6.0, format="%.6f")
       # Predict the cluster
       
       sample_prediction = predict(selected_model, model, ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
       

       if st.button("Predict"):
        if sample_prediction == 0:
            st.write('### WATER POLLUTED')
        elif sample_prediction == 1:
            st.write(' ### WATTER NOT POLLUTED')



elif choose == 'Graphs':
    st.write('# water quality Graphs')
    st.write('---')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("## pairplot Graph:")
    st.image("assets\\pairplot.png")
    st.write("## boxplot Graph:")
    st.image("assets\\boxplot.png")
    st.write("## correlation heatmap Graph:")
    st.image("assets\\correlation heatmap.png")
    st.write("## scatterplot Graph")
    st.image("assets\\scatterplot.png")
    st.write("## histplot Graph")
    st.image("assets\\histplot.png")
    
    
    data = pd.read_csv('water_potability.csv')
    # Create a DataFrame
    df = pd.DataFrame(data)
    