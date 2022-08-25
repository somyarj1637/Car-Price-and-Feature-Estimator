from asyncio.windows_utils import pipe
from tkinter.tix import COLUMN
from sklearn import set_config
import pickle
set_config(display='diagram')
import pandas as pd
import numpy as np
from pyexpat import model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import streamlit.components.v1 as components

import warnings
warnings.filterwarnings('ignore')

st.title("Car Price and Feature Estimator")


car=pickle.load(open('./model/car.pkl','rb'))
predict_pipe=pickle.load(open('./model/predict_pipe.pkl','rb'))


nav = st.sidebar.radio("Navigation",["Home","Analysis","Prediction","Contribute"])
if nav == "Home":
    st.subheader("Car Details fetcher :")
    
    car_mod=car
    car_mod['car_name']=car['Make']+" "+car['Model']
    car_mod=car_mod.drop(columns=['Make','Model'])

    def extract_feature(c_name):
        s = car_mod.loc[car_mod['car_name'].str.contains(c_name, case=False)]
        return s.head(5)
    l=car_mod['car_name'].unique()
    x_name=st.selectbox('Car Company Name',l)
    st.text("Fetching the details of "+x_name)
    top=extract_feature(x_name)
    st.table(top)





    st.subheader('Graph Plots :')
    graph = st.selectbox("What kind of Graph ? ",["Vehicle Style vs Popularity","Transmission Type","Vehicle Size","Brand Price"])

    if graph=='Vehicle Style vs Popularity':
        plt.figure(figsize=(50,20))
        sns.barplot(x=car['Vehicle Style'],y=car['Popularity'])
        plt.tight_layout()
        st.pyplot(plt)
    if graph=='Transmission Type':
        plt.figure(figsize=(40,20))
        car['Transmission Type'].value_counts().plot(kind='pie')
        plt.tight_layout()
        st.pyplot(plt)
    if graph=='Vehicle Size':
        plt.figure(figsize=(40,20))
        car['Vehicle Size'].value_counts().plot(kind='pie')
        plt.tight_layout()
        st.pyplot(plt)
    if graph=='Brand Price':
        plt.figure(figsize=(50,20))
        sns.scatterplot(data=car,x='Make',y='MSRP')
        plt.tight_layout()
        st.pyplot(plt)  

    st.subheader('Pipeline used in this model :')
    st.image('./data/pipeline_pic.png')


    st.subheader('Car Table Data :')
    show_tb=st.radio('Table Display Selection',['Sample Data','Full Data (This may take a few minutes to load)'])
    if show_tb=='Sample Data':
        st.table(car.head(11))    
    if show_tb=='Full Data (This may take a few minutes to load)':
        st.table(car)

if nav== "Analysis":
    st.header("CAR Profile Analysis")

    HtmlFile = open("./data/analysis.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code, height=2600 ,width=700)


if nav == "Prediction":
    st.header("Get Your Car Price")

    st.text("Enter all the details :")


    year =int(st.number_input("Year",0,2022))
    engine_hp=st.number_input("Engine HP",100.0,10000.0 )
    t_type=st.radio("Transmission Type",  ['MANUAL','AUTOMATIC','AUTOMATED_MANUAL', 'DIRECT_DRIVE','UNKNOWN'] )
    n_door=st.number_input("Number of Doors ",2.0,12.0)
    v_size=st.radio("Vehicle Size",['Large','Midsize','Compact'])
    v_style=st.selectbox("Vehicle Style",['4dr SUV', 'Sedan', 'Passenger Minivan', 'Extended Cab Pickup',
       'Wagon', 'Coupe', '2dr Hatchback', 'Regular Cab Pickup',
       '4dr Hatchback', 'Crew Cab Pickup', 'Cargo Van', 'Passenger Van',
       '2dr SUV', 'Convertible', 'Cargo Minivan', 'Convertible SUV'])
    pop=int(st.number_input("Popularity",0,8000))
    mileage=st.number_input("Mileage (in KML)",0.0,100.0)


    val=np.array([[year,engine_hp,t_type,n_door,v_size,v_style,pop,mileage]]).reshape(-1,8)
    df=pd.DataFrame(data=val,columns=['Year', 'Engine HP', 'Transmission Type', 'Number of Doors','Vehicle Size', 'Vehicle Style', 'Popularity', 'mileage KML'])
    pred =predict_pipe.predict(df)


    if st.button("Predict"):
        st.success(f"Your predicted salary is {pred}")




if nav == "Contribute":
    st.header("Contribute to our dataset")

    make=st.text_input("Company Name")
    model_name=st.text_input("Model Name")
    year =int(st.number_input("Year",0,2022))
    engine_hp=st.number_input("Engine HP",100.0,10000.0 )
    t_type=st.radio("Transmission Type",  ['MANUAL','AUTOMATIC','AUTOMATED_MANUAL', 'DIRECT_DRIVE','UNKNOWN'] )
    n_door=st.number_input("Number of Doors ",2.0,12.0)
    m_cat=st.text_input('Market Category')
    v_size=st.radio("Vehicle Size",['Large','Midsize','Compact'])
    v_style=st.selectbox("Vehicle Style",['4dr SUV', 'Sedan', 'Passenger Minivan', 'Extended Cab Pickup',
       'Wagon', 'Coupe', '2dr Hatchback', 'Regular Cab Pickup',
       '4dr Hatchback', 'Crew Cab Pickup', 'Cargo Van', 'Passenger Van',
       '2dr SUV', 'Convertible', 'Cargo Minivan', 'Convertible SUV'])
    pop=int(st.number_input("Popularity",0,8000))
    msrp=int(st.number_input('MSRP',0,1000000))
    mileage=st.number_input("Mileage (in KML)",0.0,100.0)


    if st.button("submit"):
        to_add = {"YearsExperience":[ex],"Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("./data/data.csv",mode='a',header = False,index= False)
        st.success("Submitted Successfully")
