
import pandas as pd
import numpy as np
import streamlit as st
import sklearn


st.markdown("<h1 style='text-align: center; color: grey;'>Air Flights Tickets Price Prediction</h1>", unsafe_allow_html=True)
st.title(' ')

col3,col4,col5 = st.columns(3)
with col3:
    st.write(' ')
with col4:
    st.image('air.png',width = 550) 
with col5:
    st.write(' ')

#st.image('air.png',width = 500)


#load data
df = pd.read_csv('Clean_data.csv')
Preprocessor = pd.read_pickle('Preprocessor.pkl')
model = pd.read_pickle('model.pkl')

#App
col1 , col2 = st.columns(2)

with col1:
    airline = st.selectbox('Airline',df['Airline'].unique())
    Source = st.selectbox('Source',df['Source'].unique())
    Destination = st.selectbox('Destination',df['Destination'].unique())
    Route = st.selectbox('Route',df['Route'].unique())
    Add = st.selectbox('Additional Info',df['Additional_Info'].unique())
    Total_Stops = st.number_input('Total Stops',df['Total_Stops'].min(),df['Total_Stops'].max())
    Duration_in_minutes = st.number_input('Duration in minutes',df['Duration_in_minutes'].min() ,df['Duration_in_minutes'].max() )

with col2:
    Month = st.number_input('Departure Month',df['Month'].min() ,df['Month'].max() )
    Dep_Time_day = st.number_input('Departure Day',df['Dep_Time_day'].min() ,df['Dep_Time_day'].max() )
    Dep_Time_hour = st.number_input('Departure Hour',df['Dep_Time_hour'].min() ,df['Dep_Time_hour'].max() )
    Dep_Time_minute = st.number_input('Departure Minute',df['Dep_Time_minute'].min() ,df['Dep_Time_minute'].max() )
    Arrival_Time_day = st.number_input('Arrival Day',df['Arrival_Time_day'].min() ,df['Arrival_Time_day'].max() )
    Arrival_Time_hour = st.number_input('Arrival Hour',df['Arrival_Time_hour'].min() ,df['Arrival_Time_hour'].max() )
    Arrival_Time_minute = st.number_input('Arrival Minute',df['Arrival_Time_minute'].min() ,df['Arrival_Time_minute'].max())

# Preprocessor
new_data = {'Airline':airline,'Source':Source,'Destination':Destination,'Route':Route,
            'Additional_Info':Add,'Total_Stops':Total_Stops,'Duration_in_minutes':Duration_in_minutes,'Month':Month,
            'Dep_Time_day':Dep_Time_day,'Dep_Time_hour':Dep_Time_hour,'Dep_Time_minute':Dep_Time_minute
            ,'Arrival_Time_day':Arrival_Time_day,'Arrival_Time_hour':Arrival_Time_hour,'Arrival_Time_minute':Arrival_Time_minute}


new_data = pd.DataFrame(new_data,index=[0])
new_data_Preprocessed = Preprocessor.transform(new_data)

log_price = model.predict(new_data_Preprocessed)
price = np.expm1(log_price)

# Output
if st.button('Predict'):
    st.markdown('## Price:')
    st.markdown(price.round(2))
