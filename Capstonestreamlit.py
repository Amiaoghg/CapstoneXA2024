import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import plotly.express as px
import plotly.graph_objects as go
import json
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.io as pio
import datetime

# Initialize df2 with the 'Date' column
if 'df2' not in st.session_state:
    st.session_state.df2 = pd.DataFrame(columns=['Heart rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic', 'Sleep Duration', 'Body Temperature', 'Respiratory Rate', 'Skin Conductance', 'Date'])

df = pd.read_csv("df.csv.csv")
df.rename(columns={'BloodPressureSystolic___': 'BloodPressureSystolic', 'BloodPressureDiastolic___': 'BloodPressureDiastolic', 'SleepDuration____': 'SleepDuration','RespiratoryRate____':'RespiratoryRate','SkinConductance____':'SkinConductance', 'Unnamed: 8':"ab",'low/remission _/__':'ba'}, inplace=True)
df.drop('ab', axis=1, inplace=True)
df.drop('ba', axis=1, inplace=True)
df['AddictionSeverity'] = df['AddictionSeverity'].replace({'Mild': '1', 'Moderate': '2', 'Severe': '3', 'Low/Remission': '0'})
df['AddictionSeverity'] = df['AddictionSeverity'].astype(float)
df.rename(columns={'AddictionSeverity': 'AS'}, inplace=True)
X = df.drop('AS', axis=1)
y = df['AS']

with st.sidebar: 
    selected = option_menu(
        menu_title='Navigation',
        options=['View', 'History'],
        menu_icon='arrow-down-right-circle-fill',
        icons=['bookmark-check', 'book'],
        default_index=0,
    )

if selected == 'View':
    st.title('Enter your data')

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with st.form("Insert stats"):
        col1 = st.number_input(label='Heart rate')
        col2 = st.number_input(label='Blood Pressure Systolic')
        col3 = st.number_input(label='Blood Pressure Diastolic')
        col4 = st.number_input(label='Sleep Duration')
        col5 = st.number_input(label='Body Temperature')
        col6 = st.number_input(label='Respiratory Rate')
        col7 = st.number_input(label='Skin Conductance')

        submitted = st.form_submit_button("View Data")

    if submitted:
        now = datetime.datetime.now()
        new_data = {
            'Heart rate': col1,
            'Blood Pressure Systolic': col2,
            'Blood Pressure Diastolic': col3,
            'Sleep Duration': col4,
            'Body Temperature': col5,
            'Respiratory Rate': col6,
            'Skin Conductance': col7,
            'Date': now
        }
        st.session_state.df2 = pd.concat([st.session_state.df2, pd.DataFrame([new_data])], ignore_index=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        new_data_scaled = scaler.transform([list(new_data.values())[:-1]])  # Exclude 'Date' for scaling
        prediction = knn.predict(new_data_scaled)
        st.write(f'Predicted class: {prediction[0]}')
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy: {accuracy:.2f}')

if selected == 'History':
    st.title('View your history')
    st.markdown("This page shows the history of your health, and trends on whether your addiction level has changed")

    if not st.session_state.df2.empty:
        # Ensure the 'Date' column is in datetime format to plot correctly
        st.session_state.df2['Date'] = pd.to_datetime(st.session_state.df2['Date'])

        # Melt the DataFrame to long format for plotting with plotly
        df_long = st.session_state.df2.melt(id_vars=['Date'], var_name='Variable', value_name='Value')

        # Create a line plot with all variables
        fig = px.line(df_long, x='Date', y='Value', color='Variable', 
                      title='Health Metrics Over Time', labels={'Value': 'Measurement'})
        st.plotly_chart(fig)
    else:
        st.write("No data to display.")
