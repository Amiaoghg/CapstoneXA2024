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
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



image = Image.open("1.jpg")

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

def send_email(to_email, subject, body):
    from_email = "your_email@example.com"
    password = "your_email_password"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.example.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        st.success('Email sent successfully')
    except Exception as e:
        st.error(f'Failed to send email: {e}')

with st.sidebar: 
    selected = option_menu(
        menu_title='Navigation',
        options=['View', 'History','About'],
        menu_icon='arrow-down-right-circle-fill',
        icons=['bookmark-check', 'book','book'],
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
        
     if prediction[0] >= 1:
        subject = "Health Alert"
        body = f"Your predicted Addiction Severity (AS) is {prediction[0]}. Please take necessary actions."
        send_email(email_address, subject, body)



if selected == 'About':
        
        st.title("WHAT IS OPIOID USE DISORDER?")
        st.write("Opioid use disorder (OUD) is a mental health condition in which a problematic pattern of opioid misuse causes distress and/or impairs your daily life.Some examples of prescription opioids and opiates include oxycodone, oxymorphone, morphine, codeine and fentanyl. Opioids and opiates can become addictive because they not only dull pain, but can also produce a sense of euphoria in some people. This, combined with tolerance build (needing to increase doses to produce the same effect) can lead to opioid use disorder.")
        st.title("OPIOID EPIDEMIC IN NUMBERS")
        st.write("Opioid use disorder is common. It affects over 20 million people worldwide and over 3 million people in the United States. Opioid use disorder is an epidemic in the U.S.Opioids are responsible for over 120,000 deaths worldwide every year")
        st.title("SYMPTOMS OF OUD")
        st.write("If you have a physical dependence on opioids, you may experience the following withdrawal symptoms if you stop taking them.Generalize pain, chills and fever, diarrhea, dilated pupils, restlessness and agitation, anxiety, nausea and vomiting, intense carvings, elevated heart rate and blood pressure, sweating and insomnia.")
        st.title("TREATMENTS FOR OUD")
        st.write("Three U.S. Food and Drug Administration (FDA)-approved medications are commonly used to treat opioid use disorder")
        st.write("1. Methadone")
        st.write("2. Buprenorphine")
        st.write("3. Naltrexone")

