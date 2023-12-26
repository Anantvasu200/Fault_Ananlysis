import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
from datetime import datetime

@st.cache
def load_data(file_path):
    return pd.read_csv(file_path, encoding='latin-1')

@st.cache
def preprocess_data(df):
    # Perform data preprocessing steps here
    return df


st.set_page_config(
    page_title="Fault Analysis Dashboard !!!",
    page_icon="📊",
    layout="wide",)

st.title(" :bar_chart: Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem}</style>', unsafe_allow_html=True)

fl =st.file_uploader("Upload your file", type=["csv","xlsx","xls","txt"])
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding='latin-1')
else:
    os.chdir(r"C:\Users\Anant\OneDrive\Desktop\project\Interactive Dasboard")
    df = pd.read_csv(r"C:\Users\Anant\OneDrive\Desktop\project\Interactive Dasboard\psc_final_dataset.csv", encoding='latin-1')

#Main Developement starts here
col1, col2 = st.columns((2))
df['Fault_Date'] = pd.to_datetime(df['OPEN_DATE'])

# Getting the min and max date from the dataset
startDate = pd.to_datetime(df['OPEN_DATE']).min()
endDate = pd.to_datetime(df['CLOSE_DATE']).max()

with col1:
    date1 = pd.to_datetime(st.date_input('OPEN_DATE', startDate))
with col2:
    date2 = pd.to_datetime(st.date_input('CLOSE_DATE', endDate))

# Filtering based on the date range
df['OPEN_DATE'] = pd.to_datetime(df['OPEN_DATE'])  # Convert 'OPEN_DATE' to Timestamp
df['CLOSE_DATE'] = pd.to_datetime(df['CLOSE_DATE'])  # Convert 'CLOSE_DATE' to Timestamp

df = df[(df['OPEN_DATE'] >= date1) & (df['CLOSE_DATE'] <= date2)].copy()

st.sidebar.header("Choose your options here")
CIRCLE = st.sidebar.multiselect("Select the Circle", df['CIRCLE'].unique())
if len(CIRCLE) > 0:
    df = df[df['CIRCLE'].isin(CIRCLE)]

District = st.sidebar.multiselect("Select the District", df['DISTRICT'].unique())
if len(District) > 0:
    df = df[df['DISTRICT'].isin(District)]

ZONE_ID = st.sidebar.multiselect("Select the Zone", df['ZONE_ID'].unique())
if len(ZONE_ID) > 0:
    df = df[df['ZONE_ID'].isin(ZONE_ID)]
    last_protection_visitor_name = df['LAST_PROTECTION_VISITOR_NAME'].iloc[0]  # Get the first value, assuming it's the same for all rows
    st.write(f"LAST_PROTECTION_VISITOR's_NAME for {ZONE_ID[0]}: {last_protection_visitor_name}")


Feeder = st.sidebar.multiselect("Select the Feeder", df['FEEDER_NAME'].unique())
if len(Feeder) > 0:
    df = df[df['FEEDER_NAME'].isin(Feeder)]
    last_protection_visitor_name = df['LAST_PROTECTION_VISITOR_NAME'].iloc[0]  # Get the first value, assuming it's the same for all rows
    st.write(f"LAST_PROTECTION_VISITOR's_NAME for {Feeder[0]}: {last_protection_visitor_name}")

substation = st.sidebar.multiselect("Select the Substation", df['SUBSTATION_NAME'].unique())
if len(substation) > 0:
    df = df[df['SUBSTATION_NAME'].isin(substation)]

Grid = st.sidebar.multiselect("Select the Grid", df['GRID_NAME'].unique())
if len(Grid) > 0:
    df = df[df['GRID_NAME'].isin(Grid)]



grouped_df = df.groupby(by=['FAULT_TYPE', 'FEEDER_NAME', 'ZONE_ID', 'FAULT_REASON']).size().reset_index(name='Number of Faults')

col1, col2 = st.columns((2))

#Graphical Representation of Faults
with col2:
    st.header("Where Faults Meet Their Reasons")
    
    # Dropdown to select 'Feeder'
    selected_feeder = st.selectbox("Select a Feeder", df['FEEDER_NAME'].unique())
    
    # Filtered DataFrame based on the selected feeder
    filtered_df = grouped_df[grouped_df['FEEDER_NAME'] == selected_feeder]
    
    # Bar chart for faults based on 'Reason' for the selected feeder
    fig = px.bar(filtered_df, x='FAULT_REASON', y='Number of Faults', color='FAULT_TYPE', title=f'Faults by Reason for {selected_feeder}')
    st.plotly_chart(fig)

col1, col2 = st.columns((2))

with col2:
    with st.expander("Where Faults Meet Their Reasons"):
        st.write(filtered_df.style.background_gradient(cmap="Blues"))
        csv = filtered_df.to_csv(index=False)  # some strings <-> bytes conversions necessary here
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='category_df.csv',
            mime='text/csv',
            key="download_button_2",  # Unique key for the second download button
            help="Click here to download the data"
        )

with col1:
    st.header("Fault Category")
    
    # Pie chart for overall fault types
    fig = px.pie(grouped_df, values='Number of Faults', names='FAULT_TYPE', title='Fault Category')
st.plotly_chart(fig, margin=dict(l=10, r=10, t=20, b=20))

with col1:
    with st.expander("Fault Category"):
# Adjust the margin to move the graph slightly to the right
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write(filtered_df.style.background_gradient(cmap="Blues"))
        csv = filtered_df.to_csv(index=False)  # some strings <-> bytes conversions necessary here
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='category_df.csv',
            mime='text/csv',
            key="download_button_1",  # Unique key for the first download button
            help="Click here to download the data"
        )


        
# Time series Analysis
st.header("Time Series Analysis - Open and Close Dates of Faults")

# Filtered DataFrame based on the selected feeder
time_series_df = df[df['FEEDER_NAME'] == selected_feeder]

# Line chart for faults over time with open and close dates
fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=time_series_df['OPEN_DATE'], y=time_series_df['CLOSE_DATE'], mode='lines+markers', name='Faults'))

fig3.update_layout(title=f'Faults Over Time for Feeder: {selected_feeder}',
                   xaxis_title='OPEN_DATE',
                   yaxis_title='CLOSE_DATE')

st.plotly_chart(fig3)

## ZSO_NAME display based on selected Zone
selected_zone_id = st.sidebar.selectbox("Select the Zone ID", df['ZONE_ID'].unique())
if selected_zone_id is not None:
    selected_zone_data = df[df['ZONE_ID'] == selected_zone_id].head(1)  # Assuming ZSO_NAME is the same for all rows with the same ZONE_ID
    
    if 'ZSO_NAME' in selected_zone_data.columns:
        selected_zso_name = selected_zone_data['ZSO_NAME'].iloc[0]
        st.sidebar.write(f"ZSO_NAME for {selected_zone_id}: {selected_zso_name}")
    else:
        st.sidebar.warning("ZSO_NAME column not found for the selected ZONE_ID.")
