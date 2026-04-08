import os
import cloudpickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import streamlit as st
import plotly.express as px
import boto3
from io import BytesIO
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# Streamlit page setup
st.set_page_config(page_title="SegmenAI - Customer Segmentation", layout="wide")
st.title("SegmenAI - Customer Segmentation System")
st.markdown("""
Analyze your customer base using **RFM segmentation** and **KMeans clustering**. 
Use the interactive charts to explore segments and understand customer behavior.
""")


# Load dataset from Cloudflare R2 using Streamlit Secrets
try:
    st.info("Loading dataset from Cloudflare R2 bucket")

    # Get credentials from Streamlit Secrets
    endpoint_url = st.secrets["R2_ENDPOINT_URL"]
    access_key = st.secrets["R2_ACCESS_KEY_ID"]
    secret_key = st.secrets["R2_SECRET_ACCESS_KEY"]
    bucket_name = st.secrets["R2_BUCKET_NAME"]

    # Connect to R2
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # List CSV files in bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)
    if 'Contents' not in objects:
        raise Exception(f"No objects found in bucket {bucket_name}")

    # Pick first CSV file
    file_name = next((obj['Key'] for obj in objects['Contents'] if obj['Key'].lower().endswith('.csv')), None)
    if file_name is None:
        raise Exception(f"No CSV file found in bucket {bucket_name}")

    # Read CSV
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(BytesIO(obj['Body'].read()), parse_dates=['InvoiceDate'], encoding='ISO-8859-1')

except Exception as e:
    st.error(f"Could not load dataset from Cloudflare R2: {e}")
    st.stop()


# Load pre-trained model & scaler
MODEL_PATH = "models/kmeans_rfm.pkl"
SCALER_PATH = "models/scaler_rfm.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        kmeans = cloudpickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = cloudpickle.load(f)
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()  # stop the app if models cannot be loaded


# RFM Feature Engineering
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                   # Frequency
    'TotalAmount': 'sum'                                      # Monetary
}).reset_index()

rfm.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalAmount':'Monetary'}, inplace=True)

# Remove rows with null CustomerID
rfm = rfm[rfm['CustomerID'].notnull()]

# Convert CustomerID to integer then string to remove decimals
rfm['CustomerID'] = rfm['CustomerID'].apply(lambda x: str(int(x)))

# Transform features & predict segments
X_scaled = scaler.transform(rfm[['Recency','Frequency','Monetary']])
rfm['Segment'] = kmeans.predict(X_scaled)

# Descriptive Segment Labels
segment_labels = {
    0: "Recent Moderate",
    1: "Inactive Low-Value",
    2: "High-Value Loyal",
    3: "Occasional Buyer"
}
rfm['Segment_Description'] = rfm['Segment'].map(segment_labels)

# Personalized Recommendations for Segments
recommendations = {
    "High-Value Loyal": "Reward with VIP offers, loyalty perks, early access to products.",
    "Inactive Low-Value": "Send re-engagement campaigns with discounts or reminders.",
    "Recent Moderate": "Upsell or cross-sell complementary products.",
    "Occasional Buyer": "Encourage engagement through seasonal campaigns or bundles."
}


rfm['Recommendation'] = rfm['Segment_Description'].map(recommendations)

# Metrics & Visualizations
st.metric("Silhouette Score", f"{silhouette_score(X_scaled, rfm['Segment']):.2f}")

# Customer Count per Segment
st.subheader("Customer Count per Segment")
fig_count = px.histogram(
    rfm, x='Segment_Description', color='Segment_Description',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Customer Count per Segment"
)
st.plotly_chart(fig_count, use_container_width=True)

# Monetary Contribution per Segment
st.subheader("Monetary Contribution per Segment")
monetary_per_segment = rfm.groupby('Segment_Description')['Monetary'].sum().reset_index()
fig_monetary = px.bar(
    monetary_per_segment, x='Segment_Description', y='Monetary',
    text='Monetary', color='Segment_Description',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Monetary Contribution per Segment"
)
fig_monetary.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
st.plotly_chart(fig_monetary, use_container_width=True)

# Average Recency per Segment
st.subheader("Average Recency per Segment")
recency_per_segment = rfm.groupby('Segment_Description')['Recency'].mean().reset_index()
fig_recency = px.bar(
    recency_per_segment, x='Segment_Description', y='Recency',
    text='Recency', color='Segment_Description',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Average Recency per Segment (days since last purchase)"
)
fig_recency.update_traces(texttemplate='%{text:.1f}', textposition='outside')
st.plotly_chart(fig_recency, use_container_width=True)

# Average Frequency per Segment
st.subheader("Average Frequency per Segment")
frequency_per_segment = rfm.groupby('Segment_Description')['Frequency'].mean().reset_index()
fig_frequency = px.bar(
    frequency_per_segment, x='Segment_Description', y='Frequency',
    text='Frequency', color='Segment_Description',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Average Frequency per Segment (number of purchases)"
)
fig_frequency.update_traces(texttemplate='%{text:.1f}', textposition='outside')
st.plotly_chart(fig_frequency, use_container_width=True)

# Interactive Customer Table
st.subheader("View Customers by Segment")
selected_segment = st.selectbox("Select a Segment", rfm['Segment_Description'].unique())
st.dataframe(
    rfm[rfm['Segment_Description'] == selected_segment][['CustomerID','Recency','Frequency','Monetary','Segment_Description','Recommendation']].reset_index(drop=True)
)
