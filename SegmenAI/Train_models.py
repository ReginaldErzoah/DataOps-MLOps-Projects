import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import cloudpickle

# Load your CSV dataset
df = pd.read_csv("customerdata.csv", parse_dates=['InvoiceDate'], encoding='ISO-8859-1')

# RFM feature engineering
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()

rfm.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalAmount':'Monetary'}, inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Save models using cloudpickle
with open("models/scaler_rfm.pkl", "wb") as f:
    cloudpickle.dump(scaler, f)

with open("models/kmeans_rfm.pkl", "wb") as f:
    cloudpickle.dump(kmeans, f)

print("Models trained and saved successfully!")