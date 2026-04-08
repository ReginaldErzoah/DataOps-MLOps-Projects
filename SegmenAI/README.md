# SegmenAI – Customer Segmentation System

## Project Overview

SegmenAI is an interactive **Streamlit dashboard** for **customer segmentation and insights**.
It helps businesses analyze transactional data, segment customers using **RFM (Recency, Frequency, Monetary) analysis** and **KMeans clustering**, and generate actionable recommendations for marketing, retention, and revenue optimization.

The app provides:

* **Segment visualizations** with interactive Plotly charts
* **Customer tables** with personalized recommendations per segment
* **Pre-trained KMeans model** and feature scaler for fast predictions

The project demonstrates:

* Python for **data preprocessing and feature engineering**
* Pandas & NumPy for **data wrangling and numerical computation**
* Scikit-learn for **KMeans clustering and feature scaling**
* Streamlit & Plotly for **interactive dashboarding**
* MinIO for **local storage testing**
* Cloudflare R2 for **cloud-hosted dataset storage**
* Docker & Docker Compose for **local development and deployment**

---

## Live Demo

Check out the live dashboard here: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://segmenai.streamlit.app/)

---

## Use Case

This app is ideal for:

* **Marketing teams** seeking to target specific customer segments
* **Data analysts** exploring customer behavior patterns
* **Business owners** aiming to optimize revenue and retention strategies

---

## Features

**RFM-Based Customer Segmentation**

* Segment customers into **High-Value Loyal, Inactive Low-Value, Recent Moderate, and Occasional Buyer**
* Assign descriptive labels to each cluster for easy interpretation
* Segment Interpretation & Business Logic below for detailed explanation.

**Interactive Visualizations**

* Customer count per segment
* Monetary contribution per segment
* Average recency and frequency per segment

**Customer Table with Recommendations**
**Pre-trained Models**
* Efficient clustering using pre-trained **KMeans model** and **feature scaler**

---

## Segment Interpretation & Business Logic

| Segment                  | RFM Characteristics                                      | Business Logic |
|--------------------------|----------------------------------------------------------|----------------|
| High-Value Loyal         | Low Recency, High Frequency, High Monetary              | Most valuable and engaged customers. Focus on retention, loyalty programs, and upselling. |
| Inactive Low-Value       | High Recency, Low Frequency, Low Monetary               | Disengaged customers with low contribution. Consider re-engagement campaigns or deprioritize. |
| Recent Moderate          | Low Recency, Moderate Frequency, Moderate Monetary      | Recently active with growth potential. Nurture to convert into loyal high-value customers. |
| Occasional Buyer         | Moderate Recency, Low Frequency, Low Monetary           | Infrequent buyers. Use promotions and personalized offers to increase engagement and frequency. |



## Dataset
The original dataset was an E-Commerce dataset. [Link](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
The dataset consists of customer transactional data, including:

* `CustomerID`
* `InvoiceNo`
* `InvoiceDate`
* `Quantity`
* `UnitPrice`

Data was initially tested locally using MinIO, an S3-compatible object storage, and then deployed to Cloudflare R2 for scalable cloud storage. The Streamlit app accesses the data directly from the cloud bucket.

---

## Tech Stack

| Category              | Tools                                    |
| --------------------- | ---------------------------------------- |
| Programming           | Python                                   |
| Data Handling         | Pandas, NumPy                            |
| Machine Learning      | scikit-learn (KMeans, StandardScaler)    |
| Visualization         | Plotly                                   |
| Web App               | Streamlit                                |
| Deployment            | Streamlit Cloud, Docker & Docker Compose |
| Model Persistence     | cloudpickle                              |
| Notebook Analysis     | Jupyter Notebook                         |
| Cloud Storage | MinIO (S3-compatible) for local testing, Cloudflare R2 for Production data hosting  |

---

## Architecture & Workflow

```
Customer CSV Data → Cloudflare R2 → Streamlit App
           ↑
           └── KMeans Model + Scaler (Pre-trained)
```

1. **Data Upload** – Customer transactions CSV is uploaded to Cloudflare R2
2. **RFM Feature Engineering** – Calculate Recency, Frequency, and Monetary metrics
3. **Clustering** – Segment customers using KMeans
4. **Personalized Recommendations** – Assign actionable recommendations to each segment
5. **Visualization** – Interactive dashboards show segment metrics and insights

---

## Update & Version Log

* **Version 1.0** (March 2026): Initial release with RFM segmentation, KMeans clustering, interactive visualizations, pre-trained models, personalized recommendations, and support for cloud/local S3 storage

