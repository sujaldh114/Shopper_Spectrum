# Shopper Spectrum

### Customer Segmentation and Product Recommendation in E-Commerce

Shopper Spectrum is an end-to-end data analytics and machine learning project that analyzes e-commerce transaction data to understand customer behavior, identify valuable customers, and recommend relevant products.

---

## Project Overview

This project helps businesses answer questions such as:

- Who are the most valuable customers?
- Which customers are at risk of leaving?
- Which products generate the highest revenue?
- Which products should be recommended together?
- What are the overall sales trends?

---
## Machine Learning Techniques Used

### 1. K-Means Clustering
Used for customer segmentation based on:

- **Recency** – Days since last purchase
- **Frequency** – Number of purchases
- **Monetary** – Total amount spent

Customer Segments:

- Regular Customers
- At-Risk Customers
- High Value Customers
- VIP Customers

### 2. Item-to-Item Collaborative Filtering
Used for product recommendations through:

- Customer-Product Matrix
- Binary Purchase Matrix
- Cosine Similarity

The system recommends similar products based on customer purchasing patterns.

---

## Features

### Home
- Project overview
- Business objectives
- Key metrics

### Business Dashboard
- Total Revenue
- Total Customers
- Total Products
- Average Order Value
- Top Selling Products
- Top Revenue Products
- Monthly Sales Trend
- Country-wise Revenue
- Revenue by Month
- Revenue by Day
- Revenue by Part of Day
- Top Customers by Spending

### Customer Segmentation
Predict customer segments using:

- Recency
- Frequency
- Monetary

### Product Recommendation
Generate Top-5 similar product recommendations using cosine similarity.

### ML Insights
- RFM Analysis
- K-Means Clustering
- Collaborative Filtering
- Customer Segment Statistics

---

## Project Structure

```text
Shopper_Spectrum
│
├── app.py
├── requirements.txt
├── README.md
│
├── 2. Cleaned Data
│   ├── online_retail.csv
│   └── customer_rfm.csv
│
├── 3. Models
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── product_names.pkl
│
└── 4. Notebook
    ├── 01_rfm_clustering.ipynb
    └── 02_product_recommendation.ipynb
```

---

## Run the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Joblib
- Jupyter Notebook

---

## Key Achievements

✔ Customer Segmentation using K-Means Clustering

✔ Product Recommendation using Cosine Similarity

✔ Interactive Business Analytics Dashboard

✔ Streamlit Web Application

✔ Cloud Deployment

---

##  Author

**Sujal Dhiman**

MCA Student | Data Analytics Enthusiast

GitHub: https://github.com/sujaldh114