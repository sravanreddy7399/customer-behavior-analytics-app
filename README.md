# Customer Behavior & Sales Analytics Platform

## Overview
This project is an end-to-end analytics web application built using Python and Streamlit.

It performs:
- KPI computation (Revenue, Orders, ATV)
- Revenue trend analysis
- Category-level revenue breakdown
- RFM feature engineering
- Customer segmentation using KMeans clustering
- Silhouette score evaluation

## Tech Stack
- Python
- Pandas
- Scikit-Learn
- Streamlit
- KMeans Clustering
- RFM Analysis

## How to Run Locally

1. Install dependencies:
pip install -r requirements.txt

2. Run:
streamlit run app.py

## Deployment
Deployed using Streamlit Community Cloud.

Upload your transactions CSV file with the following required columns:
- order_id
- customer_id
- order_date
- category
- net_sales

link-https://customer-behavior-analytics-app-lsmvnqk6vvztag46bbcmxn.streamlit.app/ 
