import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide"
)

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "2. Cleaned Data"
MODEL_DIR = BASE_DIR / "3. Models"

ONLINE_RETAIL_PATH = DATA_DIR / "online_retail.csv"
CUSTOMER_RFM_PATH = DATA_DIR / "customer_rfm.csv"

KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
SIMILARITY_PATH = MODEL_DIR / "similarity_matrix.pkl"
PRODUCT_NAMES_PATH = MODEL_DIR / "product_names.pkl"

CLUSTER_LABELS = {
    0: "Regular Customers",
    1: "At-Risk Customers",
    2: "VIP Customers",
    3: "High Value Customers"
}

SEGMENT_DESCRIPTIONS = {
    "Regular Customers": "Moderate spending and moderate activity. These customers purchase fairly consistently but are not top spenders.",
    "At-Risk Customers": "Customers who have not purchased recently. Their engagement has dropped and they may be at risk of churning.",
    "High Value Customers": "Frequent buyers with high spending. These customers contribute significantly to revenue.",
    "VIP Customers": "Extremely valuable customers with exceptional spending. A small but critical group that drives a large share of revenue."
}

CLUSTER_SUMMARY = {
    "Regular Customers": "Moderate recency, moderate frequency, moderate monetary value.",
    "At-Risk Customers": "High recency (long time since last purchase), low frequency, low to moderate monetary value.",
    "High Value Customers": "Low recency, high frequency, high monetary value.",
    "VIP Customers": "Very low recency, very high frequency, very high monetary value."
}

# ----------------------------------------------------------------------------
# DATA / MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_online_retail():
    try:
        df = pd.read_csv(ONLINE_RETAIL_PATH)
        return df
    except Exception as e:
        st.error(f"Could not load online_retail.csv: {e}")
        return pd.DataFrame()


@st.cache_data
def load_customer_rfm():
    try:
        df = pd.read_csv(CUSTOMER_RFM_PATH)
        return df
    except Exception as e:
        st.error(f"Could not load customer_rfm.csv: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_kmeans_model():
    try:
        return joblib.load(KMEANS_PATH)
    except Exception as e:
        st.error(f"Could not load kmeans_model.pkl: {e}")
        return None


@st.cache_resource
def load_scaler():
    try:
        return joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Could not load scaler.pkl: {e}")
        return None


@st.cache_resource
def load_similarity_matrix():
    try:
        return joblib.load(SIMILARITY_PATH)
    except Exception as e:
        st.error(f"Could not load similarity_matrix.pkl: {e}")
        return None


@st.cache_resource
def load_product_names():
    try:
        return joblib.load(PRODUCT_NAMES_PATH)
    except Exception as e:
        st.error(f"Could not load product_names.pkl: {e}")
        return []


retail_df = load_online_retail()
rfm_df = load_customer_rfm()
kmeans_model = load_kmeans_model()
scaler = load_scaler()
similarity_matrix = load_similarity_matrix()
product_names = load_product_names()

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("Shopper Spectrum")
st.sidebar.caption("Customer Segmentation and Product Recommendation")

page = st.sidebar.radio(
    "Navigate to",
    [
        "Home",
        "Business Dashboard",
        "Customer Segmentation",
        "Product Recommendation",
        "ML Insights",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, scikit-learn and Plotly")

# ----------------------------------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------------------------------
if page == "Home":
    st.title("Shopper Spectrum")
    st.subheader("Customer Segmentation and Product Recommendation in E-Commerce")

    st.markdown(
        """
        This application analyzes transactional retail data to understand customer
        behavior and product relationships. It combines two machine learning
        approaches: customer segmentation based on purchase history, and a
        product recommendation engine based on purchase similarity. Together,
        these tools support data-driven marketing and merchandising decisions.
        """
    )

    st.markdown("### Business Objectives")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            - Identify distinct customer groups based on recency, frequency and
              monetary value (RFM)
            - Help marketing teams target each segment with relevant campaigns
            - Reduce churn by identifying at-risk customers early
            """
        )
    with col2:
        st.markdown(
            """
            - Recommend related products to increase basket size
            - Surface revenue and sales trends for decision making
            - Provide a single interactive tool for exploring the business data
            """
        )

    st.markdown("### Machine Learning Techniques Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**K-Means Clustering**")
        st.markdown(
            "Customers are grouped into four segments using their Recency, "
            "Frequency and Monetary (RFM) values, scaled and clustered with K-Means."
        )
    with col2:
        st.markdown("**Cosine Similarity Recommendation System**")
        st.markdown(
            "Products are compared using item-to-item collaborative filtering, "
            "where cosine similarity between purchase patterns determines which "
            "products are related."
        )

    st.markdown("### Key Metrics")

    if not retail_df.empty:
        total_customers = retail_df["CustomerID"].nunique()
        total_products = retail_df["Description"].nunique()
        total_revenue = retail_df["TotalAmount"].sum()
        total_countries = retail_df["Country"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total_customers:,}")
        c2.metric("Total Products", f"{total_products:,}")
        c3.metric("Total Revenue", f"£{total_revenue:,.0f}")
        c4.metric("Total Countries", f"{total_countries:,}")
    else:
        st.warning("Transaction data could not be loaded, so key metrics are unavailable.")

# ----------------------------------------------------------------------------
# PAGE 2: BUSINESS DASHBOARD
# ----------------------------------------------------------------------------
elif page == "Business Dashboard":
    st.title("Business Dashboard")
    st.caption("Interactive analytics built from the transaction-level data")

    if retail_df.empty:
        st.warning("Transaction data is not available. Please check that online_retail.csv exists.")
    else:
        df = retail_df.copy()

        total_revenue = df["TotalAmount"].sum()
        total_customers = df["CustomerID"].nunique()
        total_products = df["Description"].nunique()
        total_transactions = df["InvoiceNo"].nunique()
        avg_order_value = df.groupby("InvoiceNo")["TotalAmount"].sum().mean()

        st.markdown("### Key Performance Indicators")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Revenue", f"£{total_revenue:,.0f}")
        k2.metric("Total Customers", f"{total_customers:,}")
        k3.metric("Total Products", f"{total_products:,}")
        k4.metric("Avg Order Value", f"£{avg_order_value:,.2f}")
        k5.metric("Total Transactions", f"{total_transactions:,}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top 10 Selling Products (by Quantity)")
            top_selling = (
                df.groupby("Description")["Quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                top_selling.sort_values("Quantity"),
                x="Quantity", y="Description", orientation="h",
                labels={"Quantity": "Units Sold", "Description": "Product"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Top 10 Revenue Generating Products")
            top_revenue = (
                df.groupby("Description")["TotalAmount"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                top_revenue.sort_values("TotalAmount"),
                x="TotalAmount", y="Description", orientation="h",
                labels={"TotalAmount": "Revenue (£)", "Description": "Product"}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Monthly Sales Trend")
            month_order = ["January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"]
            monthly = (
                df.groupby(["InvoiceYear", "InvoiceMonth"])["TotalAmount"]
                .sum()
                .reset_index()
            )
            monthly["InvoiceMonth"] = pd.Categorical(monthly["InvoiceMonth"], categories=month_order, ordered=True)
            monthly = monthly.sort_values(["InvoiceYear", "InvoiceMonth"])
            monthly["Period"] = monthly["InvoiceMonth"].astype(str) + " " + monthly["InvoiceYear"].astype(str)
            fig = px.line(
                monthly, x="Period", y="TotalAmount", markers=True,
                labels={"TotalAmount": "Revenue (£)", "Period": "Month"}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Country-wise Revenue")
            country_revenue = (
                df.groupby("Country")["TotalAmount"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                country_revenue.sort_values("TotalAmount"),
                x="TotalAmount", y="Country", orientation="h",
                labels={"TotalAmount": "Revenue (£)", "Country": "Country"}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top 10 Customers by Spending")
            top_customers = (
                df.groupby("CustomerID")["TotalAmount"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            top_customers["CustomerID"] = top_customers["CustomerID"].astype(str)
            fig = px.bar(
                top_customers.sort_values("TotalAmount"),
                x="TotalAmount", y="CustomerID", orientation="h",
                labels={"TotalAmount": "Revenue (£)", "CustomerID": "Customer ID"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Customer Segment Distribution")
            if not rfm_df.empty:
                segment_counts = rfm_df["CustomerSegment"].value_counts().reset_index()
                segment_counts.columns = ["CustomerSegment", "Count"]
                fig = px.bar(
                    segment_counts, x="CustomerSegment", y="Count",
                    labels={"CustomerSegment": "Segment", "Count": "Number of Customers"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("customer_rfm.csv not available.")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Revenue by Part of Day")
            part_order = ["Morning", "Afternoon", "Evening", "Night"]
            part_revenue = df.groupby("PartOfDay")["TotalAmount"].sum().reindex(part_order).dropna().reset_index()
            fig = px.bar(
                part_revenue, x="PartOfDay", y="TotalAmount",
                labels={"TotalAmount": "Revenue (£)", "PartOfDay": "Part of Day"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Revenue by Day of Week")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_revenue = df.groupby("InvoiceDay")["TotalAmount"].sum().reindex(day_order).dropna().reset_index()
            fig = px.bar(
                day_revenue, x="InvoiceDay", y="TotalAmount",
                labels={"TotalAmount": "Revenue (£)", "InvoiceDay": "Day"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("#### Revenue by Month")
            month_revenue = df.groupby("InvoiceMonth")["TotalAmount"].sum()
            month_revenue = month_revenue.reindex(month_order).dropna().reset_index()
            fig = px.bar(
                month_revenue, x="InvoiceMonth", y="TotalAmount",
                labels={"TotalAmount": "Revenue (£)", "InvoiceMonth": "Month"}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 3: CUSTOMER SEGMENTATION
# ----------------------------------------------------------------------------
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.caption("Predict which segment a customer belongs to based on RFM values")

    if kmeans_model is None or scaler is None:
        st.warning("Segmentation model files are not available. Please check kmeans_model.pkl and scaler.pkl.")
    else:
        st.markdown("### Enter Customer RFM Values")
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
        with col2:
            frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5, step=1)
        with col3:
            monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, step=10.0)

        if st.button("Predict Segment"):
            try:
                input_data = np.array([[recency, frequency, monetary]])
                scaled_input = scaler.transform(input_data)
                cluster = kmeans_model.predict(scaled_input)[0]
                segment = CLUSTER_LABELS.get(cluster, "Unknown Segment")

                st.success(f"Predicted Segment: {segment}")

                if segment in SEGMENT_DESCRIPTIONS:
                    st.info(SEGMENT_DESCRIPTIONS[segment])
            except Exception as e:
                st.error(f"Could not generate a prediction: {e}")

        st.markdown("---")
        st.markdown("### Segment Descriptions")

        for seg_name, desc in SEGMENT_DESCRIPTIONS.items():
            st.markdown(f"**{seg_name}**")
            st.markdown(desc)
            st.markdown("")

# ----------------------------------------------------------------------------
# PAGE 4: PRODUCT RECOMMENDATION
# ----------------------------------------------------------------------------
elif page == "Product Recommendation":
    st.title("Product Recommendation")
    st.caption("Find products similar to a chosen item using item-to-item collaborative filtering")

    if similarity_matrix is None or not product_names:
        st.warning("Recommendation model files are not available. Please check similarity_matrix.pkl and product_names.pkl.")
    else:
        selected_product = st.selectbox("Select a product", options=sorted(product_names))

        if st.button("Get Recommendations"):
            try:
                if selected_product not in product_names:
                    st.warning("Selected product was not found in the product catalog.")
                else:
                    product_index = product_names.index(selected_product)
                    similarity_scores = similarity_matrix[product_index]

                    similar_indices = np.argsort(similarity_scores)[::-1]
                    similar_indices = [i for i in similar_indices if i != product_index][:5]

                    st.markdown(f"### Products similar to: {selected_product}")

                    for rank, idx in enumerate(similar_indices, start=1):
                        product = product_names[idx]
                        score = similarity_scores[idx]
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 5])
                            with col1:
                                st.markdown(f"**#{rank}**")
                            with col2:
                                st.markdown(f"**{product}**")
                                st.caption(f"Similarity score: {score:.3f}")
            except ValueError:
                st.warning("Selected product was not found in the product catalog.")
            except Exception as e:
                st.error(f"Could not generate recommendations: {e}")

# ----------------------------------------------------------------------------
# PAGE 5: ML INSIGHTS
# ----------------------------------------------------------------------------
elif page == "ML Insights":
    st.title("ML Insights")
    st.caption("A closer look at the models powering this application")

    st.markdown("## K-Means Clustering")

    st.markdown("### What is RFM?")
    st.markdown(
        """
        RFM stands for Recency, Frequency and Monetary value. It is a widely
        used technique in customer analytics for summarizing purchase behavior
        into three simple but powerful numbers.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Recency**")
        st.markdown("Number of days since the customer's most recent purchase. Lower values indicate recent activity.")
    with col2:
        st.markdown("**Frequency**")
        st.markdown("Total number of purchases made by the customer. Higher values indicate loyal customers.")
    with col3:
        st.markdown("**Monetary**")
        st.markdown("Total amount spent by the customer. Higher values indicate more valuable customers.")

    st.markdown("---")
    st.markdown("### Cluster Summary")

    if not rfm_df.empty:
        segment_counts = rfm_df["CustomerSegment"].value_counts()

        for seg_name in ["Regular Customers", "At-Risk Customers", "High Value Customers", "VIP Customers"]:
            count = segment_counts.get(seg_name, 0)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{seg_name}**")
                st.caption(CLUSTER_SUMMARY.get(seg_name, ""))
            with col2:
                st.metric("Customers", f"{count:,}")

        st.markdown("")
        avg_by_segment = rfm_df.groupby("CustomerSegment")[["Recency", "Frequency", "Monetary"]].mean().round(2)
        st.markdown("#### Average RFM Values by Segment")
        st.dataframe(avg_by_segment, use_container_width=True)
    else:
        st.info("customer_rfm.csv not available, so cluster counts cannot be shown.")

    st.markdown("---")
    st.markdown("## Recommendation System")

    st.markdown("### Item-to-Item Collaborative Filtering")
    st.markdown(
        """
        Rather than relying on individual customer profiles, item-to-item
        collaborative filtering looks at how products co-occur across many
        transactions. Two products are considered similar if they tend to be
        purchased by the same customers, regardless of who those customers are.
        """
    )

    st.markdown("### Cosine Similarity")
    st.markdown(
        """
        Cosine similarity measures the angle between two vectors rather than
        their magnitude. Each product is represented as a vector of customer
        purchase patterns, and the cosine similarity between two such vectors
        gives a score between 0 and 1, where values closer to 1 mean the
        products are more closely related.
        """
    )

    st.markdown("### How Recommendations Are Generated")
    st.markdown(
        """
        1. Build a product-by-customer purchase matrix from the transaction data
        2. Compute pairwise cosine similarity between all products
        3. For a selected product, sort the other products by similarity score
        4. Return the top 5 most similar products as recommendations
        """
    )

# ----------------------------------------------------------------------------
# PAGE 6: ABOUT PROJECT
# ----------------------------------------------------------------------------
elif page == "About Project":
    st.title("About Project")

    st.markdown("### Project Title")
    st.markdown("Shopper Spectrum: Customer Segmentation and Product Recommendation in E-Commerce")

    st.markdown("### Business Problem")
    st.markdown(
        """
        E-commerce businesses generate large volumes of transaction data, but
        without analysis this data offers little practical value. Two common
        challenges are understanding which customers matter most to the
        business, and helping customers discover products they are likely to
        want, in order to increase engagement and revenue.
        """
    )

    st.markdown("### Business Objectives")
    st.markdown(
        """
        - Segment customers into meaningful groups based on purchase behavior
        - Support targeted marketing and retention strategies for each segment
        - Recommend related products to increase cross-selling opportunities
        - Provide a clear, interactive view of overall business performance
        """
    )

    st.markdown("### Dataset Description")
    st.markdown(
        "The dataset contains historical e-commerce transactions with the following columns:"
    )

    columns_list = [
        "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "UnitPrice", "CustomerID", "Country", "InvoiceTime", "TotalAmount",
        "InvoiceYear", "InvoiceMonth", "InvoiceDay", "HourOfPurchase", "PartOfDay"
    ]
    col1, col2, col3 = st.columns(3)
    third = len(columns_list) // 3 + 1
    for i, col in enumerate([col1, col2, col3]):
        with col:
            for c in columns_list[i * third:(i + 1) * third]:
                st.markdown(f"- {c}")

    st.markdown("### Machine Learning Techniques Used")
    st.markdown(
        """
        1. K-Means Clustering, used for customer segmentation based on RFM values
        2. Cosine Similarity, used for item-to-item product recommendations
        """
    )

    st.markdown("### Key Achievements")
    st.markdown(
        """
        - Customer Segmentation into four actionable groups
        - Product Recommendation engine based on purchase similarity
        - Business Analytics Dashboard covering sales, revenue and customer trends
        - A single Interactive Web Application bringing all of this together
        """
    )
