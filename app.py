import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================================
# Page Config
# =========================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# =========================================
# Train & Cache Model from bundled dataset
# =========================================
@st.cache_resource
def load_model():
    df = pd.read_csv("crm_churn_dataset.csv")
    df['Calculated_CLV'] = df['Avg_Purchase_Amount'] * df['Total_Purchases']
    df['New_Activity_Freq'] = df['Total_Purchases'] / (df['Tenure_Months'] + 1)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_model()

# =========================================
# Dashboard Title
# =========================================
st.title("📉 Customer Churn Prediction & CRM Analytics Dashboard")

st.write('''
This AI/ML dashboard predicts customer churn using Machine Learning models.

The system helps businesses:
- Identify customers likely to leave
- Reduce revenue loss
- Improve customer retention
- Analyze customer behavior
- Support data-driven business decisions
''')

# =========================================
# Sidebar Menu
# =========================================
st.sidebar.title("Dashboard Menu")

menu = st.sidebar.radio(
    "Select Option",
    [
        "Upload Dataset",
        "Dataset Information",
        "Visualization Dashboard",
        "Prediction Results",
        "Customer Segmentation",
        "Revenue Analytics",
        "Customer Behavior Analysis",
        "Business Insights"
    ]
)

# =========================================
# File Upload
# =========================================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Feature Engineering
    data['Calculated_CLV'] = data['Avg_Purchase_Amount'] * data['Total_Purchases']
    data['New_Activity_Freq'] = data['Total_Purchases'] / (data['Tenure_Months'] + 1)

    # Keep a copy with Churn if present, remove for predictions
    data_with_churn = data.copy()
    if 'Churn' in data.columns:
        data_no_churn = data.drop('Churn', axis=1)
    else:
        data_no_churn = data.copy()

    # =========================================
    # Upload Dataset
    # =========================================
    if menu == "Upload Dataset":
        st.header("Uploaded Dataset")
        st.write(data.head())

    # =========================================
    # Dataset Information
    # =========================================
    elif menu == "Dataset Information":
        st.header("Dataset Information")
        st.subheader("Dataset Shape")
        st.write(data.shape)
        st.subheader("Column Names")
        st.write(data.columns.tolist())
        st.subheader("First 5 Rows")
        st.write(data.head())
        st.subheader("Statistical Summary")
        st.write(data.describe())
        st.subheader("Missing Values")
        st.write(data.isnull().sum())

    # =========================================
    # Visualization Dashboard
    # =========================================
    elif menu == "Visualization Dashboard":
        st.header("Customer Data Visualization Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Age Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            sns.histplot(data['Age'], bins=20, kde=True, ax=ax1, color='#3b82f6')
            st.pyplot(fig1)
            plt.close()

        with col2:
            st.subheader("Total Spent Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            sns.histplot(data['Total_Spent'], bins=20, kde=True, ax=ax2, color='#8b5cf6')
            st.pyplot(fig2)
            plt.close()

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Feedback Score Distribution")
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            sns.histplot(data['Feedback_Score'], bins=10, kde=True, ax=ax3, color='#10b981')
            st.pyplot(fig3)
            plt.close()

        with col4:
            st.subheader("Purchase Frequency")
            fig4, ax4 = plt.subplots(figsize=(6, 3.5))
            sns.histplot(data['Total_Purchases'], bins=20, kde=True, ax=ax4, color='#f59e0b')
            st.pyplot(fig4)
            plt.close()

        st.subheader("Correlation Heatmap")
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        corr = data.select_dtypes(include='number').corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax5)
        st.pyplot(fig5)
        plt.close()

    # =========================================
    # Prediction Results
    # =========================================
    elif menu == "Prediction Results":
        st.header("Customer Churn Prediction Results")

        try:
            predictions = model.predict(data_no_churn)
            probabilities = model.predict_proba(data_no_churn)
            risk_scores = probabilities[:, 1]

            result_df = data.copy()
            result_df['Prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
            result_df['Risk Score'] = risk_scores
            result_df['Risk Category'] = [
                'High Risk' if s > 0.7 else ('Medium Risk' if s > 0.4 else 'Low Risk')
                for s in risk_scores
            ]

            # Summary metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers", len(result_df))
            c2.metric("Predicted Churn", int(sum(predictions)))
            c3.metric("Churn Rate", f"{sum(predictions)/len(predictions)*100:.1f}%")

            st.subheader("Prediction Results Table")
            st.dataframe(result_df, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Churn Prediction Distribution")
                fig6, ax6 = plt.subplots(figsize=(5, 4))
                result_df['Prediction'].value_counts().plot.pie(
                    autopct='%1.1f%%', ax=ax6,
                    colors=['#3b82f6', '#ef4444']
                )
                ax6.set_ylabel('')
                st.pyplot(fig6)
                plt.close()

            with col2:
                st.subheader("Retention Risk Segmentation")
                fig7, ax7 = plt.subplots(figsize=(5, 4))
                sns.countplot(x='Risk Category', data=result_df, ax=ax7,
                              palette={'Low Risk': '#10b981', 'Medium Risk': '#f59e0b', 'High Risk': '#ef4444'})
                st.pyplot(fig7)
                plt.close()

            st.subheader("Risk Category Explanation")
            st.info('''
            - 🟢 **Low Risk** → Customer likely to stay  
            - 🟡 **Medium Risk** → Customer may churn  
            - 🔴 **High Risk** → Immediate retention action required
            ''')

            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name='prediction_results.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Prediction error: {e}. Make sure your dataset has the same columns as the training data.")

    # =========================================
    # Customer Segmentation
    # =========================================
    elif menu == "Customer Segmentation":
        st.header("Customer Segmentation Dashboard")

        st.subheader("High Value Customers")
        high_value = data[data['Total_Spent'] > data['Total_Spent'].mean()]
        st.write(high_value.head(10))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Age Groups")
            age_groups = pd.cut(data['Age'], bins=[18, 30, 45, 60, 100],
                                labels=['18-30', '31-45', '46-60', '60+'])
            st.bar_chart(age_groups.value_counts().sort_index())

        with col2:
            st.subheader("Customer Lifetime Value")
            fig8, ax8 = plt.subplots(figsize=(6, 3.5))
            sns.boxplot(x=data['Calculated_CLV'], ax=ax8, color='#8b5cf6')
            st.pyplot(fig8)
            plt.close()

    # =========================================
    # Revenue Analytics
    # =========================================
    elif menu == "Revenue Analytics":
        st.header("Revenue Analytics Dashboard")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Revenue", f"₹ {data['Total_Spent'].sum():,.2f}")
        c2.metric("Avg Customer Spending", f"₹ {data['Total_Spent'].mean():,.2f}")
        c3.metric("Avg Customer Lifetime Value", f"₹ {data['Calculated_CLV'].mean():,.2f}")

        clv_q75 = data['Calculated_CLV'].quantile(0.75)
        clv_q40 = data['Calculated_CLV'].quantile(0.40)
        data['CLV_Category'] = data['Calculated_CLV'].apply(
            lambda v: 'High Value' if v > clv_q75 else ('Medium Value' if v > clv_q40 else 'Low Value')
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("CLV Categories")
            fig9, ax9 = plt.subplots(figsize=(6, 3.5))
            sns.countplot(x='CLV_Category', data=data, ax=ax9,
                          palette={'High Value': '#10b981', 'Medium Value': '#f59e0b', 'Low Value': '#ef4444'})
            st.pyplot(fig9)
            plt.close()

        with col2:
            st.subheader("Tenure vs Spending")
            fig10, ax10 = plt.subplots(figsize=(6, 3.5))
            sns.scatterplot(x=data['Tenure_Months'], y=data['Total_Spent'], ax=ax10, alpha=0.5, color='#3b82f6')
            st.pyplot(fig10)
            plt.close()

        st.subheader("Top 10 Spending Customers")
        st.write(data.sort_values(by='Total_Spent', ascending=False).head(10))

    # =========================================
    # Customer Behavior Analysis
    # =========================================
    elif menu == "Customer Behavior Analysis":
        st.header("Customer Behavior Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Activity Frequency")
            fig11, ax11 = plt.subplots(figsize=(6, 3.5))
            sns.histplot(data['New_Activity_Freq'], bins=20, kde=True, ax=ax11, color='#10b981')
            st.pyplot(fig11)
            plt.close()

        with col2:
            st.subheader("Support Tickets")
            fig12, ax12 = plt.subplots(figsize=(6, 3.5))
            sns.countplot(x=data['Support_Tickets'], ax=ax12, color='#f59e0b')
            st.pyplot(fig12)
            plt.close()

        st.subheader("Feedback Score Analysis")
        fig13, ax13 = plt.subplots(figsize=(10, 4))
        sns.histplot(data['Feedback_Score'], bins=10, kde=True, ax=ax13, color='#8b5cf6')
        st.pyplot(fig13)
        plt.close()

    # =========================================
    # Business Insights
    # =========================================
    elif menu == "Business Insights":
        st.header("Business Insights & Recommendations")
        st.markdown('''
### 🔍 Key Business Insights
- Customers with **low tenure** are more likely to churn
- Customers with **many support tickets** show higher churn probability
- **High-value customers** require retention strategies
- Customers with **low activity frequency** need engagement campaigns
- **Feedback score** directly impacts customer satisfaction

### ✅ Recommended Business Actions
- Offer **loyalty rewards** for long-term customers
- Create **retention campaigns** for high-risk customers
- Improve **customer support quality**
- Provide **personalized offers** for high-value customers
- Monitor **customer complaints** regularly
- Use **targeted marketing** based on customer segmentation

### 💼 Business Benefits
- Improves customer retention
- Reduces customer churn
- Increases revenue growth
- Supports data-driven marketing
- Enhances customer relationship management
        ''')

else:
    st.warning("⬆️ Please upload your CSV dataset to continue.")
    st.info("Expected file: **crm_churn_dataset.csv** or any file with the same column structure.")
