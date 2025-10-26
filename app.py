
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📈",
    layout="wide"
)

# --- Load Model and Data ---
@st.cache_data
def load_data_and_model():
    """Loads the churn model and the full dataset for the dashboard."""
    try:
        model = joblib.load('churn_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'churn_model.pkl' is in the Colab session.")
        st.stop()

    try:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        df['ChurnValue'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        df = pd.DataFrame()
    return model, df

model, df = load_data_and_model()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
view = st.sidebar.radio("Choose a view:", ["Prediction", "Dashboard"])

# --- PREDICTION VIEW ---
if view == "Prediction":
    st.title("📈 AI-Powered Customer Churn Predictor")
    st.markdown("The app uses AI to analyze customer behavior and identify individuals who are at a high risk of leaving (churning). This allows the company to move from a reactive approach (acting only after a customer cancels) to a proactive one.")
    st.markdown("Enter customer details in the sidebar below to get a real-time churn prediction.")
    st.write("---")
    st.sidebar.header("Customer Details")

    def get_user_input():
        options = {
            'gender': ['Male', 'Female'], 'SeniorCitizen': [0, 1], 'Partner': ['Yes', 'No'],
            'Dependents': ['Yes', 'No'], 'PhoneService': ['Yes', 'No'],
            'MultipleLines': ['No', 'Yes', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'No internet service'],
            'DeviceProtection': ['Yes', 'No', 'No internet service'],
            'TechSupport': ['Yes', 'No', 'No internet service'],
            'StreamingTV': ['Yes', 'No', 'No internet service'],
            'StreamingMovies': ['Yes', 'No', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        }
        input_dict = {
            'gender': st.sidebar.selectbox('Gender', options['gender']),
            'SeniorCitizen': st.sidebar.selectbox('Senior Citizen', options['SeniorCitizen']),
            'Partner': st.sidebar.selectbox('Partner', options['Partner']),
            'Dependents': st.sidebar.selectbox('Dependents', options['Dependents']),
            'tenure': st.sidebar.slider('Tenure (Months)', 0, 72, 24),
            'PhoneService': st.sidebar.selectbox('Phone Service', options['PhoneService']),
            'MultipleLines': st.sidebar.selectbox('Multiple Lines', options['MultipleLines']),
            'InternetService': st.sidebar.selectbox('Internet Service', options['InternetService']),
            'OnlineSecurity': st.sidebar.selectbox('Online Security', options['OnlineSecurity']),
            'OnlineBackup': st.sidebar.selectbox('Online Backup', options['OnlineBackup']),
            'DeviceProtection': st.sidebar.selectbox('Device Protection', options['DeviceProtection']),
            'TechSupport': st.sidebar.selectbox('Tech Support', options['TechSupport']),
            'StreamingTV': st.sidebar.selectbox('Streaming TV', options['StreamingTV']),
            'StreamingMovies': st.sidebar.selectbox('Streaming Movies', options['StreamingMovies']),
            'Contract': st.sidebar.selectbox('Contract', options['Contract']),
            'PaperlessBilling': st.sidebar.selectbox('Paperless Billing', options['PaperlessBilling']),
            'PaymentMethod': st.sidebar.selectbox('Payment Method', options['PaymentMethod']),
            'MonthlyCharges': st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=75.5, step=0.1),
            'TotalCharges': st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=2000.0, step=1.0)
        }
        return pd.DataFrame([input_dict])

    input_df = get_user_input()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Customer Input Summary")
        st.dataframe(input_df.T.rename(columns={0: 'Values'}))
    with col2:
        st.subheader("Prediction Result")
        if st.button("Predict Churn Risk", type="primary", use_container_width=True):
            prediction_proba = model.predict_proba(input_df)[0]
            churn_probability = prediction_proba[1]
            st.write(f"**Churn Probability:** `{churn_probability:.2%}`")
            st.progress(churn_probability)
            if churn_probability > 0.6:
                st.error("🔴 High Risk: Immediate action is recommended.This is an urgent alert. The customer has a very high chance of leaving soon.", icon="🚨")
            elif churn_probability > 0.3:
                st.warning("🟡 Medium Risk: Consider a proactive check-in.The customer might be thinking about leaving, but it's not certain", icon="⚠️")
            else:
                st.success("🟢 Low Risk: Customer is likely to stay.The customer has a very low chance of leaving the company.", icon="✅")

# --- DASHBOARD VIEW ---
elif view == "Dashboard":
    st.title("📊 Customer Analytics Dashboard")
    st.markdown("Explore key metrics and visualizations of the customer dataset.")
    st.write("---")
    if not df.empty:
        total_customers = len(df)
        total_churned = df['ChurnValue'].sum()
        overall_churn_rate = total_churned / total_customers

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Total Churned Customers", f"{total_churned:,}")
        col3.metric("Overall Churn Rate", f"{overall_churn_rate:.2%}")

        st.write("---")
        col1, col2 = st.columns(2)
        sns.set_style("whitegrid")
        with col1:
            st.subheader("Churn Rate by Contract Type")
            churn_by_contract = df.groupby('Contract')['ChurnValue'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values, ax=ax, palette="viridis")
            ax.set_ylabel("Churn Rate")
            for index, value in enumerate(churn_by_contract):
                ax.text(index, value, f'{value:.1%}', ha='center', va='bottom')
            st.pyplot(fig)
        with col2:
            st.subheader("Churn Rate by Internet Service")
            churn_by_internet = df.groupby('InternetService')['ChurnValue'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=churn_by_internet.index, y=churn_by_internet.values, ax=ax, palette="plasma")
            ax.set_ylabel("Churn Rate")
            for index, value in enumerate(churn_by_internet):
                ax.text(index, value, f'{value:.1%}', ha='center', va='bottom')
            st.pyplot(fig)
    else:
        st.warning("Dashboard data could not be loaded.")
