import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
from model import model_scaled, scaler, x, train_accuracy_scaled, test_accuracy_scaled, y_test, x_test_scaled

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ Page Config ------------------
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

# ------------------ UI ------------------

# ------------------ Sidebar ------------------
with st.sidebar:
    st.write("# **Loan Approval Predictor**")
    st.image("assets/loan.png", use_container_width=True)

    st.markdown("""
    Welcome to your smart loan assistant!  
    Select a section below to get started:
    """)

    selected_tab = st.radio(
        "📂 **Choose Section**",
        ["🔍 Single Prediction", "📁 Batch Prediction", "📊 Model Insights"],
        help="Navigate between prediction modes and model insights"
    )

    st.markdown("---")
    st.markdown("""
    ### 🧠 **Tips**
    - Use **Single Prediction** for individual loan checks  
    - Use **Batch Prediction** to upload multiple applications  
    - Explore **Model Insights** to understand how predictions are made
    """)

    st.markdown("---")
    st.caption("🛡️ Your data stays private.")

# ------------------ Main Title ------------------
st.title("💳 Loan Approval Prediction System")


# ------------------ Tab 1: Single Prediction ------------------

if selected_tab == "🔍 Single Prediction":
    st.markdown("### Fill in your details below to check your loan eligibility.")

    with st.form("loan_form"):
        user_input = {}
        user_input["education"] = st.radio("🎓 Education", options=[0, 1], format_func=lambda x: "Not Graduate" if x == 0 else "Graduate")
        user_input["self_employed"] = st.radio("💼 Self Employed", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        user_input["income_annum"] = st.number_input("💰 Annual Income", min_value=0.0, value=500000.0)
        user_input["cibil_score"] = st.number_input("📊 CIBIL Score", min_value=300, max_value=900, value=750)
        user_input["bank_asset_value"] = st.number_input("🏦 Bank Asset Value", min_value=0.0, value=1000000.0)
        user_input["commercial_assets_value"] = st.number_input("🏢 Commercial Assets", min_value=0.0, value=500000.0)
        user_input["luxury_assets_value"] = st.number_input("💎 Luxury Assets", min_value=0.0, value=250000.0)
        user_input["residential_assets_value"] = st.number_input("🏠 Residential Assets", min_value=0.0, value=750000.0)
        user_input["loan_amount"] = st.number_input("💸 Loan Amount", min_value=0.0, value=200000.0)
        user_input["loan_term"] = st.number_input("📆 Loan Term", min_value=2, max_value=20, value=10)   # Convert to months
        user_input["no_of_dependents"] = st.radio("👶 No. of Dependents", options=list(range(0, 6)))
        user_input["loan_id"] = 0  # Placeholder

        submitted = st.form_submit_button("🚀 Predict Loan Status")

    if submitted:
        input_df = pd.DataFrame([user_input])[x.columns]
        input_scaled = scaler.transform(input_df)
        prediction = model_scaled.predict(input_scaled)

        if prediction[0] == 1:
            st.success("🎉 Congratulations! Your loan is likely to be **approved** ✅")
            result_label = "✅ Approved"
        else:
            st.error("❌ Unfortunately, your loan is likely to be **rejected**")
            result_label = "❌ Rejected"

        st.session_state.history.append({"Input": list(user_input.values()), "Prediction": result_label})

    if st.session_state.history:
        st.markdown("### 🕘 Recent Predictions")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

# ------------------ Tab 2: Batch Prediction ------------------
elif selected_tab == "📁 Batch Prediction":
    st.markdown("### Upload a CSV file with loan data to predict multiple applications.")
    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            if list(batch_data.columns) != list(x.columns):
                st.error("⚠️ Column mismatch. Please ensure your CSV matches the required format.")
            else:
                st.dataframe(batch_data.head(), use_container_width=True)
                batch_scaled = scaler.transform(batch_data)
                batch_predictions = model_scaled.predict(batch_scaled)
                batch_data["Prediction"] = ["✅ Approved" if p == 1 else "❌ Rejected" for p in batch_predictions]
                st.success("✅ Predictions completed!")
                st.dataframe(batch_data, use_container_width=True)

                csv = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", data=csv, file_name="loan_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"🚫 Error processing file: {e}")

# ------------------ Tab 3: Model Insights ------------------
elif selected_tab == "📊 Model Insights":
    st.markdown("### 📊 Feature Importance (Logistic Coefficients)")
    coef_df = pd.DataFrame({
        "Feature": x.columns,
        "Coefficient": model_scaled.coef_[0]
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    st.bar_chart(coef_df.set_index("Feature"))
    st.caption("Higher absolute values indicate stronger influence on loan approval.")

    st.markdown("---")
    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{train_accuracy_scaled:.2f}")
    col2.metric("Testing Accuracy", f"{test_accuracy_scaled:.2f}")
    st.progress(train_accuracy_scaled, text="Training Accuracy")
    st.progress(test_accuracy_scaled, text="Testing Accuracy")

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, model_scaled.predict(x_test_scaled))
    cm_df = pd.DataFrame(cm, index=["Rejected", "Approved"], columns=["Predicted Rejected", "Predicted Approved"])
    st.dataframe(cm_df)

# ------------------ Footer ------------------
st.markdown("---")
st.caption("🔍 This app is for educational purposes only and should not be used for actual financial decisions.")