import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
from model import model_scaled, scaler, loan_data, x, train_accuracy_scaled, test_accuracy_scaled, conf_matrix, y_test, x_test_scaled

# ------------------ Page Config ------------------
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

# ------------------ UI ------------------
st.title("💳 Loan Approval Prediction System")
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction", "📊 Model Insights", "📁 Sample CSV"])

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ Tab 1: Single Prediction ------------------
with tab1:
    st.markdown("Fill in your details below to check your loan eligibility.")
    with st.expander("ℹ️ Input Format Guide"):
        st.markdown("""
        Enter **11 numeric values** separated by commas in the following order:
        - 🎓 Education (1=Graduate, 0=Not Graduate)  
        - 💼 Self Employed (1=Yes, 0=No)  
        - 💰 Applicant Income  
        - 👥 Coapplicant Income  
        - 🏦 Loan Amount  
        - 📆 Loan Term  
        - 📊 Credit History (1=Good, 0=Bad)  
        - 🚻 Gender (1=Male, 0=Female)  
        - 💍 Married (1=Yes, 0=No)  
        - 👶 Dependents (0, 1, 2, 3)  
        - 🌆 Property Area (0=Rural, 1=Semiurban, 2=Urban)  
        """)
        st.caption("📌 Example: `1,0,2500,0.0,150.0,360.0,1,1,1,0,2`")

    input_loan_data = st.text_input("📥 Paste your values here:", placeholder="e.g. 1,0,2500,0.0,150.0,360.0,1,1,1,0,2")
    

    if st.button("🚀 Predict Loan Status"):
        try:
            input_list = [float(i.strip()) for i in input_loan_data.split(',')]
            if len(input_list) != len(x.columns):
                st.error(f"⚠️ Expected {len(x.columns)} values, but got {len(input_list)}.")
            else:
                input_df = pd.DataFrame([input_list], columns=x.columns)
                input_scaled = scaler.transform(input_df)
                prediction = model_scaled.predict(input_scaled)

                if prediction[0] == 1:
                    st.success("🎉 Congratulations! Your loan is likely to be **approved** ✅")
                    result_label = "✅ Approved"
                else:
                    st.error("❌ Unfortunately, your loan is likely to be **rejected**")
                    result_label = "❌ Rejected"

                st.session_state.history.append({"Input": input_list, "Prediction": result_label})
        except ValueError:
            st.error("🚫 Invalid input format. Ensure all values are numeric and comma-separated.")

    if st.session_state.history:
        st.markdown("### 🕘 Recent Predictions")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

# ------------------ Tab 2: Batch Prediction ------------------
with tab2:
    st.markdown("Upload a CSV file with loan data to predict multiple applications.")
    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            if list(batch_data.columns) != list(x.columns):
                st.error("⚠️ Column mismatch. Please ensure your CSV matches the required format.")
            else:
                st.dataframe(batch_data.head(), use_container_width=True)
                batch_df = pd.DataFrame(batch_data, columns=x.columns)
                batch_scaled = scaler.transform(batch_df)
                batch_predictions = model_scaled.predict(batch_scaled)
                batch_data["Prediction"] = ["✅ Approved" if p == 1 else "❌ Rejected" for p in batch_predictions]
                st.success("✅ Predictions completed!")
                st.dataframe(batch_data, use_container_width=True)

                csv = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", data=csv, file_name="loan_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"🚫 Error processing file: {e}")

# ------------------ Tab 3: Model Insights ------------------
with tab3:
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

# ------------------ Tab 4: Sample CSV Viewer ------------------
with tab4:
    st.markdown("### 📊 Sample CSV Format Preview")
    st.markdown("Here’s a random sample of 15 rows from the loan dataset to help you format your own CSV correctly.")
    st.dataframe(loan_data.sample(15), use_container_width=True)

    st.caption("📌 Use this format when preparing your batch CSV for prediction.")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("🔍 This app is for educational purposes only and should not be used for actual financial decisions.")