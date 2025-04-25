import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular

# Page setup
st.set_page_config(page_title="Credit Risk App", layout="wide")

# Load model and features
model = joblib.load("models/credit_risk_model.pkl")
feature_cols = pd.read_csv("models/feature_columns.csv").values.flatten().tolist()

# Theme Toggle
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

theme = st.sidebar.radio("🌗 Choose Theme", ["Light", "Dark"], index=["Light", "Dark"].index(st.session_state["theme"]))
st.session_state["theme"] = theme

if st.session_state["theme"] == "Light":
    st.markdown("""
        <style>
            body, .stApp { background-color: #ffffff; color: #000000; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp { background-color: #0e1117; color: #ffffff; }
        </style>
    """, unsafe_allow_html=True)

st.title("🏦 Credit Risk Prediction App")

# Sidebar inputs
st.sidebar.header("Enter Applicant Details")
input_data = {
    "Age": st.sidebar.slider("Age", 18, 75, 30),
    "Credit amount": st.sidebar.number_input("Credit Amount", 100, 100000, 2000),
    "Duration": st.sidebar.slider("Duration (months)", 4, 72, 12),
    "Sex": st.sidebar.selectbox("Sex", ["male", "female"]),
    "Housing": st.sidebar.selectbox("Housing", ["own", "for free", "rent"]),
    "Saving accounts": st.sidebar.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich"]),
    "Checking account": st.sidebar.selectbox("Checking account", ["little", "moderate", "rich"]),
    "Purpose": st.sidebar.selectbox("Purpose", [
        "radio/TV", "education", "furniture/equipment", "car", "business",
        "repairs", "domestic appliances", "vacation/others"
    ]),
    "Job": st.sidebar.selectbox("Job", ["0", "1", "2", "3"])
}

# Transform input
input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df, drop_first=True)
for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_cols]

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Input Summary")
    st.write(input_data)

with col2:
    if st.sidebar.button("Predict Credit Risk"):
        proba = model.predict_proba(input_encoded)[0]
        pred = model.predict(input_encoded)[0]
        confidence = proba[int(pred)] * 100

        if pred == 1:
            risk_color = "#d4edda" if st.session_state["theme"] == "Light" else "#1e4620"
            text_color = "#155724" if st.session_state["theme"] == "Light" else "#c3e6cb"
            st.markdown(f"""
                <div style="background-color:{risk_color}; padding:10px; border-radius:5px;">
                    <strong style="color:{text_color}; font-size:18px;">🟢 Predicted Risk: GOOD</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("🔴 Predicted Risk: BAD")


        st.progress(int(confidence))
        st.markdown(f"**Confidence:** {int(confidence)}%")

        # Permutation-based feature importance
        st.subheader("📊 Feature Importance (Permutation-Based)")
        try:
            X_sample = input_encoded.copy()
            y_sample = [pred]
            result = permutation_importance(model, X_sample, y_sample, n_repeats=10, random_state=42)
            importances_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": result.importances_mean
            }).replace([np.inf, -np.inf], np.nan).dropna()
            importances_df = importances_df[importances_df["Importance"] > 0].sort_values(by="Importance", ascending=False).head(10)

            if not importances_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=importances_df, x="Importance", y="Feature", ax=ax)
                ax.set_title("Top 10 Important Features")
                st.pyplot(fig)
            else:
                info_bg = "#e0f7fa" if st.session_state["theme"] == "Light" else "#1b2a2f"
                info_text = "#004d40" if st.session_state["theme"] == "Light" else "#b2dfdb"
                st.markdown(f"""
                    <div style="background-color:{info_bg}; color:{info_text}; padding:10px; border-radius:5px;">
                        <strong>No significant feature importances were detected.</strong>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")

        # Save history
        st.session_state.history.append({
            "input": input_data,
            "prediction": "GOOD" if pred == 1 else "BAD",
            "confidence": int(confidence)
        })

        # LIME Explanation
        st.subheader("🧠 Local Interpretation (LIME)")
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array([input_encoded.iloc[0].values]*100),
                feature_names=input_encoded.columns,
                class_names=["BAD", "GOOD"],
                mode="classification"
            )
            explanation = explainer.explain_instance(
                input_encoded.iloc[0].values, 
                model.predict_proba, 
                num_features=10
            )
            html_expl = explanation.as_html()
            background = "#ffffff" if st.session_state["theme"] == "Light" else "#0e1117"
            text_color = "#000000" if st.session_state["theme"] == "Light" else "#ffffff"

            styled_html = f"""
                <div style="background-color: {background}; color: {text_color}; padding: 10px; border-radius: 5px;">
                    <style>
                        table:last-of-type {{ display: none; }}
                    </style>
                    {html_expl}
                </div>
            """


            st.components.v1.html(styled_html, height=600, scrolling=True)
        except Exception as e:
            st.warning(f"LIME explanation failed: {e}")

# Prediction history
st.subheader("🧾 Prediction History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))

# Model Evaluation
with st.expander("📊 Model Evaluation Metrics (Test Set)"):
    try:
        with open("metrics/evaluation_metrics.json") as f:
            metrics = json.load(f)

        cm = metrics.get("confusion_matrix", [])
        if not cm or not isinstance(cm[0], list):
            raise ValueError("Confusion matrix is not valid.")

        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("**Evaluation Metrics**")
        st.write({
            "Accuracy (%)": metrics.get("accuracy", "N/A"),
            "Precision (%)": metrics.get("precision", "N/A"),
            "Recall (%)": metrics.get("recall", "N/A"),
            "F1 Score (%)": metrics.get("f1_score", "N/A"),
            "ROC AUC (%)": metrics.get("roc_auc", "N/A")
        })
    except Exception as e:
        st.warning(f"Could not display evaluation metrics: {e}")

# Recommendations
with st.expander("💡 Recommendations for Improving Credit Evaluation Process"):
    st.markdown("""
    - **Monitor Financial Accounts**: Applicants with moderate to rich checking/saving accounts tend to default less.
    - **Shorten Loan Durations**: Shorter loans reduce risk.
    - **Assess Credit Amount Carefully**: High credit without high income is risky.
    - **Promote Financial Awareness**: Educate applicants on savings habits.
    - **Retrain Models Regularly**: Behavior trends change.
    - **Batch Predictions Help**: Use batch prediction for efficient processing.
    - **Combine Manual + Automated Checks**: Use this model as a filter, then manually review flagged cases.
    """)