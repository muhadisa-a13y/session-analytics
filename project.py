
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 🛠️ Page Configuration
st.set_page_config(page_title="Session Analytics with Random Forest", layout="wide")

# 📊 Load Data
df = pd.read_csv("data.csv")
st.title("📚 Session Analytics Dashboard")

# 🔢 Encode Categorical Columns
df_encoded = df.copy()
label_encoders = {}
for col in df_encoded.select_dtypes(include="object").columns:
    if col not in ["SessionID", "SessionDate"]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

# 📈 Feature Engineering
target = "UsedAgain"
drop_cols = ["SessionID", "SessionDate"]
X = df_encoded.drop([target] + drop_cols, axis=1)
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

# 🧭 Tabs Section
tab1, tab2, tab3, tab4 = st.tabs(["📑 Dataset", "📊 Model Insights", "📈 Feature Importance", "📦 Advanced Analytics"])

# 🔍 TAB 1: Dataset Overview
with tab1:
    st.subheader("🔎 Preview of Raw Dataset")
    st.dataframe(df)
    st.subheader("🎯 Target Distribution")
    st.bar_chart(df[target].value_counts())

# 🧠 TAB 2: Model Insights
with tab2:
    st.subheader("✅ Accuracy")
    st.metric(label="Accuracy Score", value=f"{accuracy_score(y_test, y_pred):.3f}")

    st.subheader("📋 Classification Report")
    st.dataframe(report_df.style.highlight_max(axis=0))

    st.subheader("🔁 Confusion Matrix (as Table)")
    matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
    st.dataframe(matrix)

    st.subheader("🎯 Prediction Outcome")
    outcome_df = pd.DataFrame({"True Label": y_test, "Predicted": y_pred[:len(y_test)]}).reset_index(drop=True)
    st.dataframe(outcome_df)

# 🌟 TAB 3: Feature Importance
with tab3:
    st.subheader("🌟 Feature Importance Scores")
    st.bar_chart(feature_importance)

# 📈 TAB 4: Advanced Analytics
with tab4:
    st.subheader("🎓 Student Level vs Final Outcome")
    heat_data = pd.crosstab(df["StudentLevel"], df["FinalOutcome"])
    st.dataframe(heat_data)

    st.subheader("📊 Satisfaction Rating Trends")
    st.line_chart(df.groupby("StudentLevel")["SatisfactionRating"].mean())

    st.subheader("⏱️ Average Session Length by Discipline")
    session_avg = df.groupby("Discipline")["SessionLengthMin"].mean()
    st.bar_chart(session_avg)
# 🧮 Prediction Form Section
st.subheader("📥 Predict Session Outcome")

with st.form("prediction_form"):
    st.markdown("Fill in the session details to predict whether it will be used again:")
    
    student_level = st.selectbox("Student Level", df["StudentLevel"].unique())
    discipline = st.selectbox("Discipline", df["Discipline"].unique())
    session_length = st.slider("Session Length (minutes)", min_value=0.0, max_value=60.0, value=15.0)
    total_prompts = st.slider("Total Prompts", min_value=0, max_value=20, value=5)
    task_type = st.selectbox("Task Type", df["TaskType"].unique())
    ai_level = st.slider("AI Assistance Level", min_value=1, max_value=5, value=3)
    final_outcome = st.selectbox("Final Outcome", df["FinalOutcome"].unique())
    satisfaction = st.slider("Satisfaction Rating", min_value=1.0, max_value=5.0, step=0.1, value=3.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Encode input using stored label encoders
        input_dict = {
            "StudentLevel": label_encoders["StudentLevel"].transform([student_level])[0],
            "Discipline": label_encoders["Discipline"].transform([discipline])[0],
            "SessionLengthMin": session_length,
            "TotalPrompts": total_prompts,
            "TaskType": label_encoders["TaskType"].transform([task_type])[0],
            "AI_AssistanceLevel": ai_level,
            "FinalOutcome": label_encoders["FinalOutcome"].transform([final_outcome])[0],
            "SatisfactionRating": satisfaction
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

        st.success(f"🔮 Predicted 'UsedAgain': {'True' if prediction else 'False'}")
