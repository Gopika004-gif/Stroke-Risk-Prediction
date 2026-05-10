import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

st.title("🧠 Stroke Risk Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# If no file → show message (no white screen)
if uploaded_file is None:
    st.warning("⚠️ Please upload a CSV file to continue")
    st.stop()

# Read dataset
df = pd.read_csv(uploaded_file)

st.subheader("📊 Dataset Preview")
st.write(df.head())

# Select target column
target = st.selectbox("Select Target Column", df.columns)

# Split features and target
X = df.drop(target, axis=1)
y = df[target]

# Convert categorical → numeric
X = pd.get_dummies(X)

# 🔥 HANDLE MISSING VALUES (MAIN FIX)
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Clean target (just in case)
y = y.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {acc:.2f}")

# Graph
st.subheader("📈 Accuracy Chart")
fig, ax = plt.subplots()
ax.bar(["Accuracy"], [acc])
st.pyplot(fig)

# Prediction Section
st.subheader("🔍 Make Prediction")

# Convert X back to DataFrame for column names
X_columns = pd.get_dummies(df.drop(target, axis=1)).columns

user_input = {}
for col in X_columns:
    user_input[col] = st.number_input(f"Enter {col}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    # Apply same preprocessing
    input_df = input_df.reindex(columns=X_columns, fill_value=0)
    input_df = imputer.transform(input_df)

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")