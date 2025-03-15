import streamlit as st
import pandas as pd
import pickle
import datetime

# Load trained model and data processing setup
model_path = "npi_model.pkl"  # Save your trained model separately
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load dataset structure for preprocessing
data_path = "dummy_npi_data.xlsx"
xls = pd.ExcelFile(data_path)
df = pd.read_excel(xls)

# Streamlit UI
st.title("NPI Survey Campaign Predictor")
st.write("Enter a time, and we'll predict which doctors are most likely to attend.")

# User input for time
user_time = st.time_input("Select Time:", datetime.time(12, 0))
selected_hour = user_time.hour

# Preprocess data for prediction
df['Login Hour'] = df['Login Time'].dt.hour
df_filtered = df[df['Login Hour'] == selected_hour]

if st.button("Predict"):
    if df_filtered.empty:
        st.warning("No matching records found for this time.")
    else:
        # Extract features for prediction
        X_input = df_filtered.drop(columns=['NPI', 'Login Time', 'Logout Time', 'Count of Survey Attempts'])

        # Apply the same preprocessing as training (one-hot encoding for categorical features)
        X_input = pd.get_dummies(X_input, columns=['State', 'Region', 'Speciality'], drop_first=True)

        # Ensure all columns match the model's expected input
        missing_cols = set(model.feature_names_in_) - set(X_input.columns)
        for col in missing_cols:
            X_input[col] = 0

        # Reorder columns to match training data
        X_input = X_input[model.feature_names_in_]

        # Now make predictions
        predictions = model.predict(X_input)
        df_filtered["Prediction"] = predictions
        best_doctors = df_filtered[df_filtered["Prediction"] == 1]["NPI"]

        # Show results
        st.write("Doctors most likely to attend the survey:")
        st.dataframe(best_doctors)

        # Allow CSV download
        csv = best_doctors.to_csv(index=False).encode()
        st.download_button("Download CSV", data=csv, file_name="target_doctors.csv", mime="text/csv")
