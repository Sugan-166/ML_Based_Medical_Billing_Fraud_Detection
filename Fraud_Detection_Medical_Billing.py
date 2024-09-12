import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Title for the Streamlit App
st.title("AI-Driven Fraud Detection System")

# File upload functionality using Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Print the column names to check for exact matches
    st.write("Columns in the dataset: ", data.columns)

    # Keep a copy of relevant columns before preprocessing for later display
    details_columns = ['claim_control_number', 'Part', 'HCPCS Procedure Code', 
                       'Provider Type', 'Type of Bill', 'Review Decision', 'Error Code']

    # Create a copy of these columns for later merging
    original_data = data[details_columns].copy()

    # Feature: Count of procedures in a claim (split by comma and count the elements)
    data['procedure_count'] = data['HCPCS Procedure Code'].apply(lambda x: len(str(x).split(',')))

    # Feature: Number of errors per provider (using groupby to count errors for each provider type)
    data['error_count_provider'] = data.groupby('Provider Type')['Error Code'].transform('count')

    # Identify categorical columns and encode them
    categorical_cols = ['Part', 'Provider Type', 'Type of Bill']

    # Apply One-Hot Encoding on the categorical variables
    encoder = OneHotEncoder()
    encoded_part = encoder.fit_transform(data[categorical_cols])

    # Convert the sparse matrix from OneHotEncoder to a DataFrame for easier inspection
    encoded_df = pd.DataFrame(encoded_part.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate the One-Hot encoded columns back to the original dataframe
    data = pd.concat([data, encoded_df], axis=1)

    # Label Encoding for 'Review Decision' and 'Error Code'
    label_encoder = LabelEncoder()
    data['review_decision_encoded'] = label_encoder.fit_transform(data['Review Decision'])
    data['error_code_encoded'] = label_encoder.fit_transform(data['Error Code'])

    # Drop the original categorical columns after encoding
    data = data.drop(['Part', 'Provider Type', 'Type of Bill', 'Review Decision', 'Error Code', 
                      'claim_control_number', 'HCPCS Procedure Code'], axis=1, errors='ignore')

    # Debugging: Check if 'review_decision_encoded' is in the dataframe
    st.write("Remaining columns in the dataset after preprocessing: ", data.columns)

    # Define target variable (e.g., Review Decision) and features
    if 'review_decision_encoded' not in data.columns:
        st.error("Column 'review_decision_encoded' not found in the dataframe!")
    else:
        X = data.drop(['review_decision_encoded'], axis=1)  # Features
        y = data['review_decision_encoded']  # Target

        # Ensure all columns in X are numeric
        X = X.apply(pd.to_numeric, errors='coerce')

        # Handle missing values (if any) by filling them with 0 or mean
        X.fillna(0, inplace=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Optionally print shapes of training and test data
        st.write(f'Training Features Shape: {X_train.shape}, Testing Features Shape: {X_test.shape}')
        st.write(f'Training Target Shape: {y_train.shape}, Testing Target Shape: {y_test.shape}')

        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Optionally, check model accuracy on the test set
        accuracy = model.score(X_test, y_test)
        st.write(f'Random Forest Model Accuracy: {accuracy:.4f}')

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluation metrics
        st.write(classification_report(y_test, y_pred))
        st.write("ROC AUC Score:", roc_auc_score(y_test, y_pred))

        # ---- Display Detected Fraud Payments with Specific Details ----
        # Assuming 1 means fraud, filter out fraudulent claims
        X_test['fraud_prediction'] = y_pred

        # Reset the index of X_test to match with the original data
        X_test.reset_index(drop=True, inplace=True)

        # Filter fraud cases (where y_pred == 1)
        fraud_cases = X_test[X_test['fraud_prediction'] == 1]

        # Merge fraud cases with the original details (which includes 'claim_control_number', 'Part', etc.)
        fraud_cases_full = fraud_cases.merge(original_data, left_index=True, right_index=True)

        # Debugging: Check columns after merging
        st.write("Columns in fraud_cases_full after merging:", fraud_cases_full.columns)

        # Display the detected fraudulent claims with all specified details
        if not fraud_cases_full.empty:
            st.write("Detected Fraudulent Claims with Details:")
            st.write(fraud_cases_full[['claim_control_number', 'Part', 'HCPCS Procedure Code', 
                                        'Provider Type', 'Type of Bill', 'Review Decision', 'Error Code']])
        else:
            st.write("No fraudulent claims detected.")

        # Optional: Save fraud cases to a CSV file
        fraud_cases_full[['claim_control_number', 'Part', 'HCPCS Procedure Code', 
                         'Provider Type', 'Type of Bill', 'Review Decision', 'Error Code']].to_csv('detected_fraud_cases.csv', index=False)
        st.write("Detected fraud cases saved to 'detected_fraud_cases.csv'.")
else:
    st.write("Please upload a CSV file to proceed.")
