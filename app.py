import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from prophet import Prophet
import os
import requests 
import spacy

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------------
# Custom CSS for Enhanced UI Styling
# --------------------------------------------------------
st.markdown("""
    <style>
    /* Background and Text Styling */
    .main {
        background-color: #2D2F33;
        color: #E0E0E0;
        font-family: 'Arial', sans-serif;
    }

    /* Title Section */
    .title-container {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFA500; /* Golden Orange */
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .subtitle-container {
        text-align: center;
        font-size: 1rem;
        color: #c89574; /* Light Brown */
        font-style: italic;
        margin-bottom: 30px;
    }

    /* Sidebar Styling */
    .css-1lcbmhc.e1fqkh3o3 {
        background-color: #424549 !important; /* Sidebar background */
        color: #FFFFFF !important; /* Sidebar text */
    }

    /* Filter and Search Section */
    .stTextInput > div {
        background-color: #424549 !important; /* Input background */
        border-radius: 5px !important;
        color: #FFFFFF !important; /* Input text color */
    }

    /* Data Table Styling */
    .stDataFrame {
        border: 1px solid #5A5C60;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Button Styling */
    .stButton>button {
        background-color: #FFA500; /* Golden Orange */
        color: #000000; /* Black text */
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFC107; /* Lighter Golden */
    }

    /* Spinner Styling */
    .stSpinner {
        color: #FFA500 !important; /* Golden Spinner */
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# 1) LOAD YOUR FINAL DATASET
# --------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Dynamically construct the dataset path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "data", "Final_Dataset_CAPS.csv")
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format

        # Adding default columns if missing
        required_columns = {
            "Case Outcome": "Ongoing",
            "Notes": "",
            "Confidence Score": 0.0,
            "Flagged": False,
            "Case Status": "Ongoing",
            "Assigned Investigator": "Unassigned",
        }

        for col, default in required_columns.items():
            if col not in df.columns:
                df[col] = default

        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# --------------------------------------------------------
# 2) LOAD YOUR BERT MODEL/PIPELINE
# --------------------------------------------------------
@st.cache_resource
def load_model():
    model_folder_path = "bert_abuse_model_v2"
    tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_folder_path)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)



# --------------------------------------------------------
# 3) FILTERING FUNCTION
# --------------------------------------------------------
def exact_filter(df, query, column):
    query = query.strip()
    if query == "":
        return df  # If no query, return the entire DataFrame
    
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        # Convert the query to a numeric type if the column is numeric
        try:
            query = float(query)  # Try to interpret the query as a number
        except ValueError:
            return df.iloc[0:0]  # Return an empty DataFrame if conversion fails
        # Perform exact match filtering for numeric columns
        exact_matches = df[df[column] == query]
    elif pd.api.types.is_string_dtype(df[column]):
        # Perform case-insensitive exact matching for string columns
        exact_matches = df[df[column].str.lower() == query.lower()]
    else:
        # For other data types, perform a direct exact match
        exact_matches = df[df[column] == query]

    return exact_matches



# --------------------------------------------------------
# 4) PREDICT ABUSE TYPE FUNCTION
# --------------------------------------------------------
def predict_abuse_type(text, classifier):
    outputs = classifier(text)[0]  # Get model predictions for the text
    sorted_outputs = sorted(outputs, key=lambda x: x["score"], reverse=True)
    label_mapping = {
        "LABEL_0": "Emotional",
        "LABEL_1": "Neglect",
        "LABEL_2": "Physical",
        "LABEL_3": "Sexual",
        "LABEL_4": "Other",
    }
    top_label = label_mapping.get(sorted_outputs[0]["label"], "Unknown")
    top_confidence = float(sorted_outputs[0]["score"])
    return top_label, top_confidence

def auto_flag_logic(df, classifier):
    # Ensure the "Flagged" column exists
    if "Flagged" not in df.columns:
        df["Flagged"] = False  # Default all cases to not flagged

    for idx, row in df.iterrows():
        desc = str(row.get("Case Description", ""))
        if desc.strip():  # Process only if there is a description
            predicted_label, conf_score = predict_abuse_type(desc, classifier)

            # Flag cases based on specific criteria
            if conf_score > 0.85 or (predicted_label == "Sexual" and conf_score > 0.75):
                df.at[idx, "Flagged"] = True
            else:
                df.at[idx, "Flagged"] = False

    return df

# --------------------------------------------------------
# 6) PREPARE DATA FOR FORECASTING
# --------------------------------------------------------
def prepare_forecasting_data(df):
    daily_data = df.groupby('Date').size().reset_index(name='Cases')
    daily_data.rename(columns={'Date': 'ds', 'Cases': 'y'}, inplace=True)  # Required for Prophet
    return daily_data

# --------------------------------------------------------
# 7) TRAIN PROPHET MODEL AND FORECAST
# --------------------------------------------------------
def forecast_cases(daily_data, months=12):
    model = Prophet()
    model.fit(daily_data)
    future = model.make_future_dataframe(periods=months * 30, freq='D')  # Approx. 30 days/month
    forecast = model.predict(future)
    forecast.rename(columns={'yhat': 'Predicted Cases', 'yhat_upper': 'Upper Range', 'yhat_lower': 'Lower Range'}, inplace=True)
    return forecast

# --------------------------------------------------------
# MAIN STREAMLIT APP
# --------------------------------------------------------
def main():
    df = load_data()
    classifier = load_model()

    if df.empty or classifier is None:
        st.error("Failed to initialize the application. Check dataset or model paths.")
        return

    # Title Section
    st.markdown("<div class='title-container'>AI-Powered Child Abuse Detection and Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-container'>By SAI Pro Systems LLC</div>", unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:", ["Home", "Predict Abuse Type", "Auto-Flagging Logic", "Case Management", "Visualizations", "Forecasting"]
    )

    # Home Page
    if page == "Home":
        st.sidebar.subheader("Search and Filter")
        search_query = st.sidebar.text_input("Enter a search query:")
        search_column = st.sidebar.selectbox("Search in column:", df.columns)

        # Filter and display the data
        if search_query.strip() == "" or search_column not in df.columns:
            filtered_df = df
        else:
            filtered_df = df[df[search_column].astype(str).str.contains(search_query, case=False, na=False)]
        st.subheader(f"Filtered Data ({len(filtered_df)} results)")
        st.dataframe(filtered_df)

    # Predict Abuse Type
    elif page == "Predict Abuse Type":
        st.title("Predict Abuse Type")
        case_description = st.text_area("Enter a case description:")
        if st.button("Predict"):
            if case_description.strip():
                pred_label, conf_score = predict_abuse_type(case_description, classifier)
                st.write(f"Predicted Abuse Type: *{pred_label}*")
                st.write(f"Confidence Score: *{conf_score:.2f}*")
            else:
                st.warning("Please enter a case description.")

    # Auto-Flagging Logic
    elif page == "Auto-Flagging Logic":
        st.title("Auto-Flagging Logic")
        st.write("Automatically flag the cases based on prediction confidence and severity criteria.")

        if st.button("Auto-Flag the cases"):
            st.write("Running auto-flagging logic... Please wait.")
            try:
                df = auto_flag_logic(df, classifier)
                st.success("Auto-flagging logic applied successfully!")
                flagged_cases = df[df["Flagged"] == True]
                st.subheader(f"Flagged Cases ({len(flagged_cases)}):")
                st.dataframe(flagged_cases)

                # Save the updated DataFrame
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Final_Dataset_CAPS.csv")
                df.to_csv(csv_path, index=False)
                st.success("Updated flagged cases have been saved.")
            except Exception as e:
                st.error(f"Error running auto-flagging logic: {e}")

    # Case Management
    elif page == "Case Management":
        st.title("Case Management")
        st.write("View, update, and manage case details.")
        
        case_id_to_edit = st.text_input("Enter Case ID to edit:", "")
        if case_id_to_edit:
            case_data = df[df["Child ID"].astype(str) == case_id_to_edit]
            if not case_data.empty:
                st.write("Case Details:")
                st.dataframe(case_data)

                # Editable fields
                new_status = st.selectbox(
                    "Update Case Outcome:", ["Ongoing", "Resolved", "Closed", "Dismissed"],
                    index=["Ongoing", "Resolved", "Closed", "Dismissed"].index(case_data["Case Outcome"].values[0])
                )
                new_notes = st.text_area("Update Notes:", value=case_data["Notes"].values[0])
                new_investigator = st.text_input("Update Assigned Investigator:", value=case_data["Assigned Investigator"].values[0])
                
                if st.button("Save Changes"):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if new_notes.strip():
                        updated_notes = f"{case_data['Notes'].values[0]}\n{new_notes} (Updated on: {timestamp})"
                    else:
                        updated_notes = case_data["Notes"].values[0]

                    # Update the DataFrame
                    df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Case Outcome"] = new_status
                    df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Notes"] = updated_notes
                    df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Assigned Investigator"] = new_investigator
                    
                    # Save the updated DataFrame
                    try:
                        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Final_Dataset_CAPS.csv")
                        df.to_csv(csv_path, index=False)
                        st.success(f"Changes saved for Case ID {case_id_to_edit}!")
                    except Exception as e:
                        st.error(f"Error saving changes: {e}")
            else:
                st.warning("No case found with the provided Case ID.")

    # Visualizations
#     elif page == "Visualizations":
#         st.title("Data Visualizations")
#         heatmap_data = df.groupby(["Abuse Type", "Severity"]).size().reset_index(name="Count")
#         heatmap_fig = px.density_heatmap(
#             heatmap_data, x="Abuse Type", y="Severity", z="Count", color_continuous_scale="Viridis"
#         )
#         st.plotly_chart(heatmap_fig)
        
        
    # Visualizations
    elif page == "Visualizations":
        st.title("Data Visualizations")
        st.write("Explore insights based on the dataset.")

        # Visualization 1: Cases by Region
        st.subheader("Cases by Region")
        abuse_type_filter = st.selectbox("Filter by Abuse Type:", df["Abuse Type"].unique())
        filtered_data = df[df["Abuse Type"] == abuse_type_filter]
        fig1 = px.bar(
            filtered_data,
            x="Region",
            color="Severity",
            title=f"Cases by Region for {abuse_type_filter}",
            labels={"Region": "Region", "count": "Number of Cases"},
        )
        st.plotly_chart(fig1)

        # Visualization 2: Abuse Type and Severity Heatmap
        st.subheader("Abuse Type and Severity Heatmap")
        heatmap_data = df.groupby(["Abuse Type", "Severity"]).size().reset_index(name="Count")
        heatmap_fig = px.density_heatmap(
            heatmap_data,
            x="Abuse Type",
            y="Severity",
            z="Count",
            color_continuous_scale="Viridis",
            title="Abuse Type and Severity Distribution",
        )
        st.plotly_chart(heatmap_fig)

        # Visualization 3: Severity Distribution
        st.subheader("Severity Distribution")
        severity_counts = df["Severity"].value_counts()
        fig3 = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Severity Distribution",
        )
        st.plotly_chart(fig3)


    # Forecasting
#     elif page == "Forecasting":
#         st.title("Forecasting Future Trends")
#         daily_data = prepare_forecasting_data(df)
#         forecast = forecast_cases(daily_data)
#         st.line_chart(forecast[["ds", "Predicted Cases"]].set_index("ds"))

    # Forecasting
    elif page == "Forecasting":
        st.title("Forecasting Future Trends")
        st.write("Predict future trends in case reporting using historical data.")

        # Prepare the data for forecasting
        daily_data = prepare_forecasting_data(df)

        # User input for forecasting
        months_to_forecast = st.slider(
            "Select number of months to forecast:",
            min_value=1,
            max_value=24,
            value=12,
            step=1
        )

        # Generate the forecast
        st.write("Forecasting in progress...")
        try:
            forecast = forecast_cases(daily_data, months=months_to_forecast)
            st.success("Forecast generated successfully!")

            # Display the forecasted data
            st.subheader("Forecasted Data")
            st.dataframe(forecast[["ds", "Predicted Cases", "Upper Range", "Lower Range"]].tail())

            # Visualization
            st.subheader("Forecast Visualization")
            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=daily_data["ds"],
                y=daily_data["y"],
                mode="lines",
                name="Historical Cases",
                line=dict(color="blue")
            ))

            # Predicted data
            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["Predicted Cases"],
                mode="lines",
                name="Predicted Cases",
                line=dict(color="orange")
            ))

            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["Upper Range"],
                mode="lines",
                name="Upper Range",
                line=dict(dash="dot", color="green")
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["Lower Range"],
                mode="lines",
                name="Lower Range",
                line=dict(dash="dot", color="red")
            ))

            fig.update_layout(
                title="Forecast of Future Cases",
                xaxis_title="Date",
                yaxis_title="Number of Cases",
                legend_title="Legend",
                template="plotly_dark"
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error generating forecast: {e}")


if __name__ == "__main__":
    main()
