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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# spacy.cli.download("en_core_web_sm")
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
        csv_path = os.path.join(current_dir,"data", "Final_Dataset.csv")
        df = pd.read_csv(csv_path)
        # st.write(f"Loading dataset from: {csv_path}")
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format

        # Adding default columns if missing
        required_columns = {
            "Case Outcome": "Ongoing",
            "Notes": "",

            "Flagged": False,

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
# -------------------------------------------------------

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


def find_similar_cases(description, df):
    if "Case Description" not in df.columns:
        st.error("The dataset does not contain a 'Case Description' column.")
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer()
    case_descriptions = df["Case Description"].fillna("").tolist()
    tfidf_matrix = tfidf_vectorizer.fit_transform(case_descriptions)

    query_vector = tfidf_vectorizer.transform([description])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    df["Similarity"] = cosine_similarities
    similar_cases = df.sort_values(by="Similarity", ascending=False).head(5)
    return similar_cases


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
    page = st.sidebar.radio("Go to:", ["Home", "Predict Abuse Type", "Auto-Flagging Logic", "Similar Case Discovery","Case Management", "Visualizations", "Forecasting"]
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
            filtered_df = exact_filter(df, search_query, search_column)
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
                current_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(current_dir, "data", "Final_Dataset.csv")
                df.to_csv(csv_path, index=False)
                st.success("Updated flagged cases have been saved.")
            except Exception as e:
                st.error(f"Error running auto-flagging logic: {e}")

    # Case Management
#     elif page == "Case Management":
#         st.title("Case Management")
#         st.write("View, update, and manage case details.")
        
#         case_id_to_edit = st.text_input("Enter Case ID to edit:", "")
#         if case_id_to_edit:
#             case_data = df[df["Child ID"].astype(str) == case_id_to_edit]
#             if not case_data.empty:
#                 st.write("Case Details:")
#                 st.dataframe(case_data)

#                 # Editable fields
#                 new_status = st.selectbox(
#                     "Update Case Outcome:", ["Ongoing", "Resolved", "Closed", "Dismissed"],
#                     index=["Ongoing", "Resolved", "Closed", "Dismissed"].index(case_data["Case Outcome"].values[0])
#                 )
#                 new_notes = st.text_area("Update Notes:", value=case_data["Notes"].values[0])
#                 new_investigator = st.text_input("Update Assigned Investigator:", value=case_data["Assigned Investigator"].values[0])
                
#                 if st.button("Save Changes"):
#                     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     if new_notes.strip():
#                         updated_notes = f"{case_data['Notes'].values[0]}\n{new_notes} (Updated on: {timestamp})"
#                     else:
#                         updated_notes = case_data["Notes"].values[0]

#                     # Update the DataFrame
#                     df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Case Outcome"] = new_status
#                     df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Notes"] = updated_notes
#                     df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Assigned Investigator"] = new_investigator
                    
#                     # Save the updated DataFrame
#                     try:
# #                         csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final_Dataset_CAPS.csv")
#                         csv_path = os.path.join(os.getcwd(), "Final_Dataset_CAPS.csv")
#                         df.to_csv(csv_path, index=False)
#                         st.success(f"Changes saved for Case ID {case_id_to_edit}!")
#                     except Exception as e:
#                         st.error(f"Error saving changes: {e}")
#             else:
#                 st.warning("No case found with the provided Case ID.")
                
    elif page == "Case Management":
        st.title("Case Management")
        st.write("View, update, and manage case details.")
        case_id_to_edit = st.text_input("Enter Case ID to edit:", "")
        if case_id_to_edit:
            case_data = df[df["Child ID"].astype(str) == case_id_to_edit]
            if not case_data.empty:
                st.write("Case Details:")
                st.dataframe(case_data)
                new_status = st.selectbox(
                    "Update Case Outcome:", ["Ongoing", "Resolved", "Closed","Dismissed"],
                    index=["Ongoing", "Resolved", "Closed","Dismissed"].index(case_data["Case Outcome"].values[0])
                )
                new_notes = st.text_area("Update Notes:", value=case_data["Notes"].values[0])
                new_investigator = st.text_input("Update Assigned Investigator:", value=case_data["Assigned Investigator"].values[0])
                if st.button("Save Changes"):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if new_notes.strip():
                        updated_notes = f"{case_data['Notes'].values[0]}\n{new_notes} (Updated on: {timestamp})"
                    else:
                        updated_notes = case_data["Notes"].values[0]
                    df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Case Outcome"] = new_status
                    df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Notes"] = updated_notes
                    df.loc[df["Child ID"].astype(str) == case_id_to_edit, "Assigned Investigator"] = new_investigator
                    try:
#                         csv_path = "Final_Dataset_CAPS.csv"
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        csv_path = os.path.join(current_dir, "data", "Final_Dataset.csv")
                        df.to_csv(csv_path, index=False)
                        load_data.clear()
                        df = load_data()
                        st.success(f"Changes saved for Case ID {case_id_to_edit}!")
                    except Exception as e:
                        st.error(f"Failed to save changes: {e}")
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
        
        # Visualization 3: Sankey Diagram
        st.subheader("Case Flow Visualization (Region → Severity → Outcome)")
        sankey_data = df.groupby(["Region", "Severity", "Case Outcome"]).size().reset_index(name="Count")

        fig3 = go.Figure(
            go.Sankey(
                node=dict(
                    pad=10,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=list(set(sankey_data["Region"]) | set(sankey_data["Severity"]) | set(sankey_data["Case Outcome"])),
                ),
                link=dict(
                    source=sankey_data["Region"].apply(lambda x: list(sankey_data["Region"]).index(x)),
                    target=sankey_data["Severity"].apply(lambda x: list(sankey_data["Severity"]).index(x) + len(set(sankey_data["Region"]))),
                    value=sankey_data["Count"]
                )
            )
        )

        fig3.update_layout(title_text="Flow of Cases (Region → Severity → Outcome)", font_size=10)
        st.plotly_chart(fig3)
        



        # Load your dataset
#         csv_path = os.path.join(os.path.dirname(__file__), "Final_Dataset_CAPS.csv")
#         df = pd.read_csv(csv_path)

#         # Ensure county names are standardized (e.g., without " County" suffix)
#         df["County"] = df["County"].str.replace(" County", "", regex=False)

#         # Aggregate data by county
#         county_data = df.groupby("County").size().reset_index(name="Case Count")

#         # Load Mississippi county GeoJSON
#         geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

#         # Filter GeoJSON for Mississippi counties (FIPS codes starting with 28)
# #         import requests
#         geojson_data = requests.get(geojson_url).json()
#         mississippi_geojson = {
#             "type": "FeatureCollection",
#             "features": [
#                 feature for feature in geojson_data["features"] if feature["properties"]["STATE"] == "28"
#             ]
#         }

#         # Add a title
#         st.title("Case Distribution by County in Mississippi")

#         # Create a choropleth map
#         fig = px.choropleth(
#             county_data,
#             geojson=mississippi_geojson,
#             locations="County",  # Match county names with GeoJSON
#             featureidkey="properties.NAME",  # Match GeoJSON feature with county names
#             color="Case Count",  # Color by the number of cases
#             color_continuous_scale="Viridis",
#             title="Case Distribution Across Mississippi Counties",
#         )

#         # Update geographic scope to focus on Mississippi
#         fig.update_geos(
#             visible=False,
#             resolution=50,
#             projection=dict(type="mercator"),
#             center=dict(lat=32.7765, lon=-89.6678),  # Mississippi center coordinates
#             fitbounds="locations",
#         )

#         # Display the map in Streamlit
#         st.plotly_chart(fig)

        # Ensure county names are standardized (remove " County" suffix)
#         df["County"] = df["County"].str.replace(" County", "", regex=False)

#         # Aggregate data by county
#         county_data = df.groupby("County").size().reset_index(name="Case Count")

#         # Load Mississippi county GeoJSON
#         geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
#         geojson_data = requests.get(geojson_url).json()

#         # Filter GeoJSON for Mississippi counties (FIPS codes starting with 28)
#         mississippi_geojson = {
#             "type": "FeatureCollection",
#             "features": [
#                 feature for feature in geojson_data["features"] if feature["properties"]["STATE"] == "28"
#             ]
#         }

#         # Create a list of all counties in Mississippi
#         mississippi_counties = [feature["properties"]["NAME"] for feature in mississippi_geojson["features"]]

#         # Add counties with zero cases to the dataset to ensure all counties are displayed
#         for county in mississippi_counties:
#             if county not in county_data["County"].values:
#                 county_data = pd.concat([county_data, pd.DataFrame({"County": [county], "Case Count": [0]})])

#         # Create a choropleth map
#         fig = px.choropleth(
#             county_data,
#             geojson=mississippi_geojson,
#             locations="County",  # Match county names with GeoJSON
#             featureidkey="properties.NAME",  # Match GeoJSON feature with county names
#             color="Case Count",  # Color based on case count
#             color_continuous_scale="Viridis",
#             title="Case Distribution Across Mississippi Counties",
#             labels={"Case Count": "Number of Cases"},
#         )

#         # Update geographic scope to focus on Mississippi
#         fig.update_geos(
#             visible=False,
#             resolution=50,
#             projection=dict(type="mercator"),
#             center=dict(lat=32.7765, lon=-89.6678),  # Mississippi center coordinates
#             fitbounds="locations",
#         )
#         csv_path = os.path.join(os.path.dirname(__file__), "Final_Dataset_CAPS.csv")
#         df = pd.read_csv(csv_path)

#         # Ensure county names are standardized (remove " County" suffix)
#         df["County"] = df["County"].str.replace(" County", "", regex=False)

#         # Aggregate data by county
#         county_data = df.groupby("County").size().reset_index(name="Case Count")

#         # Load Mississippi county GeoJSON
#         geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
#         geojson_data = requests.get(geojson_url).json()

#         # Filter GeoJSON for Mississippi counties (FIPS codes starting with 28)
#         mississippi_geojson = {
#             "type": "FeatureCollection",
#             "features": [
#                 feature for feature in geojson_data["features"] if feature["properties"]["STATE"] == "28"
#             ]
#         }

#         # Create a list of all counties in Mississippi
#         mississippi_counties = [feature["properties"]["NAME"] for feature in mississippi_geojson["features"]]

#         # Add counties with zero cases to the dataset to ensure all counties are displayed
#         for county in mississippi_counties:
#             if county not in county_data["County"].values:
#                 county_data = pd.concat([county_data, pd.DataFrame({"County": [county], "Case Count": [0]})])

#         # Choropleth Map
#         fig_choropleth = px.choropleth(
#             county_data,
#             geojson=mississippi_geojson,
#             locations="County",  # Match county names with GeoJSON
#             featureidkey="properties.NAME",  # Match GeoJSON feature with county names
#             color="Case Count",  # Color based on case count
#             color_continuous_scale="Viridis",
#             title="Case Distribution Across Mississippi Counties",
#             labels={"Case Count": "Number of Cases"},
#         )

#         # Update geographic scope to focus on Mississippi
#         fig_choropleth.update_geos(
#             visible=False,
#             resolution=50,
#             projection=dict(type="mercator"),
#             center=dict(lat=32.7765, lon=-89.6678),  # Mississippi center coordinates
#             fitbounds="locations",
#         )

#         # Bubble Map Data Preparation
#         bubble_data = df.groupby(["County"]).agg({"Severity": "count"}).reset_index()
#         bubble_data.columns = ["County", "Bubble Size"]

#         # Bubble Map
#         fig_bubble = px.scatter_geo(
#             bubble_data,
#             geojson=mississippi_geojson,
#             locations="County",
#             featureidkey="properties.NAME",
#             size="Bubble Size",
#             color="Bubble Size",
#             color_continuous_scale="Plasma",
#             title="Bubble Map of Severity Across Counties",
#             labels={"Bubble Size": "Severity Count"},
#         )

#         # Update Bubble Map to align with Mississippi focus
#         fig_bubble.update_geos(
#             visible=False,
#             resolution=50,
#             projection=dict(type="mercator"),
#             center=dict(lat=32.7765, lon=-89.6678),  # Mississippi center coordinates
#             fitbounds="locations",
#         )

#         # Display Both Maps in Streamlit
#         st.title("Integrated Visualization for Mississippi Case Distribution")
#         st.subheader("Choropleth Map: Case Distribution")
#         st.plotly_chart(fig_choropleth)

#         st.subheader("Bubble Map: Case Severity Across Counties")
#         st.plotly_chart(fig_bubble)

    

#         # Display the map in Streamlit
#         st.title("Case Distribution by County in Mississippi")
#         st.plotly_chart(fig)
#         st.subheader("Trend Over Time by County")

#         # Prepare data for time trend visualization
#         time_trend_data = df.groupby(["Date", "County"]).size().reset_index(name="Case Count")

#         # Create a line chart
#         fig_time = px.line(
#             time_trend_data,
#             x="Date",
#             y="Case Count",
#             color="County",
#             title="Case Trends Over Time by County",
#             labels={"Case Count": "Number of Cases"}
#         )

#         # Display chart in Streamlit
#         st.plotly_chart(fig_time)
        
        
#         csv_path = os.path.join(os.path.dirname(__file__), "Final_Dataset_CAPS.csv")
#         df = pd.read_csv(csv_path)        
        
#         # Ensure county names are standardized (remove " County" suffix)
#         df["County"] = df["County"].str.replace(" County", "", regex=False)

#         # Aggregate data by county
#         bubble_data = df.groupby(["County"]).size().reset_index(name="Case Count")

#         # Load Mississippi county GeoJSON
#         geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
#         geojson_data = requests.get(geojson_url).json()

#         # Filter GeoJSON for Mississippi counties (FIPS codes starting with 28)
#         mississippi_geojson = {
#             "type": "FeatureCollection",
#             "features": [
#                 feature for feature in geojson_data["features"] if feature["properties"]["STATE"] == "28"
#             ]
#         }

#         # Bubble Map with Mississippi Counties
#         fig = px.scatter_geo(
#             bubble_data,
#             geojson=mississippi_geojson,
#             locations="County",
#             featureidkey="properties.NAME",
#             size="Case Count",
#             color="Case Count",
#             color_continuous_scale="Viridis",
#             title="Bubble Map: Case Distribution Across Mississippi Counties",
#         )

#         # Update the map's focus to Mississippi
#         fig.update_geos(
#             visible=False,
#             resolution=50,
#             projection=dict(type="mercator"),
#             center=dict(lat=32.7765, lon=-89.6678),  # Mississippi's center coordinates
#             fitbounds="locations",
#         )

#         # Display the map in Streamlit
#         st.title("Bubble Map: Case Distribution in Mississippi")
#         st.plotly_chart(fig)
       

        county_coordinates = {
    'Adams': {'lat': 31.4904067, 'lon': -91.3297733},
    'Alcorn': {'lat': 34.8734817, 'lon': -88.5667345},
    'Amite': {'lat': 31.1729865, 'lon': -90.8217568},
    'Attala': {'lat': 33.0777868, 'lon': -89.5680661},
    'Benton': {'lat': 34.8277278, 'lon': -89.1960486},
    'Bolivar': {'lat': 33.7702263, 'lon': -90.8519798},
    'Calhoun': {'lat': 33.9253624, 'lon': -89.3231269},
    'Carroll': {'lat': 33.4230182, 'lon': -89.9110025},
    'Chickasaw': {'lat': 33.9137474, 'lon': -88.9357382},
    'Choctaw': {'lat': 33.3569348, 'lon': -89.2390212},
    'Claiborne': {'lat': 31.9640367, 'lon': -90.9141968},
    'Clarke': {'lat': 32.0396256, 'lon': -88.7003965},
    'Clay': {'lat': 33.6393012, 'lon': -88.7957063},
    'Coahoma': {'lat': 34.1931314, 'lon': -90.5653906},
    'Copiah': {'lat': 31.8563532, 'lon': -90.4798717},
    'Covington': {'lat': 31.6143213, 'lon': -89.5278045},
    'DeSoto': {'lat': 34.8702932, 'lon': -89.9778704},
    'Forrest': {'lat': 31.1540772, 'lon': -89.2398601},
    'Franklin': {'lat': 31.467398, 'lon': -90.8992648},
    'George': {'lat': 30.857421, 'lon': -88.6537118},
    'Greene': {'lat': 31.2049136, 'lon': -88.6429687},
    'Grenada': {'lat': 33.772401, 'lon': -89.7674875},
    'Hancock': {'lat': 30.3892456, 'lon': -89.4782116},
    'Harrison': {'lat': 30.4553392, 'lon': -89.1313136},
    'Hinds': {'lat': 32.2506391, 'lon': -90.4793259},
    'Holmes': {'lat': 33.1018124, 'lon': -90.0656168},
    'Humphreys': {'lat': 33.1282074, 'lon': -90.5408182},
    'Issaquena': {'lat': 32.6773851, 'lon': -91.0097866},
    'Itawamba': {'lat': 34.2774018, 'lon': -88.3713687},
    'Jackson': {'lat': 30.4899024, 'lon': -88.6486325},
    'Jasper': {'lat': 32.0216075, 'lon': -89.1132111},
    'Jefferson': {'lat': 31.7256789, 'lon': -91.0335766},
    'Jefferson Davis': {'lat': 31.5576951, 'lon': -89.8386843},
    'Jones': {'lat': 31.6058959, 'lon': -89.1636245},
    'Kemper': {'lat': 32.7307497, 'lon': -88.6604179},
    'Lafayette': {'lat': 34.3519035, 'lon': -89.4664677},
    'Lamar': {'lat': 31.2109862, 'lon': -89.5152337},
    'Lauderdale': {'lat': 32.3905206, 'lon': -88.689636},
    'Lawrence': {'lat': 31.5129753, 'lon': -90.1205396},
    'Leake': {'lat': 32.7401114, 'lon': -89.5241292},
    'Lee': {'lat': 34.265749, 'lon': -88.6913264},
    'Leflore': {'lat': 33.5244154, 'lon': -90.288095},
    'Lincoln': {'lat': 31.4885409, 'lon': -90.4480916},
    'Lowndes': {'lat': 33.4466925, 'lon': -88.4108384},
    'Madison': {'lat': 32.6308318, 'lon': -90.0040817},
    'Marion': {'lat': 31.2247688, 'lon': -89.8217717},
    'Marshall': {'lat': 34.7329208, 'lon': -89.4725635},
    'Monroe': {'lat': 33.8884929, 'lon': -88.4747009},
    'Montgomery': {'lat': 33.5074528, 'lon': -89.6182472},
    'Neshoba': {'lat': 32.7237573, 'lon': -89.103199},
    'Newton': {'lat': 32.3882835, 'lon': -89.1317486},
    'Noxubee': {'lat': 33.1190254, 'lon': -88.5916703},
    'Oktibbeha': {'lat': 33.4131096, 'lon': -88.8937318},
    'Panola': {'lat': 34.3793392, 'lon': -89.9601925},
    'Pearl River': {'lat': 30.7761604, 'lon': -89.5984726},
    'Perry': {'lat': 31.153317, 'lon': -88.9830577},
    'Pike': {'lat': 31.1712493, 'lon': -90.4093394},
    'Pontotoc': {'lat': 34.2114655, 'lon': -89.0382651},
    'Prentiss': {'lat': 34.6222512, 'lon': -88.5153286},
    'Quitman': {'lat': 34.1920186, 'lon': -90.2946542},
    'Rankin': {'lat': 32.2338702, 'lon': -89.9484651},
    'Scott': {'lat': 32.3805744, 'lon': -89.5092509},
    'Sharkey': {'lat': 32.8742585, 'lon': -90.8478339},
    'Simpson': {'lat': 31.8919593, 'lon': -89.9323619},
    'Smith': {'lat': 32.0018336, 'lon': -89.4877278},
    'Stone': {'lat': 30.7843863, 'lon': -89.1342764},
    'Sunflower': {'lat': 33.4980504, 'lon': -90.602067},
    'Tallahatchie': {'lat': 33.9414425, 'lon': -90.1577411},
    'Tate': {'lat': 34.6391136, 'lon': -89.9335244},
    'Tippah': {'lat': 34.7296814, 'lon': -88.9119403},
    'Tishomingo': {'lat': 34.7238634, 'lon': -88.2268403},
    'Tunica': {'lat': 34.6398783, 'lon': -90.3623833},
    'Union': {'lat': 34.4834555, 'lon': -88.9789198},
    'Walthall': {'lat': 31.1241919, 'lon': -90.1197687},
    'Warren': {'lat': 32.3649516, 'lon': -90.7975574},
    'Washington': {'lat': 33.2626706, 'lon': -90.9183546},
    'Wayne': {'lat': 31.6212491, 'lon': -88.6858852},
    'Webster': {'lat': 33.6197203, 'lon': -89.2646598},
    'Wilkinson': {'lat': 31.1371924, 'lon': -91.3329359},
    'Winston': {'lat': 33.0603103, 'lon': -89.0495278},
    'Yalobusha': {'lat': 34.0213455, 'lon': -89.6991703},
    'Yazoo': {'lat': 32.770896, 'lon': -90.4120889}
}
        # Prepare data for visualization
        #csv_path = os.path.join(os.path.dirname(__file__), "Final_Dataset_CAPS.csv")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir,"data", "Final_Dataset.csv")
        df = pd.read_csv(csv_path)

        # Standardize county names
        df["County"] = df["County"].str.replace(" County", "", regex=False)

        # Aggregate data by county
        bubble_data = df.groupby("County").size().reset_index(name="Case Count")

        # Add latitude and longitude to the aggregated data
        bubble_data["Latitude"] = bubble_data["County"].map(lambda x: county_coordinates.get(x, {}).get("lat"))
        bubble_data["Longitude"] = bubble_data["County"].map(lambda x: county_coordinates.get(x, {}).get("lon"))

        # Ensure no mising latitude or longitude values
        bubble_data = bubble_data.dropna(subset=["Latitude", "Longitude"])

        # Bubble Map Visualization
        st.title("Bubble Map: Case Distribution Across Mississippi Counties")
        fig = px.scatter_mapbox(
            bubble_data,
            lat="Latitude",
            lon="Longitude",
            size="Case Count",
            color="Case Count",
            color_continuous_scale="Viridis",
            title="Case Distribution Across Mississippi Counties",
            mapbox_style="carto-positron",
            hover_data={
        "County": True,  # Display County name
        "Case Count": True,  # Display Case Count
        "Latitude": True,  # Display Latitude
        "Longitude": True  # Display Longitude
    },
            zoom=6,  # Adjust zoom level for Mississippi
            center={"lat": 32.7765, "lon": -89.6678}  # Center around Mississippi
        )


        # Display the map
        st.plotly_chart(fig)

#         # File path for the dataset
#         csv_path = os.path.join(os.path.dirname(__file__), "Final_Dataset_CAPS.csv")
#         df = pd.read_csv(csv_path)

#         # Ensure county names are standardized (remove " County" suffix)
#         df["County"] = df["County"].str.replace(" County", "", regex=False)

#         # Aggregate data by county
#         bubble_data = df.groupby(["County"]).size().reset_index(name="Case Count")

#         # Load Mississippi county GeoJSON
#         geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
#         geojson_data = requests.get(geojson_url).json()

#         # Filter GeoJSON for Mississippi counties (FIPS codes starting with 28)
#         mississippi_geojson = {
#             "type": "FeatureCollection",
#             "features": [
#                 feature for feature in geojson_data["features"] if feature["properties"]["STATE"] == "28"
#             ]
#         }

#         # Bubble Map with Mississippi Counties
#         fig = px.scatter_mapbox(
#             bubble_data,
#             geojson=mississippi_geojson,
#             locations="County",
#             featureidkey="properties.NAME",
#             size="Case Count",
#             color="Case Count",
#             color_continuous_scale="Viridis",
#             title="Bubble Map: Case Distribution Across Mississippi Counties",
#             hover_name="County",
#             mapbox_style="carto-darkmatter",  # Use Mapbox's dark theme for better contrast
#             center={"lat": 32.7765, "lon": -89.6678},  # Center on Mississippi
#             zoom=6,  # Adjust zoom level for Mississippi
#         )

#         # Display the map in Streamlit
#         st.title("Bubble Map: Case Distribution in Mississippi")
#         st.plotly_chart(fig, use_container_width=True)

        # Time Series Analysis
        st.subheader("Time Series Analysis: Number of Cases Over Time")

        # Group by date and count cases
        time_series_data = df.groupby("Date").size().reset_index(name="Case Count")

        # Line chart
        fig = px.line(
            time_series_data,
            x="Date",
            y="Case Count",
            title="Number of Cases Over Time",
            labels={"Date": "Date", "Case Count": "Number of Cases"},
            markers=True,
        )

        # Add rangeslider for better interactivity
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

        st.plotly_chart(fig)

        # Treemap: Case Severity and Abuse Type
        st.subheader("Treemap of Severity by Abuse Type")

        # Create treemap
        fig = px.treemap(
            df,
            path=["Abuse Type", "Severity"],
            values="Child ID",  # Use the Child ID as a proxy for the number of cases
            title="Distribution of Severity Across Abuse Types",
            color="Severity",
            color_discrete_map={
                "Low": "lightgreen",
                "Medium": "orange",
                "High": "red"
            },
        )

        st.plotly_chart(fig)
        
        
        # Heatmap of Cases by Region and Severity
        st.subheader("Heatmap: Cases by Region and Severity")

        heatmap_data = df.groupby(["Region", "Severity"]).size().reset_index(name="Case Count")

        # Heatmap
        fig = px.density_heatmap(
            heatmap_data,
            x="Region",
            y="Severity",
            z="Case Count",
            color_continuous_scale="Viridis",
            title="Heatmap of Cases by Region and Severity",
            labels={"Region": "Region", "Severity": "Severity", "Case Count": "Number of Cases"}
        )

        st.plotly_chart(fig)

        # Sunburst Chart
        st.subheader("Sunburst Chart: Multi-Level Breakdown")

        # Create a Sunburst chart
        fig = px.sunburst(
            df,
            path=["Region", "Abuse Type", "Severity"],
            values="Child ID",  # Use Child ID as a proxy for the number of cases
            title="Multi-Level Breakdown of Cases",
            color="Severity",
            color_discrete_map={
                "Low": "green",
                "Medium": "orange",
                "High": "red"
            },
        )

        st.plotly_chart(fig)




    
        # 3D Scatter Plot
        st.subheader("3D Scatter Plot: Multidimensional View")

        # Aggregate data by region
        scatter_data = df.groupby("Region").agg({"Age": "mean", "Child ID": "count", "Severity": "count"}).reset_index()
        scatter_data.columns = ["Region", "Average Age", "Case Count", "Severity Count"]

        # Create a 3D scatter plot
        fig = px.scatter_3d(
            scatter_data,
            x="Average Age",
            y="Case Count",
            z="Severity Count",
            color="Case Count",
            size="Case Count",
            hover_data=["Region"],
            title="3D Scatter Plot of Cases",
        )

        st.plotly_chart(fig)


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
    # Similar Cases Page
    elif page == "Similar Case Discovery":
        st.title("Find Cases with Similar Patterns")
        case_description = st.text_area("Enter a case description to find similar cases:")
        if st.button("Find Similar Cases"):
            if case_description.strip():
                similar_cases = find_similar_cases(case_description, df)
                st.subheader("Top Similar Cases")
                st.dataframe(similar_cases)
            else:
                st.warning("Please enter a case description.")

         

            





if __name__ == "__main__":
    main()
