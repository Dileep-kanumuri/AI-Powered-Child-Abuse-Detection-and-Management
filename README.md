<<<<<<< HEAD
# **AI-Powered Child Abuse Detection and Management**
### **By SAI Pro Systems LLC**

![Project Screenshot](https://path-to-your-screenshot.png)

---

## **Overview**
"AI-Powered Child Abuse Detection and Management" is a Streamlit-based application designed to analyze and manage child abuse cases using advanced machine learning techniques. This tool leverages natural language processing (NLP) powered by a pre-trained BERT model for abuse type prediction and employs data visualization and forecasting to assist investigators, social workers, and administrators in their decision-making process.

> **Note**: All data used in this project is **synthetically generated** and does not represent real-world cases. This application is for **educational and research purposes only**.

---

## **Features**
- **Home Page**: Search and filter case data using various attributes (e.g., `Child ID`, `Region`, `Abuse Type`).
- **Predict Abuse Type**: Use the BERT model to predict abuse types (e.g., `Neglect`, `Physical`, `Emotional`) and get confidence scores based on case descriptions.
- **Auto-Flagging Logic**: Automatically flag cases based on severity and confidence scores for quicker intervention.
- **Case Management**: View, edit, and update case details (e.g., `Case Outcome`, `Notes`, `Assigned Investigator`).
- **Visualizations**: Explore interactive charts, including heatmaps, bar charts, and pie charts, for insights into abuse cases.
- **Forecasting**: Predict future trends in child abuse cases using Prophet forecasting models.
- **Professional UI**: Modern dark-themed interface with vibrant orange accents for an intuitive user experience.

---

## **Technology Stack**
- **Frontend**: [Streamlit](https://streamlit.io/) (Interactive web interface)
- **Backend**: Python
- **Machine Learning**: BERT (pre-trained NLP model from Hugging Face's Transformers library)
- **Visualization**: Plotly
- **Forecasting**: Prophet (Time-series forecasting)

---

## **Dataset**
The dataset used in this project is synthetically generated and contains the following columns:
- `Child ID`
- `Age`
- `Gender`
- `Region`
- `Abuse Type`
- `Severity`
- `Reported By`
- `Support Provided`
- `Case Outcome`
- `Case Status`
- `Assigned Investigator`
- `Notes`
- `Date`

**Disclaimer**: The dataset is entirely synthetic and has been created solely for the purpose of demonstrating the capabilities of this application. It does not contain any real-world data or sensitive information.

---

## **Project Structure**
```plaintext
Child_Abuse/
│
├── app.py                         # Main Streamlit application
├── bert_model.py                  # Code for training the BERT model
├── requirements.txt               # Python dependencies
├── data/                          # Folder containing datasets
│   ├── Final_Dataset_CAPS.csv     # Main dataset
│   ├── synthetic_child_abuse_dataset.csv
│   ├── updated_synthetic_child_abuse_dataset.csv
├── bert_abuse_model_v2/           # Pre-trained BERT model files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
├── README.md                      # Project documentation
└── venv/                          # Python virtual environment (optional)
=======
# AI-Powered-Child-Abuse-Detection-and-Management
AI-Powered Child Abuse Detection and Management" is a cutting-edge application developed by SAI Pro Systems LLC that leverages machine learning and natural language processing (NLP) to analyze, predict, and manage child abuse cases.
>>>>>>> f1492243870d002de2f5a125ac80429b86ad8aba
