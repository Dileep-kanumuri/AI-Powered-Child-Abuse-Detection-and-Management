# **AI-Powered Child Abuse Detection and Management**
### **By SAI Pro Systems LLC**

---

## **Overview**
The **AI-Powered Child Abuse Detection and Management** project leverages artificial intelligence and natural language processing (NLP) to analyze and manage child abuse cases. Built using Streamlit, this prototype demonstrates how advanced machine learning models like BERT can be utilized for abuse detection, case management, and predictive analytics. The project is open-source and deployed for public exploration and experimentation.

> **Note**: All data used is **synthetically generated** for demonstration purposes. This application is for **educational and research purposes only** and does not contain real-world child welfare data.

---

## **Features**
- **Home Page**: Search and filter child welfare case data by attributes like Child ID, Region, and Abuse Type.
- **Abuse Type Prediction**: Predicts abuse types (e.g., Neglect, Physical, Emotional) using a fine-tuned BERT model.
- **Auto-Flagging System**: Flags high-risk cases based on severity and confidence thresholds for faster interventions.
- **Similar Case Discovery**: Finds the hight priority cases with same patterns and displays the top 5 cases .
- **Case Management**: Update and manage case details such as status, assigned investigator, and notes.
- **Visualizations**: Interactive charts, including heatmaps, bar charts, pie charts, and sunburst diagrams.
- **Forecasting**: Predicts future trends in child abuse case reporting to aid planning and resource allocation.

---

## **Explore the Project**
This project is open for exploration and contribution. Access it here:

- **GitHub Repository**: [AI-Powered Child Abuse Detection and Management](https://github.com/Dileep-kanumuri/AI-Powered-Child-Abuse-Detection-and-Management)
- **Streamlit Application**: [AI-Powered Application Deployment](https://ai-powered-child-abuse-detection.streamlit.app/)

We encourage you to experiment with the app, provide feedback, and contribute enhancements to foster collaboration and innovation in addressing child abuse through AI-powered solutions.

---

## **Technology Stack**
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **Machine Learning**: BERT (via Hugging Face Transformers)
- **Visualization**: Plotly
- **Forecasting**: Prophet

---

## **How to Use**
1. Clone the repository:
   ```bash
   git clone https://github.com/Dileep-kanumuri/AI-Powered-Child-Abuse-Detection-and-Management.git
   cd AI-Powered-Child-Abuse-Detection-and-Management

1.  Install dependencies:

    bash

    CopyEdit

    `pip install -r requirements.txt`

2.  Run the Streamlit application:

    bash

    CopyEdit

    `streamlit run app.py`

3.  Access the application in your browser and explore its features.

* * * * *

**Project Structure**
---------------------

plaintext

CopyEdit

`AI-Powered-Child-Abuse-Detection-and-Management/
│
├── app.py                         # Main Streamlit application
├── bert_model.py                  # Abuse type classification model
├── requirements.txt               # Python dependencies
├── data/                          # Dataset folder
│   ├── Final_Data.csv
│   ├── updated_synthetic_child_abuse_dataset.csv
├── bert_abuse_model/              # Pre-trained BERT model files
├── README.md                      # Project documentation
└── venv/                          # Python virtual environment (optional)`

* * * * *

**Disclaimer**
--------------

This project is a prototype using synthetic data to simulate real-world scenarios. All examples, visuals, and metrics are derived from artificial datasets for educational purposes only.

* * * * *

**Contributing**
----------------

We welcome contributions to improve this project. Feel free to submit a pull request or report an issue via GitHub.

* * * * *

**License**
-----------

This project is licensed under the MIT License.


CopyEdit

 `You can copy and paste this into your `README.md` file directly. Let me know if you need any further adjustments!`
