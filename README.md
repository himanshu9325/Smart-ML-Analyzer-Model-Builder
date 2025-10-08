# Smart-ML-Analyzer-Model-Builder
🤖 Smart ML Analyzer & Model Builder
📘 Overview

Smart ML Analyzer is an interactive Streamlit web app that allows users to:
Upload a CSV file
Automatically clean missing values
Explore their dataset visually through charts and correlations
Select and train machine learning models (Regression or Classification)
View model metrics and performance plots — all in one place
This tool is perfect for quick exploratory data analysis (EDA) and machine learning experimentation — no coding required!

🚀 Features
✅ CSV Upload & Auto-Cleaning
Upload any structured dataset
Automatically removes rows with null values
Displays dataset summary and statistics
✅ Interactive Data Visualization
Correlation heatmap
Distribution plots
Scatter plots for numeric relationships
✅ Machine Learning Model Builder
Choose between Regression or Classification tasks

Available models:
Regression: Linear Regression, Random Forest Regressor
Classification: Logistic Regression, Random Forest Classifier
Auto data encoding (for categorical features)

Model metrics and plots:
R² Score, MSE for regression
Accuracy, Confusion Matrix, and Classification Report for classification


🛠️ Tech Stack
Frontend/UI: Streamlit
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn

📦 Installation

Clone this repository:
git clone https://github.com/himanshu9325/smart-ml-analyzer.git
cd smart-ml-analyzer

Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate        # For Windows

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py
Open your browser:
http://localhost:8501

🧩 Example Workflow

Go to the Upload & Clean tab
→ Upload your CSV → App removes nulls and displays clean summary

Switch to Data Visualization
→ Explore correlations and data distributions

Move to Model Training
→ Choose target column and ML model
→ View metrics and performance charts

📁 Project Structure
smart-ml-analyzer/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── sample_dataset.csv      # (Optional) Example dataset for testing

🧑‍💻 Author
Himanshu Sangitrao
🔗 GitHub
💼 Passionate about Data Science, Machine Learning, and Intelligent Automation.
