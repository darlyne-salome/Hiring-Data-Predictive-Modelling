# Hiring_data
This project focuses on analyzing and predicting the time it takes to fill job positions based on hiring data. It involves data preprocessing, feature engineering, and predictive modeling using Python. The project also includes visualizations and a final report with key findings.

# Folder Structure
Hiring_data/
│
├── Original_data/              # Raw monthly datasets used in the project
├── visuals/                    # Charts and visualizations generated during EDA and modeling
├── Combined_hiring_data.xlsx   # Cleaned and merged dataset used for modeling
├── Final_Prediction_Report.xlsx # Final report including predicted values
├── Hiring_script.py            # Main Python script for preprocessing, modeling, and prediction
├── path_finding.py             # Helper script for locating and combining datasets from folders
└── README.md                   # Project overview and documentation

# Project Objectives
- Merge multiple monthly datasets into a single comprehensive dataset.
- Clean, preprocess, and engineer relevant features for model training.
- Build a machine learning model to predict Time to Fill for job positions.
- Generate actionable insights and present them in a summarized report.

# Tools & Technologies
- Python: Data processing, modeling and automation
- Pandas & NumPy: Data manipulation
- Scikit-learn: Machine learning
- Matplotlib & Seaborn: Data visualization
- Excel: Manual data review and final report formatting

# Workflow Overview
Raw Data Storage:
Monthly hiring files are stored in the Original_data/ folder.

# Data Merging & Cleaning:
power query (Excel) is used to clean and combine the data into Combined_hiring_data.xlsx.
path_finding.py is used to locate and load the data.


# Feature Engineering & Modeling:
Columns are encoded, irrelevant features dropped, and a model is trained to predict hiring duration.

# Results & Reporting:
Predictions are exported into Final_Prediction_Report.xlsx.
Visuals are stored in the visuals/ folder for insights and presentation.

# Key Insights
- Visuals show trends, correlations, and model performance.
- The final model provides estimates of how long it will take to fill upcoming job roles, aiding HR planning.

 # Future Improvements
- Hyperparameter tuning for improved model accuracy
- Integration with a dashboard for dynamic prediction updates
