~Job Salary Predictor~
A Machine Learning project that predicts estimated salaries based on job-related features such as experience, education, skills,certifications, comapny size and locaton, and job role.
The project uses XGBoost for accurate salary prediction and provides an interactive interface using Streamlit.

----Features----
Salary prediction using Machine Learning
XGBoost regression model
Data preprocessing and feature engineering
Interactive Streamlit web application
Organized project structure
Real-time prediction system

----Tech Stack----
Python
Pandas
NumPy
Scikit-learn
XGBoost
Streamlit
Matplotlib

----Project Structure----
.
├── data/               # Dataset files
├── model/              # Trained model files
├── notebooks/          # Jupyter notebooks
├── app.py              # Streamlit application
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── .gitignore

----Installation----
Clone the repository:
git clone https://github.com/AvaniAgarwal-23/Job-Salary-Predictor.git

Move to project directory:
cd Job-Salary-Predictor

Install dependencies:
pip install -r requirements.txt

----Run the Project----
Start the Streamlit application:
streamlit run app.py

----Workflow----
Load dataset
Preprocess data
Train XGBoost model
Evaluate model performance
Deploy using Streamlit
Predict salary based on user inputs.
