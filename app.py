import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
  page_title="Job Salary Predictor",
  page_icon="💼",
  layout="wide"
)

# Load model + encoders
model = joblib.load('model/model.pkl')
explainer = joblib.load('model/explainer.pkl')
le_industry = joblib.load('model/le_industry.pkl')
le_location = joblib.load('model/le_location.pkl')
le_title = joblib.load('model/le_title.pkl')
oe = joblib.load('model/oe_education.pkl')
oe2 = joblib.load('model/oe_company.pkl')

# Load raw data for percentile + comparator
df_raw = pd.read_csv('data/job_salary.csv').dropna()

def get_seniority(title):
  title = str(title).lower()
  if any(x in title for x in ['junior','jr','entry']): return 1
  elif any(x in title for x in ['senior','sr','lead','principal']): return 3
  elif any(x in title for x in ['manager','director','head','vp']): return 4
  else: return 2

def predict_salary(title, exp, edu, skills, industry, company, location, remote, certs):
  seniority = get_seniority(title)
  edu_enc = oe.transform([[edu]])[0][0]
  comp_enc = oe2.transform([[company]])[0][0]
  ind_enc = le_industry.transform([industry])[0]
  loc_enc = le_location.transform([location])[0]
  title_enc = le_title.transform([title])[0]
  remote_b = 1 if remote == 'Yes' else 0

  row = pd.DataFrame([[exp, skills, certs, remote_b, seniority,
                        edu_enc, comp_enc, ind_enc, loc_enc, title_enc]],
                  columns=['experience_years','skills_count','certifications',
                             'remote_work','seniority','education_encoded',
                             'company_size_encoded','industry_encoded',
                             'location_encoded','job_title_encoded'])
  pred_log = model.predict(row)[0]
  return np.expm1(pred_log), row

st.title("Job Salary Predictor")
st.markdown("Fill in your profile to get your predicted salary and insights.")

st.sidebar.header("Your Profile")

title = st.sidebar.selectbox("Job title", sorted(df_raw['job_title'].unique()))
exp = st.sidebar.slider("Years of experience", 0, 30, 3)
edu = st.sidebar.selectbox("Education level", ['High School','Diploma','Bachelor','Master','PhD'])
skills = st.sidebar.slider("Number of skills", 1, 20, 5)
industry = st.sidebar.selectbox("Industry", sorted(df_raw['industry'].unique()))
company = st.sidebar.selectbox("Company size", ['Small','Medium','Large'])
location = st.sidebar.selectbox("Location", sorted(df_raw['location'].unique()))
remote = st.sidebar.selectbox("Remote work", ['Yes','No'])
certs = st.sidebar.slider("Certifications", 0, 10, 0)

salary, input_row = predict_salary(title, exp, edu, skills, industry, company, location, remote, certs)

# Salary range from dataset group
group = df_raw[df_raw['industry'] == industry]['salary']
p25 = int(np.percentile(group, 25))
p75 = int(np.percentile(group, 75))

# Percentile rank
from scipy.stats import percentileofscore
pct = int(percentileofscore(df_raw['salary'], salary))

st.subheader("Predicted Salary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Estimated salary", f"${int(salary):,}")
col2.metric("Industry P25", f"${p25:,}")
col3.metric("Industry P75", f"${p75:,}")
col4.metric("You are in top", f"{100 - pct}%")

st.subheader("What's affecting your salary?")
shap_vals = explainer.shap_values(input_row)
fig, ax = plt.subplots()
shap.waterfall_plot(
  shap.Explanation(values=shap_vals[0],
    base_values=explainer.expected_value,
    data=input_row.values[0],
    feature_names=input_row.columns.tolist()),
  show=False)
st.pyplot(fig)

st.subheader("Your salary across all industries")
industries = sorted(df_raw['industry'].unique())
salaries = []
for ind in industries:
  s, _ = predict_salary(title, exp, edu, skills, ind, company, location, remote, certs)
  salaries.append(s)

fig2 = px.bar(x=industries, y=salaries,
              labels={'x':'Industry','y':'Predicted Salary'},
              color=salaries, color_continuous_scale='teal')
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Negotiation Coach")
offer = st.number_input("Enter the offer you received ($)", min_value=0, value=50000, step=1000)
if offer > 0:
  diff = int(salary - offer)
  pct_diff = round((diff / offer) * 100, 1)
  if diff > 0:
    st.success(f"This offer is {pct_diff}% below market. You can negotiate up to ${int(salary):,}")
  elif diff < 0:
    st.info(f"Great offer! This is {abs(pct_diff)}% above market rate.")
  else:
    st.write("This offer matches market rate exactly.")

