
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydeck as pdk
import altair as alt

st.set_page_config(
    page_title="CHD RISK",
    page_icon="🫀"
    
)

st.sidebar.title("Cronical Heart Disease RISK")
st.sidebar.markdown("**Keno Palen**")
st.sidebar.divider()
st.sidebar.subheader("Navigation")

sections = ["🏠Home", "🗃️Dataset", "📊Column Visualizations", "🧔Age Metrics", "🫀CHDRisk Metrics"]
selection = st.sidebar.radio("Go to", sections)

data = pd.read_csv('heart_disease.csv')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')

data['sex'] = imputer.fit_transform(data['sex'] .values.reshape(-1, 1)).ravel()
data['smokingStatus'] = imputer.fit_transform(data['smokingStatus'] .values.reshape(-1, 1)).ravel()


if selection == "🏠Home":
    st.title("🫀Cronical Heart Disease Risk")
    image_path = 'heart.jpg'
    st.image(image_path, use_column_width=True)
    st.write("""Predictive Factors and Risk Assessment for Coronary Heart Disease(CHD).
             The dataset contains information collected from a study on Chronic Heart Disease (CHD) risk over a 
             period of time. It encompasses various attributes related to occupation, family size, lifestyle factors, 
             medical history, and feedback from patients. Explore the data and gain insights into the factors influencing 
             CHD risk through the different sections provided.""")
    st.markdown("**Source:** [Kaggle](https://www.kaggle.com/datasets/mahdifaour/heart-disease-dataset/data)")
    st.divider()
    
    st.write("### Age")
    age_hist = alt.Chart(data).mark_bar().encode(
       alt.X("age:Q"),
       y='count()'
       ).properties(title="Age of Patient")
    st.altair_chart(age_hist, use_container_width=True)
    st.write("The bar chart titled 'Age of Patient' shows the count of records for different age groups among patients")
    st.divider()
    
    st.write("### Gender")
    gender_pie = alt.Chart(data).mark_arc().encode(
        theta=alt.Theta(field="sex", type="nominal", aggregate="count"),
        color=alt.Color(field="sex", type="nominal")
    ).properties(title="Gender of Patient")
    st.altair_chart(gender_pie, use_container_width=True)
    st.write("The pie chart titled 'Gender' displays the proportion of male and female patients in the dataset.")
    st.divider()
    
    st.write("### CHDRisk")
    if 'CHDRisk' in data.columns:
        feedback_pie = alt.Chart(data).mark_arc().encode(
            theta=alt.Theta(field="CHDRisk", type="nominal", aggregate="count"),
            color=alt.Color(field="CHDRisk", type="nominal")
        ).properties(title="CHDRisk")
        st.altair_chart(feedback_pie, use_container_width=True)
        st.write("The Predictive Factors and Risk Assessment for Coronary Heart Disease (CHD) pie chart illustrates predominantly positive indicators")
    else:
        st.write("CHDRisk column is not available in the dataset.")
    st.divider()
    
    st.write("### Sex")
    age_hist = alt.Chart(data).mark_bar().encode(
       alt.X("sex"),
       y='count()'
       ).properties(title="Sex of Patient")
    st.altair_chart(age_hist, use_container_width=True)
    st.write("The bar chart titled 'Sex of Patient'shows that female had high risk in CHD")
    st.divider()
    
    st.write("### Age vs. CHDRisk")
    age_vs_CHDRisk = alt.Chart(data).mark_point().encode(
       x='CHDRisk',
       y='age'
       ).properties(title="Age vs. CHDRisk")
    st.altair_chart(age_vs_CHDRisk, use_container_width=True)
    st.write("This scatter plot reveals ages with no chdrisk and with chdrisk");
    
if selection == '🗃️Dataset':
    st.title("🗃️Dataset")
    st.write("""This section presents the raw dataset containing information regarding patients diagnosed with coronary heart disease (CHD). Feel free to scroll through the table to review all collected data points.""")
    st.markdown("**Source:** [Kaggle](https://www.kaggle.com/datasets/mahdifaour/heart-disease-dataset/data)")
    st.write(data);

# Column Visualizations Section
if selection == "📊Column Visualizations":
    st.title("Column Visualizations")
    st.write("""
    This section includes visual representations for every column within the dataset, employing various charts and plots to aid in comprehending the data's distribution and attributes.""")

    chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Area Chart"])

    # Numeric columns
    st.write("### Numeric Columns")
    for column in data.select_dtypes(include=[np.number]).columns:
        st.write(f"#### {column}")
        if chart_type == "Line Chart" or data[column].nunique() > 20:
            st.line_chart(data[column])
        elif chart_type == "Area Chart":
            st.area_chart(data[column])
        else:
            st.bar_chart(data[column].value_counts())

    # Categorical columns
    st.write("### Categorical Columns")
    for column in data.select_dtypes(include=[object]).columns:
        st.write(f"#### {column}")
        st.bar_chart(data[column].value_counts())

    # Specific visualizations for some columns
    st.write("### Specific Visualizations")
    
    # Gender distribution
    st.write("#### sex Distribution")
    st.bar_chart(data['sex'].value_counts())
    
    # Age distribution
    st.write("#### Age Distribution")
    st.bar_chart(data['age'].value_counts().sort_index())
    
    # Family size distribution
    st.write("#### CHDRisk Distribution")
    st.area_chart(data['CHDRisk'].value_counts().sort_index())
    
    # Feedback distribution
    if 'CHDRisk' in data.columns:
        st.write("#### CHDRisk Distribution")
        CHDRisk_counts = data['CHDRisk'].value_counts()
        st.bar_chart(CHDRisk_counts)
    else:
        st.write("#### CHDRisk Distribution")
        st.write("CHDRisk column is not available in the dataset.")

if selection == "🧔Age Metrics":
    st.title("Age Metrics")
    st.write("""
    This section provides metrics related to the age of the customers. 
    It displays the average age, minimum age, and maximum age of the customers in the dataset.
    """)
    average_age = data['age'].mean()
    min_age = data['age'].min()
    max_age = data['age'].max()
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Age", f"{average_age:.2f}")
    col2.metric("Min Age", f"{min_age}")
    col3.metric("Max Age", f"{max_age}");
    
if selection == "🫀CHDRisk Metrics":
    st.title("Feedback Metrics")
    st.write("""
    This section provides metrics related to CHDRisk. 
    It displays the count of each type of feedback received.
    """)
    if 'CHDRisk' in data.columns:
        feedback_types = data['CHDRisk'].value_counts()
        for feedback_type, count in feedback_types.items():
            st.metric(f"{feedback_type} CHDRisk Count", count)
    else:
        st.write("CHDRisk column is not available in the dataset.")
