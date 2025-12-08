import streamlit as st
import pandas as pd

st.title("OptiStock – AI Inventory Insights")

st.write("### Upload your inventory dataset")

uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
    except:
        df = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())

    st.write("### Data Summary")
    st.write(df.describe(include='all'))
else:
    st.write("No file uploaded yet.")
