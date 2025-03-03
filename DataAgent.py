import os
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


st.title("CSV Query App with LangChain and OpenAI")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    llm = ChatOpenAI(model="gpt-4o")


    # Create the pandas DataFrame agent
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True
    )

    # Query input
    query = st.text_input("Enter your query about the data:")

    if st.button("Submit Query"):
        if query:
            # Execute the query using the agent
            response = agent.run(query)
            st.write("Response:")
            st.write(response)
        else:
            st.write("Please enter a query.")
