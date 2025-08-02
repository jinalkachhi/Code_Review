# here, we seriously build. one thing at a time. 
# we first test with one table and then fix all the things and scale.

import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.prompts import ChatPromptTemplate

# Load environment
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load your database
db = SQLDatabase.from_uri(
    os.environ["DATABASE_URL"],
    include_tables=["sales_by_distributor"],   
    sample_rows_in_table_info=3              
)


# Refined system + human prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful business analyst AI. You help users understand the 'sales_by_distributor' table in a PostgreSQL database.

Your job is to analyze and explain data insights — not just output SQL unless asked.

 Do NOT hallucinate:
- Only use column names that exist in: {table_info}.
- If a column is missing or not relevant, clearly state that.
- If you can’t answer due to data limitations, say so. Don’t guess.

 When a question asks about totals, distributions, or patterns:
- Use correct **aggregations** (e.g., SUM, AVG, COUNT) by product or distributor or time.
- Return **complete answers**, not partial lists (e.g., don’t truncate at top 5 unless explicitly asked).

 If the user asks for rankings or top values:
- Respect the value of top_k: {top_k}.
- Always sort based on appropriate metric (e.g., revenue, volume, count).
- Specify what the ranking is based on.

 Overall:
- Base every answer on factual data from {table_info}.
- Be precise and quantitative: use numbers, comparisons, or summaries.
- Prefer clarity over vagueness.
"""
    ),
    ("human", "{input}")
])

# Chain setup — 
chain = create_sql_query_chain(
    llm=llm,
    db=db,
    prompt=prompt
    
)

import streamlit as st

st.set_page_config(page_title=" AI Sales Assistant", page_icon="📊")
st.title("📊 AI Sales Assistant")
st.caption("Ask anything about your distributor sales data.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about your distributor sales...")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({"input": user_input})
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("ai", response))
        except Exception as e:
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("ai", f"⚠️ Error: {str(e)}"))


for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(msg)

