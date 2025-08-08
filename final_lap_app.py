# this is the version of llm chaining where instead of single shot prompt, we aim at prompt chaining
# concepts 'chained' together: prompt chaining + whitelisting + metadata + guardrailing + logging 
# we go step by step and get the desired output - fingers crossed

from langchain.prompts import PromptTemplate

# Step 1: Define prompt template with placeholders and strict instructions
SQL_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
You are an expert SQL generator for Neon Postgres.
You may ONLY use this table and these columns:
- bm_full_sales(date_id, year, month, brand, product, sub_product, distributor_name, city, total_quantity, total_sales)
Never use any other table or column.
Never invent, repeat, or hallucinate columns or values.
Output ONLY the SQL query in a markdown code block (``````).
User question: {query}
"""
)


# we define a logging and validation function
# this validation function takes in 3 parameters 
# step - a string label for the validation step
# condition - a boolean check
# success and fail messages to log accordingly


def log_and_validate(step, condition, success_msg, fail_msg):
    if condition:
        print(f"{step}: {success_msg}")
        return True
    else:
        print(f"{step}: {fail_msg}")
        return False

def main():
    # Hardcoded test query for step 1
    user_question = "total sales by city"

    # Build the final prompt with user question injected
    final_prompt = SQL_PROMPT.format(query=user_question)

    # Log and validate prompt injection
    log_and_validate(
        "Prompt Injection",
        user_question in final_prompt,
        "User question successfully injected into prompt.",
        "User question MISSING from prompt!"
    )

    print("\n--- Prompt sent to LLM ---\n")
    print(final_prompt)
    print("\n--- End prompt ---")

if __name__ == "__main__":
  main()


# step 2 : sending this prompt to llm

import os
import openai
import dotenv
import psycopg2
import pandas as pd
import re
from openai import OpenAI

dotenv.load_dotenv()  # Loads your .env file

# Retrieve API keys and DB URL
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

if not OPENAI_API_KEY or not DATABASE_URL:
    raise Exception("OPENAI_API_KEY or DATABASE_URL missing in .env file!")

# Initialize OpenAI client for GPT-3.5-turbo (LangChain no longer supports ChatOpenAI directly)
llm_client = OpenAI(api_key=OPENAI_API_KEY)

import os
import re
import dotenv
import pandas as pd
import psycopg2
from langchain.prompts import PromptTemplate
from openai import OpenAI
from openai import OpenAIError

# Load .env variables
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

if not OPENAI_API_KEY or not DATABASE_URL:
    raise RuntimeError("OPENAI_API_KEY or DATABASE_URL missing. Check your .env")

# Initialize OpenAI client for GPT-3.5-turbo (LangChain no longer supports ChatOpenAI directly)
llm_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Step 1: Load Metadata dynamically ---
def load_metadata(filepath='Metadata.xlsx', sheet_name='Technical_Metadata'):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df

def get_columns_for_table(metadata_df, table_name):
    cols = metadata_df[metadata_df["Table Name"] == table_name]["Column Name"].tolist()
    return cols

# --- Step 2: Intent & Mart Detection (Optional but good for scaling multiple marts) ---
# For simplicity, assuming single mart 'bm_full_sales' for now
def detect_mart(user_question):
    # You can expand this with LLM prompt or keyword mapping.
    # For demo, we return bm_full_sales always
    return "bm_full_sales"

# --- Step 3: Build Dynamic SQL Prompt Template ---
def build_sql_prompt(mart_name, allowed_columns, user_question):
    cols_str = ", ".join(allowed_columns)
    prompt_text = f"""
You are an expert SQL generator for Neon Postgres.
Use ONLY this table and columns:
- {mart_name}({cols_str})

Never use any other table or column. Never hallucinate.
Return ONLY the SQL code in a markdown code block using triple backticks (```
User question: {user_question}
"""
    return prompt_text

# --- Step 4: Call OpenAI ChatCompletion ---
def call_openai_chat(prompt, temperature=0):
    messages = [
        {"role": "system", "content": "You are a reliable, precise SQL generator."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None

# --- Step 5: Extract SQL code from triple backticks in LLM response ---
def extract_sql(llm_response):
    if not llm_response:
        return None
    pattern = r"```(?:sql)?\s*([\s\S]*?)```"
    match = re.search(pattern, llm_response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # fallback: return raw if no backticks found
        return llm_response.strip()

# --- Step 6: Validate SQL columns against whitelist ---
def validate_sql(sql, allowed_columns, allowed_tables=None):
    if allowed_tables is None:
        allowed_tables=[]
    
    # Extract all column-like words from SQL
    col_candidates = set(re.findall(r"\b[a-zA-Z_]+\b", sql.lower()))
    allowed_set = set([col.lower() for col in allowed_columns])
    allowed_tables_set = set(t.lower() for t in allowed_tables)
    # Ignoring SQL keywords etc is crude but simple here:
    sql_keywords = {
        "select", "from", "where", "group", "by", "order", "asc", "desc", 
        "sum", "count", "avg", "min", "max", "as", "and", "or", "on", "join",
        "inner", "left", "right", "full", "outer", "limit", "offset", "not",
        "in", "exists", "case", "when", "then", "else", "end"
    }
    
    unknown_cols = col_candidates - allowed_set - allowed_tables_set - sql_keywords
    
    if unknown_cols:
        print(f"SQL Validation Warning: Unknown columns detected: {unknown_cols}")
        return False
    return True

# --- Step 7: Execute SQL and get result as dataframe ---
def execute_sql(sql):
    try:
        import psycopg2.extras
        with psycopg2.connect(DATABASE_URL) as conn:
            df = pd.read_sql_query(sql, conn)
            return df
    except Exception as e:
        print(f"Database error: {e}")
        return None

# --- Step 8: Generate simple business explanation ---
def generate_user_friendly_explanation(user_question, df):
    if df is None or df.empty:
        return "No data found to answer the question, or database query failed."

    # Limit to top 10 rows for explanation to avoid LLM token bloat
    table_md = df.head(10).to_markdown(index=False)

    # Build explanation prompt
    explanation_prompt = f"""
You are a business analyst who explains data clearly in plain English for non-technical users.

Here is the user's question:
{user_question}

Here is the data from the database:
{table_md}

Please provide a clear, concise explanation answering the user's question in plain language.
"""
    explanation = call_openai_chat(explanation_prompt, temperature=0)
    return explanation

# --- Main orchestrator function ---
def run_full_pipeline(user_question):
    print("\n Loading metadata and detecting mart...")
    metadata_df = load_metadata()
    mart = detect_mart(user_question)
    print(f"Detected mart: {mart}")

    allowed_columns = get_columns_for_table(metadata_df, mart)
    print(f"Allowed columns for {mart}: {allowed_columns}")

    print("\n Building dynamic prompt for SQL generation...")
    sql_prompt = build_sql_prompt(mart, allowed_columns, user_question)

    print("\n Calling LLM to generate SQL query...")
    llm_response = call_openai_chat(sql_prompt)

    if llm_response is None:
        print("LLM SQL generation failed.")
        return

    print("\n Extracting SQL from LLM response...")
    sql_query = extract_sql(llm_response)
    if not sql_query:
        print("Failed to extract SQL from LLM response.")
        return
    print(f"Extracted SQL:\n{sql_query}")

    print("\n Validating SQL columns against whitelist...")
    if not validate_sql(sql_query, allowed_columns, allowed_tables=[mart]):
        print("SQL validation failed! Disallowed columns detected. Aborting execution.")
        return
    print("SQL validation passed.")

    print("\n Executing SQL against database...")
    results_df = execute_sql(sql_query)
    if results_df is None:
        print("SQL execution failed or returned no data.")
        return
    print(f"SQL execution successful. Rows returned: {len(results_df)}")

    print("\n Generating user-friendly explanation...")
    explanation = generate_user_friendly_explanation(user_question, results_df)
    print("\n--- Explanation for User ---\n")
    print(explanation)

    # Return all relevant info could be useful for UI or logging
    return {
        "mart": mart,
        "allowed_columns": allowed_columns,
        "sql_prompt": sql_prompt,
        "llm_sql_response": llm_response,
        "extracted_sql": sql_query,
        "validation_passed": True,
        "results": results_df,
        "explanation": explanation
    }


if __name__ == "__main__":
    # Example: dynamic user question input, no hardcoding here
    user_question_input = input("Enter your business question: ").strip()
    if not user_question_input:
        print("Empty question provided. Please try again.")
    else:
        run_full_pipeline(user_question_input)


# step 3 : generate user friendly answer

def generate_user_friendly_explanation(user_question, df):
    if df is None or df.empty:
        return "No data found to answer the question, or query failed."

    # Convert the full DataFrame to markdown for feeding entire results to LLM
    table_md = df.to_markdown(index=False)

    explanation_prompt = f"""
You are a business analyst who explains data clearly in plain English for non-technical users.

User question:
{user_question}

Here is the full data returned from the database:
{table_md}

Please provide a clear, concise explanation answering the user's question in simple language.
"""

    explanation = call_openai_chat(explanation_prompt, temperature=0)
    return explanation



# step 4 : we scaffold streamlit app...fingers crossed hard

import streamlit as st
import pandas as pd

# Assuming all your backend functions (incl. run_full_pipeline) are above or imported
# If your big backend code is in the same file -- perfect.
# If it's in another module, replace the import here to import it.

# === Streamlit dark theme CSS ===
dark_style = """
<style>
    .stApp {
        background-color: #000000;
        color: #bbbbbb;
    }
    .user-msg {
        color: #ffffff;
        text-align: right;
        font-weight: bold;
        padding-right: 10px;
        padding-top: 5px;
    }
    .system-msg {
        color: #bbbbbb;
        text-align: left;
        padding-left: 10px;
        padding-top: 5px;
        white-space: pre-wrap; /* preserve new lines */
    }
    .dataframe th {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    .dataframe td {
        background-color: #111111 !important;
        color: #ffffff !important;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #ff3333 #222222;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #444;
        margin-bottom: 10px;
    }
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: #222222;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background-color: #ff3333;
        border-radius: 4px;
    }
</style>
"""
st.markdown(dark_style, unsafe_allow_html=True)

st.title("Unlock Data Insights Smartly")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def append_message(user_input, bot_response, df_results):
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": bot_response,
        "df": df_results
    })

def display_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        user_text = chat["user"]
        bot_text = chat["bot"]
        df = chat["df"]

        # Display user query
        st.markdown(f'<div class="user-msg">You: {user_text}</div>', unsafe_allow_html=True)

        # Display AI explanation
        st.markdown(f'<div class="system-msg">{bot_text}</div>', unsafe_allow_html=True)

        # Display full query results DataFrame in styled dark mode if available
        if df is not None and not df.empty:
            st.dataframe(
                df.style.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#333'), ('color', '#fff')]},
                    {'selector': 'td', 'props': [('background-color', '#111'), ('color', '#fff')]}
                ]),
                height=400,
                use_container_width=True
            )
    st.markdown('</div>', unsafe_allow_html=True)


def on_submit():
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return

    # Run your real pipeline here, producing explanation and results DataFrame:
    pipeline_output = run_full_pipeline(user_input)

    explanation = pipeline_output.get("explanation", "No explanation available.")
    results_df = pipeline_output.get("results", pd.DataFrame())

    append_message(user_input, explanation, results_df)
    st.session_state.user_input = ""

# Text input for continuous chat with auto submit on Enter
st.text_input(
    "Ask your business question:",
    key="user_input",
    placeholder="Type your question here and press Enter...",
    on_change=on_submit
)

# Show chat history on page (scrollable container)
display_chat()

