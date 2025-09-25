# state mgmt output

# streamlit code with continued cha t interface 

import streamlit as st
import pandas as pd
from state_mgmt_model import run_full_pipeline  # Replace with your actual import

# we build the formatting function here

import re

import pandas as pd
import re

import streamlit as st
import pandas as pd
import re

def format_currency_indian(df):
    def indian_comma_format(num):
        try:
            if pd.isnull(num):
                return ""
            s = str(num)
            if '.' in s:
                int_part, dec_part = s.split('.')
            else:
                int_part, dec_part = s, None
            int_part = int_part.replace(',', '')

            # Indian comma grouping: last 3 digits stay together, earlier digits grouped in 2s
            if len(int_part) > 3:
                last3 = int_part[-3:]
                rest = int_part[:-3]

                parts = []
                while len(rest) > 2:
                    parts.insert(0, rest[-2:])
                    rest = rest[:-2]
                if rest:
                    parts.insert(0, rest)

                formatted_int = ','.join(parts) + ',' + last3
            else:
                formatted_int = int_part

            if dec_part:
                return f"{formatted_int}.{dec_part}"
            else:
                return formatted_int
        except:
            return str(num)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(indian_comma_format)
            if any(keyword in col.lower() for keyword in [
                "total_sales",
                "total_secondary_sales",
                "total_order_revenue",
                "revenue",
                "amount",
                "value",
                "price"
            ]):
                df[col] = df[col].apply(lambda x: f"Rs.{x}" if x else x)
    return df





# image from the folder. 
SMART_DOOR_IMAGE_URL = "assets/QUAD-RG-1.jpg"  

# Dark theme CSS with readable chat bubbles
dark_style = """
<style>
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 10px 15px 70px 15px; /* space for fixed input */
    }
    .chat-container {
        max-height: 75vh;
        overflow-y: auto;
        padding: 12px 20px;
        border-radius: 10px;
        background: #1f1f1f;
        border: 1px solid #333;
        box-shadow: 0 0 10px #2a2a2a;
    }
    .chat-message {
        max-width: 80%;
        padding: 12px 18px;
        margin: 8px 0;
        border-radius: 20px;
        font-size: 15px;
        white-space: pre-wrap;
        word-wrap: break-word;
        clear: both;
        display: inline-block;
    }
    .user-msg {
        background: #7c3aed;
        color: white;
        float: right;
        border-radius: 20px 20px 0 20px;
        font-weight: 600;
        text-align: right;
    }
    .bot-msg {
        background: #2f2f4f;
        color: #c1c1e0;
        float: left;
        border-radius: 20px 20px 20px 0;
        text-align: left;
        font-weight: 400;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .dataframe-container {
        clear: both;
        margin: 4px 0 18px 0;
        border-radius: 6px;
        overflow-x: auto;
    }
    div.stTextInput > div > input {
        background-color: #2a2a4f;
        color: #ddd;
        border-radius: 12px;
        border: 1px solid #7c3aed;
        padding: 10px 15px;
        font-size: 16px;
    }
    #smart-door-btn {
        cursor: pointer;
        height: 38px;
        width: 38px;
    }
</style>
"""

st.markdown(dark_style, unsafe_allow_html=True)

st.title("Smart Data Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def append_message(user_text, bot_text, df=None):
    if df is not None and not df.empty:
        df = format_currency_indian(df.copy())
    minimal_bot_text = "Results shown below as table."
    st.session_state.chat_history.append({
        "user": user_text,
        "bot": minimal_bot_text,
        "df": df
    })
    # Persist chat history in local session state (streamlit auto persists during session reload)
    st.experimental_set_query_params(chat_len=len(st.session_state.chat_history))

def display_chat():
    st.markdown('<div class="chat-container" id="chat-scroll">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        # User message
        st.markdown(f'<div class="chat-message user-msg">{chat["user"]}</div>', unsafe_allow_html=True)
        # Bot message + explanation combined inside bubble
        bot_html = chat["bot"].replace("\n", "<br>")
        st.markdown(f'<div class="chat-message bot-msg">{bot_html}</div>', unsafe_allow_html=True)
        # Dataframe below bot message
        if chat["df"] is not None and not chat["df"].empty:
            with st.container():
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(chat["df"].style.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#5a4acb'), ('color', '#f0e9ff')]},
                    {'selector': 'td', 'props': [('background-color', '#3c3576'), ('color', '#dcd6ff')]}
                ]), height=300, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def on_submit():
    user_text = st.session_state.user_input.strip()
    if not user_text:
        return
    with st.spinner("Unlocking smart insights..."):
        response = run_full_pipeline(user_text)  # Your pipeline call
    if "error" in response:
        st.error(response["error"])
        return
    append_message(user_text, response.get("explanation", "No explanation available."), response.get("results"))
    st.session_state.user_input = ""

# Display current chat history
display_chat()

# Input area + smart door image button horizontally aligned
input_col, button_col = st.columns([0.9, 0.1])

with input_col:
    st.text_input(
        "Ask your question here…",
        key="user_input",
        placeholder="Type here and hit Enter or click the smart door",
        on_change=on_submit,
        label_visibility="collapsed"
    )

with button_col:
    # Using image as button for smart door
    if st.button(label="", key="smart_door_button", help="Unlock smart insights"):
        on_submit()

    # Alternatively, show image inside button area (would require separate JS or custom components):
    # st.markdown(f'<img src="{SMART_DOOR_IMAGE_URL}" id="smart-door-btn" alt="Smart Door" />', unsafe_allow_html=True)
