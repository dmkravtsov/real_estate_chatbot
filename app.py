import streamlit as st
import pandas as pd
from openai_client import handle_user_query, load_and_preprocess_data

# Заголовок вашего приложения
st.title('Real Estate Investment Chatbot')

# Sidebar with bot information
with st.sidebar:
    st.write("## Bot Description")
    st.write("""
    This bot analyzes real estate investment data and answers questions about:
    - Property profitability (NOI, Cap rate, CoC)
    - Homeowners' Association fees (HOA fee)
    - Price distribution and other factors.

    The bot supports multiple languages.
    """)
    st.write("### Example Questions:")
    st.write("- Show me the most profitable properties")
    st.write("- What are the house prices in area X?")
    st.write("- Plot the price distribution of houses")


# Загрузка и предобработка данных
file_path = 'df.csv'
df, _, _, _, _, _, _ = load_and_preprocess_data(file_path)

# Инициализация состояния сессии
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "plot_data" not in st.session_state:
    st.session_state["plot_data"] = []

# Отображение истории чата
if st.session_state["chat_history"]:
    st.write("## Chat History")
    for i, message in enumerate(st.session_state["chat_history"]):
        role = "User" if message["role"] == "user" else "Bot"
        st.write(f"**{role}:** {message['content']}")
        if role == "Bot" and len(st.session_state["plot_data"]) > i // 2:
            st.image(st.session_state["plot_data"][i // 2])

# Форма для ввода пользовательского сообщения внизу страницы
user_message = st.text_input("Enter your message:", key="user_message_input_new")

if st.button("Send") and user_message:
    response, plot_path = handle_user_query(user_message)
    st.session_state["chat_history"].append({"role": "user", "content": user_message})
    st.session_state["chat_history"].append({"role": "assistant", "content": response})
    if plot_path:
        st.session_state["plot_data"].append(plot_path)
    st.experimental_rerun()  # Перезагрузка страницы для обновления контента
