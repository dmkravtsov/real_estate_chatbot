import sys
import os

# Добавьте корневую директорию в sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from openai_client import handle_user_query, load_and_preprocess_data

# Заголовок вашего приложения
st.title('Real Estate Investment Chatbot')

# Загрузка данных и отображение первых строк
file_path = 'df.csv'
df, _, _, _, _, _, _ = load_and_preprocess_data(file_path)
st.write("## Real Estate Data")
st.write(df.head())

# Инициализация или загрузка истории чата из сессии
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Отображение истории чата
st.write("## Chat History")
for chat in st.session_state['chat_history']:
    if chat['role'] == 'user':
        st.write(f"**User:** {chat['content']}")
    else:
        st.write(f"**Bot:** {chat['content']}")

# Форма для ввода пользовательского сообщения
user_message = st.text_input("Enter your message:")

if st.button("Send"):
    if user_message:
        # Обработка запроса пользователя
        response = handle_user_query(user_message)
        
        # Добавление сообщения пользователя и ответа бота в историю чата
        st.session_state['chat_history'].append({"role": "user", "content": user_message})
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        
        # Очистка поля ввода
        st.session_state.user_message = ""
