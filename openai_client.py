import openai
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import mlflow
from faq import faq_data, get_faq_response  # Абсолютный импорт
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import tempfile



# Загрузка переменных окружения
load_dotenv()

# Установка ключа OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Загрузка и предобработка данных по недвижимости
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    rename_dict = {
        'Lat': 'Latitude',
        'Lon': 'Longitude',
        'Beds': 'Bedrooms',
        'Street name': 'street_name'
    }
    df.rename(columns=rename_dict, inplace=True)
    df['Has pool'] = df['Pool'].notna()
    df['Has garage'] = df['Garage'].notna()
    df['Age'] = datetime.now().year - df['Year Built'].fillna(datetime.now().year)
    categorial = ['Property ID', 'Address', 'Status', 'Street number', 'street_name', 'Unit', 'City', 'Zip', 'Neighborhood', 'Area', 'MSA', 'County', 'State', 'FIPS Code', 'APN', 'Property Type', 'Style(s)', 'Pool', 'Heating', 'Cooling', 'Subdivision', 'Listing office name', 'Agent name', 'Agent phone', 'Agent email', 'MLS', 'FEMA zone', 'Noise / Airport', 'Noise / Traffic', 'Noise / Local', 'Noise / Score', 'Listing URL', 'Virtual tour', 'Description', 'Image URL', 'Listing ID', 'Owner 1 Email 1', 'Owner 1 Email 2', 'Owner 1 Phone Numbers', 'Owner 2 Name', 'Owner 2 Email 1', 'Owner 2 Email 2', 'Owner 2 Phone Numbers', 'Owner 3 Name', 'Owner 3 Email 1', 'Owner 3 Email 2', 'Owner 3 Phone Numbers', 'Owner 4 Name', 'Owner 4 Email 1', 'Owner 4 Email 2', 'Owner 4 Phone Numbers']
    for col in categorial:
        df[col] = df[col].fillna('No data available').astype(str)
    numeric = ['Price', 'Stories', 'Age', 'Latitude', 'Longitude', 'Year Built', 'Bedrooms', 'Baths - full', 'Baths - half', 'Building area', 'Lot area', 'Garage', 'Price excludes land', 'List price per square foot', 'Days on market', 'Price reduction', 'Price reduction percentage', 'Value estimate', 'Value low', 'Value high', '6 month forecast', '12 month forecast', 'Rent estimate', 'Tax year', 'Tax amount', 'Property insurance rate', 'Property insurance estimate (annual)', 'Downpayment', 'Mortgage interest rate', 'Monthly mortgage payment (est.)', 'HOA fee', 'Gross yield', 'Cap rate', 'Fixed expenses per month', 'Variable expenses per month', 'Fixed closing costs', 'Variable closing costs', 'NOI (monthly)', 'Annual pre-tax cash flow', 'CoC', 'Last sold amount', 'Time since last sale (y)', 'Change since last sold', 'Change since last sold - %', 'Average annual change since last sold', 'Flood - FEMA factor score', 'Median age', 'Male percentage', 'Female percentage', 'Married percentage', 'Divorced percentage', 'Never married percentage', 'Widowed percentage', 'Average family size', 'Home ownership percentage', 'Unemployment rate', 'Race percentage: White', 'Race percentage: Black', 'Race percentage: Asian', 'Race percentage: Native or Alaska', 'Race percentage: Pacific', 'Race percentage: Other', 'Race percentage: Multiple', 'Ethnicity percentage: Hispanic', 'Residents aged 0-9 percentage', 'Residents aged 10-19 percentage', 'Residents aged 20-29 percentage', 'Residents aged 30-39 percentage', 'Residents aged 40-49 percentage', 'Residents aged 50-59 percentage', 'Residents aged 60-69 percentage', 'Residents aged 70-79 percentage', 'Residents aged over 80 percentage', 'Families with dual income percentage', 'Households with income under $5k', 'Households with income $5k-$10k', 'Households with income $10k-$15k', 'Households with income $15k-$20k', 'Households with income $20k-$25k', 'Households with income $25k-$35k', 'Households with income $35k-$50k', 'Households with income $50k-$75k', 'Households with income $75k-$100k', 'Households with income $100k-$150k', 'Households with income over $150k', 'Median household income', 'Median individual income', 'Rent burden', 'Less than high school percentage', 'High school percentage', 'Some college percentage', 'Batchelor percentage', 'Graduate degree percentage', 'Self-employed percentage', 'Median commute time', 'Uninsured percentage']
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    dates = ['Date listed', 'Last sold date']
    df['Date listed'] = pd.to_datetime(df['Date listed'])
    df['Last sold date'] = pd.to_datetime(df['Last sold date'])
    bool_cols = ['Has pool', 'Has garage', 'New construction', 'Foreclosure', 'Coming soon', 'Pending', 'Short sale', 'Senior community', 'New Listing']
    for col in bool_cols:
        df[col] = df[col].fillna(False)
    column_names = df.columns
    column_types = df.dtypes
    dummies = pd.get_dummies(df['Property Type'])
    df = pd.concat([df, dummies], axis=1)
    df.drop(['Baths - full', 'Baths - half'], axis=1, inplace=True)

    return df, column_names, column_types, numeric, categorial, dates, bool_cols

# Загрузка данных из CSV
file_path = "df.csv"
df, column_names, column_types, numeric, categorial, dates, bool_cols = load_and_preprocess_data(file_path)
unique_property_types_string = ", ".join(df['Property Type'].unique().tolist())


# Инициализация истории чата для поддержания контекста
chat_history = []

# Инициализация TF-IDF Vectorizer и создание векторных представлений для вопросов из FAQ
questions = list(faq_data.keys())
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

def detect_support_request(prompt):
    support_keywords = ["help", "support", "problem", "issue", "trouble", "assistance", "contact", "urgent"]
    if any(keyword in prompt.lower() for keyword in support_keywords):
        return True
    return False

def save_support_request(prompt):
    with open("support_requests.txt", "a") as file:
        file.write(f"User Message: {prompt}\n\n")
    return "Support request saved successfully."

def vector_search_faq(prompt):
    prompt_vector = vectorizer.transform([prompt])
    cosine_similarities = cosine_similarity(prompt_vector, question_vectors).flatten()
    most_similar_question_index = np.argmax(cosine_similarities)
    if cosine_similarities[most_similar_question_index] > 0.7:
        return questions[most_similar_question_index]
    return None

# Настройка агента LangChain для работы с CSV
agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-4-turbo-preview"),
    file_path,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True
)



def handle_user_query(user_query):
    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.start_run()
    mlflow.log_param("user_prompt", user_query)

    plot_path = None  # Инициализация переменной для хранения пути к графику

    try:
        if matched_question := vector_search_faq(user_query):
            response = get_faq_response(matched_question)
            mlflow.log_params({"response_type": "FAQ", "matched_question": matched_question})
            mlflow.end_run()
            return response, plot_path

        if detect_support_request(user_query):
            response = save_support_request(user_query)
            mlflow.log_params({"response_type": "Support Request", "support_request_saved": "yes"})
            mlflow.end_run()
            return f"It looks like you need support. {response}", plot_path

        # Использование агента LangChain для обработки запроса
        prompt = """
        Give the answer in the language in which the user asks the questions.
        The following analysis is based on real estate investment data.
        This includes factors like:
         - Net Operating Income (NOI)
         - Capitalization Rate (Cap rate)
         - Cash on Cash Return (CoC)
         - Homeowners' Association fee (HOA fee).
        We analyze various properties to determine the most profitable investment opportunities.
        To find and assess profitability, focus on NOI, Cap rate and CoC and compare it.
        Find unique property types before filtering if the request concerns the type of property.
        Find unique property styles before filtering if the request concerns the style of property.
        If you can't find an answer - analyze Description by its content carefully in several steps step by step with sum() or any() on first step or paraphrase request.
        If you can't find a result in the neighborhood column - try it in Description.
        Column 'Median age' means age of citizens. Calculate the age of properties/houses by: the current year minus the year built.
        Feature df['Noise / Airport'] means noise level due to the airport, but you can check proximity to the airport in Description also.
        If you are asked to find or show any objects: after receiving the data, it is not enough to say how many objects you found; you need to display selective 5 objects: ID and address.
        If you are asked to plot or build a graph of something, generate the Python code in triple quotes to create the plot using matplotlib without any explanatory text.
        Answer a user request for the following and explain your answer:
        """

        response = agent.run(f"{prompt} {user_query}")
        mlflow.log_params({"response_type": "Agent", "response_length": len(response)})

        # Если ответ содержит код для графика, выполняем его и сохраняем график
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
            try:
                exec(code, globals())
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    plt.savefig(tmpfile.name)
                    plot_path = tmpfile.name
                plt.close()
            except Exception as e:
                response += f"\n\nError generating plot: {str(e)}"

        mlflow.end_run()
        return response.split("```")[0], plot_path  # Убираем отображение кода
    except openai.error.InvalidRequestError as e:
        error_message = f"Invalid request error: {str(e)}"
        mlflow.log_params({"response_type": "Error", "error_message": error_message})
        mlflow.end_run()
        return error_message, plot_path
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        mlflow.log_params({"response_type": "Error", "error_message": error_message})
        mlflow.end_run()
        return error_message, plot_path
