# Real Estate Investment Chatbot

This project is a real estate investment chatbot that utilizes GPT-4 and LangChain to analyze data, answer FAQs, and provide insights on property investments. The chatbot is designed to help users with various real estate-related queries by leveraging advanced natural language processing and machine learning techniques.

## Features

- **Data Analysis**: Analyzes real estate data to determine the most profitable investment opportunities.
- **FAQ Handling**: Answers frequently asked questions using a pre-defined FAQ dataset with vector search.
- **Support Requests**: Detects and logs support requests for further assistance.
- **Interactive Chat**: Engages users in a conversational manner to address their queries.
- **Contextual Understanding**: Maintains context across multiple interactions to provide relevant responses.
- **Graph Plotting**: Generates distribution and dependency graphs based on user queries.
- **Robust Answer Generation**: Utilizes an agent and CSV file with data to answer various types of questions.

## Project Structure
<pre>
real_estate_chatbot/
├── app.py                 # Streamlit application
├── openai_client.py       # Handles OpenAI API interactions and response generation
├── faq.py                 # Contains FAQ data and response logic
├── venv/                  # Virtual environment for project dependencies
├── .env                   # Environment variables file (contains OpenAI API key)
├── requirements.txt       # Project dependencies
├── mlflow_test.py         # Script for testing MLflow logging
</pre>

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/dmkravtsov/real_estate_chatbot.git
    cd real_estate_chatbot
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

5. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

## Usage

- Access the chatbot interface through the Streamlit web application.
- Enter your queries related to real estate investments, and the chatbot will provide relevant answers and insights based on the data and FAQ information.
- The bot can plot distribution and dependency graphs, answer questions from `faq.py` using vector search, and handle other types of queries using an agent and a CSV file with data.

## Screenshot

    ```markdown
    ![Chatbot Interface](screenshot.png)
    ```

