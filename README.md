
# Flask OpenAI Chatbot

## Overview

This project implements a conversational chatbot using Flask and OpenAI. The chatbot is designed for a hypothetical platform, TeeCustomizer, where users can design and order customizable t-shirts. The chatbot assists users with selecting styles, colors, sizes, and printing options, answers frequently asked questions (FAQs), and logs support requests. Additionally, the project includes basic logging of progress using MLflow.

## Features

- Guides users through the process of ordering a customizable t-shirt.
- Provides answers to frequently asked questions.
- Logs support requests based on user interactions.
- Integrates with OpenAI for generating responses.
- Uses MLflow for logging the chatbot's responses and parameters.

## Project Structure
<pre>
flask_openai_chatbot/
│
├── app/
│ ├── init.py # Initializes the Flask app and registers routes
│ ├── routes.py # Defines the routes for the web application
│ ├── openai_client.py # Handles OpenAI API interactions and response generation
│ ├── faq.py # Contains FAQ data and response logic
│ ├── static/ # Static files (e.g., CSS, JS)
│ └── templates/
│ └── index.html # HTML template for the chatbot interface
│
├── venv/ # Virtual environment for project dependencies
├── config.py # Configuration file for Flask application
├── .env # Environment variables file (contains OpenAI API key)
├── requirements.txt # Project dependencies
├── run.py # Entry point for running the Flask application
└── mlflow_test.py # Script for testing MLflow logging
</pre>

## Setup Instructions

### Prerequisites

- Python 3.7 or later
- Virtual environment (venv)
- OpenAI API key

### Steps to Setup and Run the Project

1. **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd flask_openai_chatbot
    ```

2. **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**
    - Create a `.env` file in the root directory with the following content:
      ```
      OPENAI_API_KEY=<your_openai_api_key>
      ```

5. **Run the Application:**
    ```bash
    python run.py
    ```

6. **Access the Application:**
    - Open a browser and go to `http://127.0.0.1:5000`.

7. **Run MLflow UI (Optional):**
    ```bash
    mlflow ui --port 5001
    ```
    - Access MLflow UI at `http://127.0.0.1:5001` to view logs and metrics.

## Detailed Description of the Components

### `app/__init__.py`

This file initializes the Flask application and registers the routes defined in the `routes.py` file. It loads configuration settings from `config.py`.

### `app/routes.py`

Defines the routes for the application:
- `/`: Renders the main HTML page.
- `/chat`: Handles POST requests with user messages and returns chatbot responses.

### `app/openai_client.py`

Handles interactions with the OpenAI API and generates responses:
- `generate_response(prompt)`: Generates responses based on user input.
- `get_customization_option(option_type)`: Provides details about t-shirt customization options.
- `detect_support_request(prompt)`: Checks if a user request needs support assistance.
- `save_support_request(prompt)`: Saves support requests to a file.

### `app/faq.py`

Contains FAQ data and logic to provide responses to frequently asked questions:
- `faq_data`: Dictionary containing questions and answers.
- `get_faq_response(question)`: Returns the answer to a frequently asked question.

### `config.py`

Holds configuration settings for the Flask application. Typically includes settings like debug mode, secret keys, and database configurations.

### `mlflow_test.py`

A script to test basic logging capabilities with MLflow. It demonstrates logging of parameters and metrics to track progress and issues in the chatbot's responses.

### `run.py`

The entry point for running the Flask application. It initializes the application by calling `create_app()` and starts the server on port 5000.

## Logging with MLflow

MLflow is integrated to log chatbot interactions, parameters, and metrics:
- `mlflow.start_run()`: Starts a new logging run.
- `mlflow.log_param("param_name", value)`: Logs a parameter.
- `mlflow.log_metric("metric_name", value)`: Logs a metric.
- `mlflow.end_run()`: Ends the current logging run.

The logging captures:
- User prompts.
- Response types (FAQ, customization, support, OpenAI API).
- Length of responses.
- Errors, if any.

MLflow UI can be used to view and analyze the logged data, helping in understanding the chatbot's performance and areas for improvement.

## Next Steps

- **Extend Language Support**: Implement multilingual support using language detection and translation services.
- **Enhance Customization**: Provide more detailed customization options for t-shirts.
- **Advanced Logging**: Integrate more detailed logging for better analysis and debugging.
- **Deployment**: Deploy the chatbot to a cloud platform for broader access.

## References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
