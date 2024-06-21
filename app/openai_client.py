import openai  # Importing the OpenAI library for interacting with OpenAI's API.
import os  # Importing the OS library to work with environment variables.
from dotenv import load_dotenv  # Importing the dotenv library to load environment variables from a .env file.
import mlflow  # Importing the MLflow library for logging and tracking machine learning experiments.
from .faq import faq_data, get_faq_response  # Importing FAQ data and response function from the faq module.

# Load environment variables from a .env file to set up configuration.
load_dotenv()

# Set the OpenAI API key from the environment variable.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define customization options available for the T-shirts.
customization_options = {
    "styles": ["Crew Neck", "V-Neck", "Long Sleeve", "Tank Top"],
    "genders": ["Male", "Female", "Unisex"],
    "colors": ["White", "Black", "Blue", "Red", "Green", "Custom Colors"],
    "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
    "printing_options": ["Screen Printing", "Embroidery", "Heat Transfer", "Direct-to-Garment"]
}

def detect_support_request(prompt):
    """
    Detects if the user's prompt is related to a support request.
    Checks for keywords that typically indicate a need for support.
    """
    support_keywords = ["help", "support", "problem", "issue", "trouble", "assistance", "contact", "urgent"]
    for keyword in support_keywords:
        if keyword in prompt.lower():
            return True
    return False

def save_support_request(prompt):
    """
    Saves the support request to a text file for further processing.
    """
    with open("support_requests.txt", "a") as file:
        file.write(f"User Message: {prompt}\n\n")
    return "Support request saved successfully."

def get_customization_option(option_type):
    """
    Retrieves and returns available options for a specified customization type.
    """
    if option_type in customization_options:
        return f"Available {option_type.replace('_', ' ')} are: {', '.join(customization_options[option_type])}."
    else:
        return f"Invalid option type: {option_type}. Please choose from: styles, genders, colors, sizes, or printing options."

def generate_response(prompt):
    """
    Generates a response to the user's prompt using predefined logic and OpenAI's API.
    Logs interactions using MLflow for tracking.
    """
    # Start a new MLflow run for logging.
    mlflow.start_run()

    # Log the user's original prompt.
    mlflow.log_param("user_prompt", prompt)

    try:
        # Check if the prompt matches any frequently asked questions (FAQ).
        for question in faq_data.keys():
            if question.lower() in prompt.lower():
                response = get_faq_response(question)
                mlflow.log_param("response_type", "FAQ")
                mlflow.log_param("matched_question", question)
                mlflow.log_metric("response_length", len(response))
                mlflow.end_run()
                return response

        # Check if the prompt is related to customization options.
        for option in customization_options.keys():
            if option in prompt.lower():
                response = get_customization_option(option)
                mlflow.log_param("response_type", "Customization")
                mlflow.log_param("matched_option", option)
                mlflow.log_metric("response_length", len(response))
                mlflow.end_run()
                return response

        # Check if the prompt indicates a support request.
        if detect_support_request(prompt):
            response = save_support_request(prompt)
            mlflow.log_param("response_type", "Support Request")
            mlflow.log_param("support_request_saved", "yes")
            mlflow.log_metric("response_length", len(response))
            mlflow.end_run()
            return f"It looks like you need support. {response}"

        # Use OpenAI's API to generate a response if no other conditions are met.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        ).choices[0].message['content'].strip()

        # Log the type of response and its length.
        mlflow.log_param("response_type", "OpenAI API")
        mlflow.log_metric("response_length", len(response))
        mlflow.end_run()
        return response
    except openai.error.InvalidRequestError as e:
        # Handle invalid request errors and log them.
        error_message = f"Invalid request error: {str(e)}"
        mlflow.log_param("response_type", "Error")
        mlflow.log_param("error_message", error_message)
        mlflow.end_run()
        return error_message
    except Exception as e:
        # Handle general exceptions and log them.
        error_message = f"An error occurred: {str(e)}"
        mlflow.log_param("response_type", "Error")
        mlflow.log_param("error_message", error_message)
        mlflow.end_run()
        return error_message
