from flask import Blueprint, request, jsonify, render_template  # Import necessary Flask modules.
from .openai_client import generate_response  # Import the function to generate responses from the OpenAI client module.

# Create a Blueprint named 'routes' for organizing the routes of the Flask application.
bp = Blueprint('routes', __name__)

@bp.route('/')
def index():
    """
    This route handles the root URL ('/') of the application.
    When a user visits the root URL, it renders the 'index.html' template.
    """
    return render_template('index.html')

@bp.route('/chat', methods=['POST'])
def chat():
    """
    This route handles POST requests to the '/chat' URL.
    It expects a JSON payload containing a 'message' from the user.
    It generates a response using the OpenAI client and returns it as JSON.
    """
    # Extract the 'message' from the JSON payload of the request.
    user_message = request.json.get('message')

    # If the user message exists, generate a response.
    if user_message:
        bot_response = generate_response(user_message)
        # Return the generated response as a JSON object.
        return jsonify({"response": bot_response})

    # If no message is provided, return an error message as a JSON object.
    return jsonify({"response": "Please provide a message."})
