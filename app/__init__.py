from flask import Flask  # Import the Flask class from the Flask framework.

def create_app():
    """
    Creates and configures an instance of the Flask application.
    This function sets up the application, loads configuration,
    and registers routes (endpoints) from a Blueprint.
    """
    # Create an instance of the Flask class.
    app = Flask(__name__)

    # Load configuration settings from a Python file.
    # This file is located one directory level up from the current file.
    app.config.from_pyfile('../config.py')

    # Import the Blueprint named 'routes_bp' from the routes module.
    from .routes import bp as routes_bp

    # Register the Blueprint with the Flask application.
    # This makes the routes defined in the Blueprint available in the application.
    app.register_blueprint(routes_bp)

    # Return the configured Flask application instance.
    return app
