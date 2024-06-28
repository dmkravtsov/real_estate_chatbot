from app import create_app  # Import the create_app function from the app package.

# Call the create_app function to create and configure an instance of the Flask application.
app = create_app()

# The following code will run if this script is executed directly (not imported as a module).
if __name__ == "__main__":
    # Start the Flask application on port 5000.
    app.run(host='oftu-ml-vm', debug=True, port=9000)
