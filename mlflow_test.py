import mlflow  # Importing the MLflow library for logging and tracking machine learning experiments.

def run_test():
    # Start a new MLflow run. This ensures that all the parameters and metrics
    # logged in this context are associated with this particular run.
    with mlflow.start_run():
        # Log a parameter named "param1" with a value of 5. Parameters in MLflow
        # are typically static values that describe your experiment or model.
        mlflow.log_param("param1", 5)

        # Log a metric named "metric1" with a value of 0.85. Metrics are values
        # that measure performance or other characteristics and can change
        # throughout the course of an experiment.
        mlflow.log_metric("metric1", 0.85)

        # Print a confirmation message indicating that the test run has been
        # completed and logged successfully.
        print("Test run completed and logged.")

# The following block checks if this script is being run directly (as opposed
# to being imported as a module), and if so, it calls the run_test function.
if __name__ == "__main__":
    run_test()
