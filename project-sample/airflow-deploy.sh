#!/bin/bash

# Define the directories
SOURCE=("data_pipeline" "model_pipeline")
# VENV=("data_pipeline")
AIRFLOW_HOME="$HOME/airflow"

# Check if Airflow home exists
if [ ! -d "$AIRFLOW_HOME" ]; then
	echo "Airflow home directory not found: $AIRFLOW_HOME"
	exit 1
fi

# Deploy dependent modules
for MODULE in "${SOURCE[@]}"; do
	if [ -d "$MODULE" ]; then
		echo "Deploying module $MODULE to $AIRFLOW_HOME/dags"
		cp -r "$MODULE" "$AIRFLOW_HOME/dags/"
	else
		echo "Module directory not found: $MODULE"
		exit 1
	fi
done

echo "Deployment completed successfully."

# Check if the shell is being executed from a virtual environment
# Function to check if a virtual environment is activated
check_virtualenv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "No virtual environment is activated. You must activate the Airflow virtual environment first."
    else
        echo "Virtual environment is activated: $VIRTUAL_ENV"
		echo "Installing dependencies..."
		# Install dependencies
		pip install -r requirements.txt
    fi
}

# Call the function
check_virtualenv