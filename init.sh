#!/bin/bash
EnvironmentDirectory='env'
EnvironmentActivationPath="$EnvironmentDirectory/bin/activate"
if [ -d "$EnvironmentDirectory" ]; then
    echo "Everything is setup activating virtual environment"
	source $EnvironmentActivationPath
else
	echo "Creating python3 virtual environment"
	python_location=$(which python3)
	virtualenv -p $python_location $EnvironmentDirectory

	echo "Activating python3 virtual environment"
	source $EnvironmentActivationPath
	if [[ "$VIRTUAL_ENV" != "" ]]; then
		echo "Installing required packages python3 virtual environment"
		pip install -r requirements.txt
		echo $python_location
	else
  		echo "Virtual environment is required before proceeding, install it by running in the terminal- sudo pip3 install virtualenv"
  		exit 1
	fi

	echo "Setting up folder structure"
	mkdir inputs
	mkdir inputs/train
	mkdir inputs/validation
	mkdir inputs/test
	mkdir outputs
	mkdir checkpoints
fi

