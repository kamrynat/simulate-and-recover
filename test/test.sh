#!/bin/bash

# Ensure the script exits on first error
set -e

# Run the unit tests explicitly specifying the correct path
python3 -m unittest discover -s test -p "test_*.py"

# Notify user of completion
echo "All tests passed successfully."
