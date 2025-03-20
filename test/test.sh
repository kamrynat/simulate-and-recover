#!/bin/bash

# Ensure the script exits on first error
set -e

# Run the unit tests
python3 -m unittest discover -s tests

# Notify user of completion
echo "All tests passed successfully."
