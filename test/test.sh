#Debugged using ZotGPT
#!/bin/bash

# Ensure the script exits on first error
set -e

# Get the directory of this script (which should be the test directory)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the project root directory (one level up from the test directory)
PROJECT_ROOT="$( cd "$DIR/.." && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the tests both ways
echo "Running tests using ./test_simulate.py"
python3 "$DIR/test_simulate.py"

echo -e "\nRunning tests using python3 test_simulate.py"
python3 "$DIR/test_simulate.py"

# Notify user of completion
echo -e "\nAll tests passed successfully."
