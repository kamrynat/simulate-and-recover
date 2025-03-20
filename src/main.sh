#!/bin/bash

# Ensure the script exits on error
set -e

# Run the simulation script
python3 src/simulate_and_recover.py

# Notify user of completion
echo "Simulation complete. Results saved in results.txt."
