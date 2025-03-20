#!/bin/bash

echo "Running simulate-and-recover experiment..."
python3 src/simulate.py > results.txt
python3 src/recover.py >> results.txt
echo "Results saved to results.txt"
