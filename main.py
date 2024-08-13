# main.py

from simulation import run_simulation
import unittest
import test_simulation
import traceback
from config import CONFIG

if __name__ == "__main__":
    print("Running tests...")
    unittest.main(module=test_simulation, exit=False)
    
    print("\nRunning simulation...")
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    
    try:
        run_simulation()
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        print("Traceback:")
        traceback.print_exc()