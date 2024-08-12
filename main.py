# main.py

from simulation import run_simulation
import unittest
import test_simulation

if __name__ == "__main__":
    print("Running tests...")
    unittest.main(module=test_simulation, exit=False)
    
    print("\nRunning simulation...")
    run_simulation()
