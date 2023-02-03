import os

from gravity_model import *

""" 
Here, the input parameter for the Hyman Algorithmus can be set
"""
empirical_mean_shopping_distance = 4.65  # taken from Schlaich(2020)
tolerance = 0.01

flow = hyman_model(empirical_mean_shopping_distance, tolerance)

os.makedirs("Outputs/Flow", exist_ok=True)
flow.to_pickle("Outputs/Flow/flow.pkl")
flow.to_csv("Outputs/Flow/flow.csv")

print(flow)
