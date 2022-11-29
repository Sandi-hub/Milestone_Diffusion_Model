import os
from gravity_model import *

empirical_mean_shopping_distance = 4.65  # the empirical mean shopping distance is taken from Schlaich(2020) and needed to compare the modeled mean distance
tolerance = 0.1  # TODO: Change 0.01

flow = hyman_model(empirical_mean_shopping_distance, tolerance)

flow.to_pickle("Output_Flow/flow.pkl")

os.makedirs("Output_Flow", exist_ok=True)
flow.to_csv("Output_Flow/flow.csv")

print(flow)
