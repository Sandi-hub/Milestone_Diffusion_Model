import os

from gravity_model import hyman_model
from PreProcessing.read_census_data import get_population_data
from PreProcessing.read_shops_data import get_shops_data

""" 
Here, the input parameter for the Hyman algorithm can be set
"""

empirical_mean_shopping_distance = (
    4.65  # taken from Schlaich(2020); all units are in km
)
tolerance = 0.01

df_population = get_population_data()
df_Hamburg_shops = get_shops_data()

flow_results = hyman_model(
    empirical_mean_shopping_distance,
    tolerance,
    df_population,
    df_Hamburg_shops,
    istoy=False,
)

flow = flow_results[0]
os.makedirs("Data/Flow", exist_ok=True)
flow.to_pickle("Data/Flow/flow.pkl")

print(flow)
