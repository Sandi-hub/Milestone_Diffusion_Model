import os
import random

import pandas as pd

from gravity_model import *
from monte_carlo_simulation import generate_outbreak

##### Definition of input Data #####
# For each chain (besides other) we want to produce outbreaks for the sizes
list_outbreak_scenario_sizes = [10, 15, 20]

# As we want to make the artificial outbreaks reproducible, we set the seed for the generation of random numbers
random.seed(3)


def get_xy(outbreak_scenario):
    df = pd.DataFrame({"Gitter_ID": outbreak_scenario})
    population_data = pd.read_pickle("Outputs/Population/population.pkl")
    df = df.merge(
        population_data[["x_centroid", "y_centroid"]], on="Gitter_ID", how="left"
    )
    return df


sales_per_cell = pd.read_pickle("Outputs/Retailer/stores.pkl")
n_of_chains = sales_per_cell["Chain"].nunique()

# Number of stores per chain
chains = sales_per_cell.groupby(["Chain"])["Chain"].agg("count")


def create_shapefile(path, outbreak_scenario, outbreak_filename):
    path = path + "/Shapefiles/"
    crs = "epsg:3035"
    gdf = gpd.GeoDataFrame(
        outbreak_scenario["Gitter_ID"],
        geometry=gpd.points_from_xy(
            outbreak_scenario["x_centroid"], outbreak_scenario["y_centroid"], crs=crs
        ),
    )
    os.makedirs(
        path,
        exist_ok=True,
    )
    gdf.to_file(path + outbreak_filename + ".shp")


def create_pkl(path, outbreak_scenario, outbreak_filename):
    path = path + "/pkl/"

    os.makedirs(path, exist_ok=True)

    outbreak_scenario.to_pickle(path + outbreak_filename + ".pkl")


# For each chain
for chain in chains.index:
    # store.index = Name of store
    # store = # of stores per chain
    if chain == "Other":
        continue
    if chain == "Netto":
        for no_of_outbreak_cases in list_outbreak_scenario_sizes:
            # Set the number of simulations, If only 1 store, we only simulate one outbreak
            # if chains[chain] == 1:
            #     no_of_trials_per_scenario = 1
            # else:
            #     no_of_trials_per_scenario = 5 + round(
            #         (no_of_outbreak_cases * chains[chain]) / 5
            #     )
            no_of_trials_per_scenario = 1

            for trial in range(0, no_of_trials_per_scenario):
                outbreak_filename = (
                    chain + "_" + str(no_of_outbreak_cases) + "_" + str(trial)
                )
                path = (
                    "Outputs/Outbreaks/"
                    + chain
                    + "/"
                    + str(no_of_outbreak_cases)
                    + "/"
                    + str(trial)
                )

                outbreak_scenario_cells = generate_outbreak(
                    chain, no_of_outbreak_cases, sales_per_cell
                )

                outbreak_scenario = get_xy(outbreak_scenario_cells)

                create_shapefile(path, outbreak_scenario, outbreak_filename)

                create_pkl(path, outbreak_scenario, outbreak_filename)
