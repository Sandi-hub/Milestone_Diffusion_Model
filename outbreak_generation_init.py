import os
import random

import geopandas as gpd
import pandas as pd

from monte_carlo_simulation import create_outbreak_scenario
from PreProcessing.read_census_data import get_population_data
from PreProcessing.read_shops_data import get_shops_data

# As we want to make the artificial outbreaks reproducible, we set the seed for the generation of random numbers
random.seed(123)


def get_xy(outbreak_scenario):
    df = pd.DataFrame({"cell_id": outbreak_scenario})
    population_data = get_population_data()
    df = df.merge(
        population_data[["x_centroid", "y_centroid"]], on="cell_id", how="left"
    )
    return df


def create_shapefile(path, outbreak_scenario, outbreak_filename):
    path = path + "/Shapefiles/"
    crs = "epsg:3035"
    gdf = gpd.GeoDataFrame(
        outbreak_scenario["cell_id"],
        geometry=gpd.points_from_xy(
            outbreak_scenario["x_centroid"] * 1000,
            outbreak_scenario["y_centroid"] * 1000,
            crs=crs,
        ),
    )
    os.makedirs(
        path, exist_ok=True,
    )
    gdf.to_file(path + outbreak_filename + ".shp")


def create_pkl(path, outbreak_scenario, outbreak_filename):
    path = path + "/pkl/"

    os.makedirs(path, exist_ok=True)

    outbreak_scenario.to_pickle(path + outbreak_filename + ".pkl")


def generate_outbreak(chain_name, no_of_cases, trial):
    no_of_cases = int(no_of_cases)

    all_shops = get_shops_data()

    outbreak_scenario_cells = create_outbreak_scenario(
        chain_name, no_of_cases, all_shops
    )
    outbreak_scenario = get_xy(outbreak_scenario_cells)
    outbreak_filename = chain_name + "_" + str(no_of_cases) + "_" + str(trial)
    current_directory = os.getcwd()
    print(current_directory)
    path = os.path.join(
        current_directory,
        "..",
        "Milestone_Diffusion_Model",
        "Data",
        "Outbreaks",
        chain_name,
        str(no_of_cases),
        str(trial),
    )

    print(path)
    create_shapefile(path, outbreak_scenario, outbreak_filename)

    # create_pkl(path, outbreak_scenario, outbreak_filename)
    return outbreak_scenario

