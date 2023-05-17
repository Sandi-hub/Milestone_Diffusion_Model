import os

import geopandas as gpd

from gravity_model import *

""" 
Here, the input parameter for the Hyman algorithm can be set
"""

empirical_mean_shopping_distance = (
    4.65  # taken from Schlaich(2020); all units are in km
)
tolerance = 0.01


def import_population_data():
    gdf_population = gpd.read_file(
        r"Outputs\Shapefile Population Data\population_data_Hamburg_2011.shp"
    )

    # We drop the geometry data since we only the the centroids for the gravity model
    df_population = gdf_population[
        ["Gitter_ID", "population", "x_centroid", "y_centroid"]
    ]
    df_population = df_population.set_index("Gitter_ID")

    # The value of -1 in the column "population" means either uninhabited or to be kept secret. We will assume that all cells with a value of -1 are uninhabited
    df_population["population"] = df_population["population"].replace([-1], 0)

    os.makedirs("Outputs/Population", exist_ok=True)
    df_population.to_pickle("Outputs/Population/population.pkl")

    return df_population


def import_shop_data():
    gdf_Hamburg_shops = gpd.read_file(
        r"Input Data\Shops Data\Hamburg_Shops_with_Gitter_ID.shp"
    )
    df_Hamburg_shops = gdf_Hamburg_shops[
        ["ID", "Name", "TotalSales", "Chain", "Gitter_ID_", "Einwohner"]
    ]
    df_Hamburg_shops = df_Hamburg_shops.rename(
        columns={"Einwohner": "population", "Gitter_ID_": "Gitter_ID"}
    )

    os.makedirs("Outputs/Retailer", exist_ok=True)
    df_Hamburg_shops.to_pickle("Outputs/Retailer/stores.pkl")

    return df_Hamburg_shops


df_population = import_population_data()
df_Hamburg_shops = import_shop_data()

flow = hyman_model(
    empirical_mean_shopping_distance, tolerance, df_population, df_Hamburg_shops
)

os.makedirs("Outputs/Flow", exist_ok=True)
flow.to_pickle("Outputs/Flow/flow.pkl")
# flow.to_csv("Outputs/Flow/flow.csv")

print(flow)
