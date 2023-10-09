import os

import geopandas as gpd
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_and_save_shops_for_year(year=2011):
    path_shops_csv = os.path.join(
        parent_dir, "Data", "Shops", "CSV", "shops_final_564m_21.11.21.csv"
    )

    df_shops = pd.read_csv(path_shops_csv, encoding="latin-1")

    print(df_shops["Year"].unique())

    df_shops_2011 = df_shops[df_shops["Year"] == year]
    df_shops_2011 = df_shops_2011[
        [
            "ID",
            "TotalSales",
            "Name",
            "Address",
            "Zip",
            "Type",
            "Warehouse",
            "Chain",
            "Neighborhood",
            "District",
        ]
    ]

    df_shops_2011.to_csv(
        os.path.join(parent_dir, "Data", "Shops", "CSV", "shops_2011.csv"), index=False
    )


def shp_to_pkl_shops_data(
    path_shops_shapefile=os.path.join(
        parent_dir, "Data", "Shops", "Shapefile", "Hamburg_Shops_with_Gitter_ID.shp"
    )
):

    gdf_Hamburg_shops = gpd.read_file(path_shops_shapefile, engine="pyogrio")

    df_Hamburg_shops = gdf_Hamburg_shops[
        ["Gitter_ID_", "Chain", "Name", "TotalSales"]
    ].copy()
    df_Hamburg_shops["x"] = gdf_Hamburg_shops["geometry"].x.copy()
    df_Hamburg_shops["y"] = gdf_Hamburg_shops["geometry"].y.copy()
    df_Hamburg_shops = df_Hamburg_shops.rename(
        columns={
            "Gitter_ID_": "cell_id",
            "Chain": "chain",
            "TotalSales": "sales",
            "Einwohner": "population",
        }
    )

    # Transform to KM
    df_Hamburg_shops["x"] = df_Hamburg_shops["x"] / 1000
    df_Hamburg_shops["y"] = df_Hamburg_shops["y"] / 1000

    os.makedirs(os.path.join(parent_dir, "Data", "Shops"), exist_ok=True)
    df_Hamburg_shops.to_pickle(os.path.join(parent_dir, "Data", "Shops", "shops.pkl"))


shp_to_pkl_shops_data()


def get_shops_data():
    return pd.read_pickle(os.path.join(parent_dir, "Data", "Shops", "shops.pkl"))

