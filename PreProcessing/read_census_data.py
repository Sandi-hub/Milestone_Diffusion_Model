import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def xlsx_to_shp_population_data():
    path_census_data = os.path.join(
        parent_dir,
        "Data",
        "Population",
        "Zensus2011_Ergebnisse_nach_Gitterzellen_fuer_Hamburg.xlsx",
    )

    df_census = pd.read_excel(
        path_census_data, sheet_name="Einwohnerzahl 100m-Gitterzellen", header=2
    )

    # The last two rows are just source references but do not belong to the actual population data
    df_census = df_census.drop([76039, 76038])

    # Compute the latitude and longitude values for each Polygon given the centroid
    ls_polygon_geometries = []
    for i in range(len(df_census)):
        lon_point_list = [
            df_census.iloc[i, 3] - 50,
            df_census.iloc[i, 3] - 50,
            df_census.iloc[i, 3] + 50,
            df_census.iloc[i, 3] + 50,
        ]
        lat_point_list = [
            df_census.iloc[i, 4] - 50,
            df_census.iloc[i, 4] + 50,
            df_census.iloc[i, 4] + 50,
            df_census.iloc[i, 4] - 50,
        ]
        polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
        ls_polygon_geometries.append(polygon_geom)

    crs = "epsg:3035"
    gdf_census = gpd.GeoDataFrame(df_census, crs=crs, geometry=ls_polygon_geometries)

    gdf_census = gdf_census.drop(["OBJECTID", "CellCode"], axis=1)
    gdf_census = gdf_census.rename(
        columns={
            "Gitter_ID_100m": "cell_id",
            "x_mp_100m": "x_centroid",
            "y_mp_100m": "y_centroid",
            "Einwohner": "population",
        }
    )

    path_shapefile_population = os.path.join(
        parent_dir,
        "Data",
        "Population",
        "Shapefile",
        "population_data_Hamburg_2011.shp",
    )
    # Write the resulting polygons to a Shapefile
    gdf_census.to_file(
        filename=path_shapefile_population, driver="ESRI Shapefile",
    )


def shp_to_pkl_population_data(
    path_shapefile_population=os.path.join(
        parent_dir,
        "Data",
        "Population",
        "Shapefile",
        "population_data_Hamburg_2011.shp",
    )
):

    gdf_population = gpd.read_file(path_shapefile_population, engine="pyogrio")

    # We drop the geometry data since we only the the centroids for the gravity model
    df_population = gdf_population[
        ["cell_id", "population", "x_centroid", "y_centroid"]
    ]
    df_population = df_population.set_index("cell_id")

    # Transform to KM
    df_population["x_centroid"] = df_population["x_centroid"] / 1000
    df_population["y_centroid"] = df_population["y_centroid"] / 1000

    # The value of -1 in the column "population" means either uninhabited or to be kept secret. We will assume that all cells with a value of -1 are uninhabited
    df_population["population"] = df_population["population"].replace([-1], 0)
    os.makedirs(os.path.join(parent_dir, "Data", "Population"), exist_ok=True)
    df_population.to_pickle(
        os.path.join(parent_dir, "Data", "Population", "population.pkl")
    )


def get_population_data():
    return pd.read_pickle(
        os.path.join(parent_dir, "Data", "Population", "population.pkl")
    )

