import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

path_census_data = (
    r"Input Data\Zensus2011_Ergebnisse_nach_Gitterzellen_fuer_Hamburg.xlsx"
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
        "Gitter_ID_100m": "Gitter_ID",
        "x_mp_100m": "x_centroid",
        "y_mp_100m": "y_centroid",
        "Einwohner": "population",
    }
)

# Write the resulting polygons to a Shapefile
gdf_census.to_file(
    filename=r"Outputs\Shapefile Population Data\population_data_Hamburg_2011.shp",
    driver="ESRI Shapefile",
)
