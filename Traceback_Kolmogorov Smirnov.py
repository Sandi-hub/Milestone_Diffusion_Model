# For a given outbreak and a given chain we calculate the distance of each case to the next store of that chain
# We take this as a sample and test with the kolmogorov smirnov test whether it fit to our distribution.

import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from shapely.ops import nearest_points
https://towardsdatascience.com/nearest-neighbour-analysis-with-geospatial-data-7bcd95f34c0e

def nearest(
    row,
    geom_union,
    df1,
    df2,
    geom1_col="geometry",
    geom2_col="geometry",
    src_column=None,
):
    """Find the nearest point and return the corresponding value from specified column."""
    # Find the geometry that is closest
    nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest][src_column].get_values()[0]
    return value


gdf_outbreak = gpd.read_file(r"outbreak_cases\Aldi_outbreak_1.shp")
gdf_shops = gpd.read_file(r"Shapefiles\Hamburg_Shops_with_Gitter_ID.shp")
print(gdf_outbreak.head())
print(gdf_shops.head())

unary_union = gdf_shops.unary_union
print(unary_union)

gdf_outbreak["nearest_id"] = gdf_outbreak.apply(
    nearest,
    geom_union=unary_union,
    df1=gdf_outbreak,
    df2=gdf_shops,
    geom1_col="geometry",
    src_column="Gitter_ID_",
    axis=1,
)

print(gdf_outbreak.head())
