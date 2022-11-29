import random

import pandas as pd

from gravity_model import *

no_of_cases = 5
# As we want to make the artificial Outbreaks reproducible, we set the seed for the generation of random numbers
random.seed(4)


def get_flow(all_stores, selected_stores):
    # First we need to get all cells in which there are two stores:
    flow = pd.read_pickle("./Output_Flow/flow.pkl")

    # First we a are selecting all flows from cells where there is a store of the given chain inside
    selected_flow = flow[flow.index.isin(selected_stores.Gitter_ID_)]

    # These flows are correct unless there is more than the one store of the given chain in any cell
    # First we only selected the cells in which there are more than one store
    only_multiple = all_stores[all_stores["Markets_Count"] > 1]

    # Now we merge it to the existing flow
    selected_flow = selected_flow.merge(
        only_multiple["production_potential"], on="Gitter_ID_", how="left"
    )
    selected_stores.set_index("Gitter_ID_", inplace=True)
    selected_flow = selected_flow.merge(
        selected_stores["TotalSales"], on="Gitter_ID_", how="left"
    )

    adjusted_rows = (
        selected_flow.loc[selected_flow["production_potential"].notna()]
        .iloc[:, 0:-2]
        .multiply(
            (
                selected_flow.loc[
                    selected_flow["production_potential"].notna()
                ].TotalSales
                / selected_flow.loc[
                    selected_flow["production_potential"].notna()
                ].production_potential
            ),
            axis=0,
        )
    )

    selected_flow = selected_flow[selected_flow["production_potential"].isnull()].iloc[
        :, 0:-2
    ]

    selected_flow = selected_flow.append(adjusted_rows, verify_integrity=True)

    return selected_flow


def get_stores(chain_name):
    all_stores = import_shop_data()
    selected_stores = all_stores[all_stores["Name"] == chain_name]
    return selected_stores


def get_cumulative_distribution(flow):
    total_sales = flow.values.sum()

    flow = flow.T

    flow["ingoing_sum"] = flow.sum(axis=1)
    flow["percent"] = flow["ingoing_sum"] / total_sales
    flow["cumulated"] = flow["percent"].cumsum()

    flow = flow.iloc[:, -3:]
    return flow


def get_location_for_outbreak(cumulative_distribution):
    random_number = random.random()
    for number in range(0, len(cumulative_distribution.index)):
        if number == 0:
            if 0 <= random_number < cumulative_distribution["cumulated"][number]:
                return cumulative_distribution.iloc[[number]].index[0]
            else:
                pass
        elif number == len(cumulative_distribution) - 1:
            if cumulative_distribution["cumulated"][number - 1] <= random_number <= 1:
                return cumulative_distribution.iloc[[number]].index[0]
            else:
                pass
        else:
            if (
                cumulative_distribution["cumulated"][number - 1]
                <= random_number
                < cumulative_distribution["cumulated"][number]
            ):
                return cumulative_distribution.iloc[[number]].index[0]
            else:
                pass


def get_xy(outbreak_scenario):
    df = pd.DataFrame({"Gitter_ID_": outbreak_scenario})
    population_data = import_population_data()
    df = df.merge(population_data, on="Gitter_ID_", how="left")
    return df


def create_shapefile(outbreak_scenario):
    coordinates = get_xy(outbreak_scenario)
    gdf = gpd.GeoDataFrame(
        coordinates["Gitter_ID_"],
        geometry=gpd.points_from_xy(coordinates["x_mp_100m"], coordinates["y_mp_100m"]),
    )
    gdf.to_file("outbreak_cases/Aldi_outbreak.shp")


stores_Aldi = get_stores("Aldi")

all_stores = get_production_potential()

flow = get_flow(all_stores, stores_Aldi)

cumulative_distribution = get_cumulative_distribution(flow)


outbreak_scenario = []
for j in range(0, no_of_cases):
    outbreak_scenario.append(get_location_for_outbreak(cumulative_distribution))
print(outbreak_scenario)

create_shapefile(outbreak_scenario)
# get a list of all chain names and their stores count:
# print(import_shop_data().groupby(["Name"]).agg({"ID": "count", "TotalSales": "sum"}))

