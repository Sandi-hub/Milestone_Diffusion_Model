import os
import pdb
import random

import pandas as pd

from gravity_model import get_production_potential


def load_flow_data():
    path_to_flow = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "Milestone_Diffusion_Model",
        "Data",
        "Flow",
        "flow.pkl",
    )
    return pd.read_pickle(path_to_flow)


def adjust_flow(flow):
    rows_to_adjust = flow.loc[flow["production_potential"].notna()]
    adjusted_rows = rows_to_adjust.iloc[:, 0:-2].multiply(
        rows_to_adjust.sales / rows_to_adjust.production_potential, axis=0,
    )

    flow = flow[flow["production_potential"].isnull()].iloc[:, 0:-2]
    return pd.concat([flow, adjusted_rows])


def get_flow_for_chain(all_stores, selected_stores):
    total_flow = load_flow_data()

    # First we a are selecting all flows from cells where there is a store of the given chain inside
    flow = total_flow[total_flow.index.isin(selected_stores.cell_id)]

    # These flows are correct unless there is more than the one store of the given chain in any cell
    multi_store_cells = all_stores[all_stores["stores_count"] > 1]

    flow = flow.merge(
        multi_store_cells["production_potential"],
        left_index=True,
        right_index=True,
        how="left",
    )

    flow = flow.merge(
        selected_stores[["cell_id", "sales"]],
        left_on="cell_id",  # Column in selected_flow to join on
        right_on="cell_id",  # Column in selected_stores to join on
        how="left",
    )
    flow = flow.set_index("cell_id")
    flow = adjust_flow(flow)
    return flow


def get_stores(chain_name, all_stores):
    return all_stores[all_stores["chain"] == chain_name]


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
    return cumulative_distribution[
        cumulative_distribution["cumulated"] > random_number
    ].index[0]


def create_outbreak_scenario(chain_name, no_of_cases, all_stores):
    stores_selected_chain = get_stores(chain_name, all_stores)
    sales_per_cell = get_production_potential(all_stores)
    flow = get_flow_for_chain(sales_per_cell, stores_selected_chain)
    cumulative_distribution = get_cumulative_distribution(flow)

    outbreak_scenario = []
    outbreak_scenario = [
        get_location_for_outbreak(cumulative_distribution) for _ in range(no_of_cases)
    ]
    return outbreak_scenario
