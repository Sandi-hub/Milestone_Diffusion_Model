import math

import geopandas as gpd
import numpy as np
import pandas as pd
from ipfn import ipfn
from scipy.spatial import distance_matrix


def import_population_data():
    gdf_population = gpd.read_file(r"Shapefiles\population_data_Hamburg_2011.shp")
    gdf_population = gdf_population[
        ["Gitter_ID_", "Einwohner", "x_mp_100m", "y_mp_100m"]
    ]
    gdf_population = gdf_population.set_index("Gitter_ID_")
    return gdf_population


def get_distance_matrix(production, consumption):
    production_centroids = pd.concat(
        [production.x_mp_100m, production.y_mp_100m], axis=1
    )
    consumption_centroids = pd.concat(
        [consumption.x_mp_100m, consumption.y_mp_100m], axis=1
    )

    arr_distance = distance_matrix(production_centroids, consumption_centroids,)
    arr_distance = arr_distance / 1000

    # TODO inneren oder äußeren Kreis?
    arr_distance[arr_distance == 0] = (128 / (45 * math.pi)) * 50

    return arr_distance


def import_shop_data():
    gdf_Hamburg_shops = gpd.read_file(r"Shapefiles\Hamburg_Shops_with_Gitter_ID.shp")
    gdf_Hamburg_shops = gdf_Hamburg_shops[
        [
            "ID",
            "Name",
            "Address",
            "Zip",
            "Type",
            "TypeAg",
            "SalesSqMTo",
            "SalesSqMOr",
            "TotalSales",
            "Size",
            "Warehouse",
            "Chain",
            "Neighborho",
            "District",
            "AreaSqM",
            "Income",
            "NEARDIST",
            "Gitter_ID_",
            "Einwohner",
        ]
    ]
    gdf_Hamburg_shops = gdf_Hamburg_shops.rename(
        columns={"AreaSqM": "NH_AreaSqM", "Neighborho": "Neighborhood"}
    )
    return gdf_Hamburg_shops


def get_production_potential():
    shops_data = import_shop_data()
    production_potential = shops_data.groupby(["Gitter_ID_"]).agg(
        {"ID": "count", "TotalSales": "sum"}
    )
    production_potential = production_potential.rename(
        columns={"ID": "Markets_Count", "TotalSales": "production_potential"}
    )
    return production_potential


def get_consumption_potential(population_data, total_revenue):
    # For this purpose, we will assume that actually no one lives in cells with -1 (TODO: mit Hanno klären)
    from cmath import nan

    population_data["Einwohner"] = population_data["Einwohner"].replace([-1], 0)
    total_population = population_data["Einwohner"].sum()
    population_data["consumption_potential"] = (
        population_data["Einwohner"].divide(total_population)
    ).multiply(total_revenue)
    population_data = population_data[population_data["Einwohner"] != 0]
    return population_data


def furness_model(
    beta: float, dist_matrix, production_potential, consumption_potential
):
    dM = np.exp(-beta * dist_matrix)

    prod_pot_new = production_potential.production_potential.to_numpy()
    cons_pot_net = consumption_potential.consumption_potential.to_numpy()

    aggregates = [
        prod_pot_new,
        cons_pot_net,
    ]
    dimensions = [[0], [1]]
    IPF = ipfn.ipfn(dM, aggregates, dimensions)

    dM = IPF.iteration()
    flowMatrix = dM
    return flowMatrix


def get_weighted_dist(flow_matrix, dist_matrix):
    WeightDist = np.sum(flow_matrix * dist_matrix) / (np.sum(flow_matrix))
    return WeightDist


def add_indices(flow, production_potential, consumption_potential):
    df_flow = pd.DataFrame(
        flow,
        columns=consumption_potential.index,  # ["Gitter_ID_"],
        index=production_potential.index,  # ["Gitter_ID_"],
    )
    return df_flow


def hyman_model(empirical_mean_shopping_distance, tolerance):
    beta_list = []  # keeping track of the betas
    modeled_means_list = []  # keeping track of the average of the modeled flow distance
    count_loops = 0

    # initializing Hyman with beta_0
    beta_0 = 1.0 / empirical_mean_shopping_distance
    beta_list.append(beta_0)

    # import required input data
    population_data = import_population_data()

    # clean input data (removal of columns)
    production_potential = get_production_potential()  # rows
    total_revenue = production_potential["production_potential"].sum()
    consumption_potential = get_consumption_potential(population_data, total_revenue)
    production_potential = production_potential.merge(
        population_data, on="Gitter_ID_", how="left",
    )

    dist_matrix = get_distance_matrix(production_potential, consumption_potential)
    # TODO die Achsten stehen jetzt für unterschiedliche Gitter_Zellen --> überprüfen ob das noch korrekt ist

    flow_0 = furness_model(
        beta_0, dist_matrix, production_potential, consumption_potential
    )

    modeled_mean_shopping_distance = get_weighted_dist(flow_0, dist_matrix)
    modeled_means_list.append(modeled_mean_shopping_distance)

    if (
        abs(empirical_mean_shopping_distance - modeled_means_list[count_loops])
        <= tolerance
    ):
        flow = flow_0
    while (
        abs(empirical_mean_shopping_distance - modeled_means_list[count_loops])
        > tolerance
    ):
        if count_loops == 0:
            beta_1 = (
                beta_0
                * modeled_means_list[count_loops]
                / empirical_mean_shopping_distance
            )
            beta_list.append(beta_1)
        elif count_loops > 0:
            beta_next = np.abs(
                (
                    (
                        (
                            empirical_mean_shopping_distance
                            - modeled_means_list[count_loops - 1]
                        )
                        * beta_list[count_loops]
                        - (
                            empirical_mean_shopping_distance
                            - modeled_means_list[count_loops]
                        )
                        * beta_list[count_loops - 1]
                    )
                    / (
                        modeled_means_list[count_loops]
                        - modeled_means_list[count_loops - 1]
                    )
                )
            )
            beta_list.append(beta_next)
        beta_current = beta_list[count_loops + 1]

        flow = furness_model(
            beta_current, dist_matrix, production_potential, consumption_potential
        )
        modeled_mean_current = get_weighted_dist(flow, dist_matrix)
        modeled_means_list.append(modeled_mean_current)

        count_loops += 1

        # break if in local minimum and check if any dist was closer to the empirical mean shopping distance
        if count_loops > 20:
            if (
                abs(
                    modeled_means_list[count_loops]
                    - modeled_means_list[count_loops - 5]
                )
            ) < 0.001:
                beta_best = beta_list[modeled_means_list.index(min(modeled_means_list))]
                flow = furness_model(
                    beta_best, dist_matrix, production_potential, consumption_potential
                )
                break

        # break if minimization routine explodes due to numerical issues
        if beta_current > 50:
            beta_best = beta_list[modeled_means_list.index(min(modeled_means_list))]
            flow = furness_model(
                beta_best, dist_matrix, production_potential, consumption_potential
            )
            break
        print(
            "On the %sd. iteration: distance between the modeled and the empirical mean shopping distance is down to %3.4f"
            % (
                count_loops,
                abs(empirical_mean_shopping_distance - modeled_means_list[count_loops]),
            )
        )

        if np.isnan(empirical_mean_shopping_distance):
            raise Exception(
                "Something went wrong, the given empirical mean shopping distance returned nan!"
            )
        if np.isnan(modeled_means_list[count_loops]):
            raise Exception(
                "Something went wrong, the current modeled mean shopping distance is nan!"
            )

    beta_best = beta_list.pop()

    # Sanity Check
    tol_this_time = np.abs(empirical_mean_shopping_distance - modeled_mean_current)
    tol_best = np.abs(
        [empirical_mean_shopping_distance - d for d in modeled_means_list]
    ).tolist()
    if tol_this_time > tol_best[tol_best.index(min(tol_best))]:
        beta_best = beta_list[tol_best.index(min(tol_best))]
        flow = furness_model(
            beta_best, dist_matrix, production_potential, consumption_potential
        )
    print(
        "On the last iteration (%2d.): tolerance is down to %3.4f"
        % (tol_best.index(min(tol_best)), tol_best[tol_best.index(min(tol_best))])
    )
    print("Beta is " + str(beta_best))

    flow_end = add_indices(flow, production_potential, consumption_potential)

    return flow_end

