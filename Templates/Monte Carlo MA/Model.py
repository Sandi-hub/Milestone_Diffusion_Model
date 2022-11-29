import random

import numpy as np
import pandas as pd

df_success_rate = pd.DataFrame()


def calculate_product(simulated_product, df_all_products, df_absolute_distribution, no_of_arrival_ports):
    print(simulated_product[0] + simulated_product[1])
    column_title = (simulated_product[0] + simulated_product[1])
    list_outbreak_scenario_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
                                    100]

    list_all_ranks_1 = []
    list_all_ranks_2 = []
    list_all_ranks_3 = []
    list_all_ranks_5 = []

    for no_of_outbreak_cases in list_outbreak_scenario_sizes:
        # Set the number of simulations
        if no_of_arrival_ports == 1:
            no_of_trials_per_scenario = 1
        else:
            no_of_trials_per_scenario = 5 + round((no_of_outbreak_cases * no_of_arrival_ports) / 5)

        list_sensitivity_1 = []
        list_sensitivity_2 = []
        list_sensitivity_3 = []
        list_sensitivity_5 = []

        for x in range(0, no_of_trials_per_scenario):
            # We create an outbreak scenario containing the given number of outbreak cases (no_of_outbreak_cases)
            outbreak_scenario = pd.DataFrame(generate_outbreak((simulated_product[0], simulated_product[1]),
                                                               no_of_outbreak_cases, df_absolute_distribution),
                                             columns=['port'])

            # Now we calculate the probabilities for each product of being the contaminated one and save the result
            # to list_likelihoods
            df_likelihoods = df_all_products.copy()[['product_category', 'country', 'prior_prob']]

            # We want to merge the outbreak scenario to each record in the likelihood table
            df_likelihoods['key'] = 0
            outbreak_scenario['key'] = 0
            merge1 = pd.merge(df_likelihoods, outbreak_scenario, on='key')
            merge2 = pd.merge(merge1, df_absolute_distribution, how='left',
                              left_on=['product_category', 'country', 'port'],
                              right_on=['product_category', 'country', 'port_of_arrival'])
            merge2['port_share'] = merge2['port_share'].fillna(0)
            df_likelihoods = merge2.groupby(['product_category', 'country', 'prior_prob'], as_index=False)[
                'port_share'].prod()
            df_likelihoods.rename({'port_share': 'outbreak_prob'}, axis='columns', inplace=True)

            df_likelihoods = df_likelihoods[df_likelihoods.outbreak_prob != 0]

            df_likelihoods['Likelihood'] = df_likelihoods['prior_prob'].values * df_likelihoods[
                'outbreak_prob'].values

            df_sorted_likelihoods = df_likelihoods.sort_values("Likelihood", ascending=False)
            df_sorted_likelihoods = df_sorted_likelihoods.reset_index(drop=True)

            # We want to find the rank on which the initially simulated contaminated product is
            rank_simulated_product = df_sorted_likelihoods.loc[
                (df_sorted_likelihoods['product_category'] == simulated_product[0]) & (
                        df_sorted_likelihoods['country'] == simulated_product[1])]

            # unique identification
            if rank_simulated_product.index == 0:
                list_sensitivity_1.append(1)
            else:
                list_sensitivity_1.append(0)

            # Set of 2
            if rank_simulated_product.index < 2:
                list_sensitivity_2.append(1)
            else:
                list_sensitivity_2.append(0)

            # Set of 3
            if rank_simulated_product.index < 3:
                list_sensitivity_3.append(1)
            else:
                list_sensitivity_3.append(0)

            # Set of 5
            if rank_simulated_product.index < 5:
                list_sensitivity_5.append(1)
            else:
                list_sensitivity_5.append(0)

        list_all_ranks_1.append(np.sum(list_sensitivity_1) / no_of_trials_per_scenario)
        list_all_ranks_2.append(np.sum(list_sensitivity_2) / no_of_trials_per_scenario)
        list_all_ranks_3.append(np.sum(list_sensitivity_3) / no_of_trials_per_scenario)
        list_all_ranks_5.append(np.sum(list_sensitivity_5) / no_of_trials_per_scenario)

    df_success_rate[column_title + "_1"] = np.array(list_all_ranks_1)
    df_success_rate[column_title + "_2"] = np.array(list_all_ranks_2)
    df_success_rate[column_title + "_3"] = np.array(list_all_ranks_3)
    df_success_rate[column_title + "_5"] = np.array(list_all_ranks_5)


def calculate_model(df_split, df_all_products, df_absolute_distribution):
    """
    Calculate the model for all products
    """
    df_split.apply(
        lambda row: calculate_product((row['product_category'], row['country']), df_all_products,
                                      df_absolute_distribution, row['no_of_arrival_ports']), axis=1)
    return df_success_rate


def generate_outbreak(simulated_product, no_of_cases, df_absolute_distribution):
    """
    Generates an outbreak scenario based on the given product and with the given number of cases using the Monte
    Carlo method

    :param df_absolute_distribution:
    :param simulated_product: Simulated contaminated product whose distribution is used
    :param no_of_cases: Number of outbreak cases that should be generated
    :return: An outbreak scenario containing port regions where an outbreak case is reported
    """
    # We need to find all ports where the product is shipped to and the corresponding weight-value
    absolute_distribution_product = df_absolute_distribution[
        (df_absolute_distribution['product_category'] == simulated_product[0]) & (
                df_absolute_distribution['country'] == simulated_product[1])]

    share_distribution = absolute_distribution_product.copy().loc[:, "port_of_arrival":"weight"]
    share_distribution = share_distribution.reset_index(drop=True)

    # We want to use the shares and not the total weight, therefore calculate total weight for that
    # product and build list with shares
    total_weight = share_distribution['weight'].sum()
    share_distribution['percent'] = share_distribution['weight'] / total_weight

    # Cumulate the percentages in order to use Monte Carlo
    share_distribution['cumulated'] = share_distribution['percent'].cumsum()

    # Now we can draw locations for our outbreak scenario
    outbreak_scenario = []
    for j in range(0, no_of_cases):
        outbreak_scenario.append(get_location_for_outbreak(share_distribution))
    return outbreak_scenario


def get_location_for_outbreak(cumulative_distribution):
    """
    Generates a random number to draw a port location from the cumulative distribution

    :param cumulative_distribution: cumulative distribution of the simulated contaminated product
    :return: port location
    """
    random_number = random.random()
    for number in range(0, len(cumulative_distribution.index)):
        if number == 0:
            if 0 <= random_number < cumulative_distribution['cumulated'][number]:
                return cumulative_distribution['port_of_arrival'][number]
            else:
                pass
        elif number == len(cumulative_distribution) - 1:
            if cumulative_distribution['cumulated'][number - 1] <= random_number <= 1:
                return cumulative_distribution['port_of_arrival'][number]
            else:
                pass
        else:
            if cumulative_distribution['cumulated'][number - 1] <= random_number < cumulative_distribution['cumulated'][
                number]:
                return cumulative_distribution['port_of_arrival'][number]
            else:
                pass
