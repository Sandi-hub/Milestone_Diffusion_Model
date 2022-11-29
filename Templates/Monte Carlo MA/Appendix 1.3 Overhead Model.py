import multiprocessing as mp
import random
from functools import partial

import numpy as np
import pandas as pd
import psycopg2

from Model import calculate_model


def total_weight_product(product):
    return df_all_products.loc[
        (df_all_products['product_category'] == product[0]) & (df_all_products['country'] == product[1])][
        'weight'].sum()


if __name__ == "__main__":
    # Open Connection to Database where the BOL Data is stored
    postgresDB = psycopg2.connect(user="postgres",
                                  password="Masterthesis123",
                                  host="localhost",
                                  port="5432",
                                  database="bol_data3")
    postgresCursor = postgresDB.cursor()

    # We query all data that we need for the algorithm: In our case data is stored in 2 different tables
    query_all_products = "Select SUM(product.weight) from bol_data3.product"
    postgresCursor.execute(query_all_products)
    weight_all_products = postgresCursor.fetchall()[0][0]

    absolute_distribution_query = "Select * from bol_data3.absolute_distribution_materialized"
    df_absolute_distribution = pd.read_sql(absolute_distribution_query, postgresDB)

    # As we want to calculate the model for all products, we first need to have a list of all products
    # One product is defined as the product category combined with the country of origin
    query_all_products = "Select product_category, country, weight, no_of_arrival_ports from bol_data3.all_products_materialized"
    df_all_products = pd.read_sql(query_all_products, postgresDB)

    postgresDB.close()

    # The prior probability of the product can already be calculated
    df_all_products['prior_prob'] = df_all_products['weight'] / weight_all_products

    # We can also already calculate the share of each port a product is shipped to
    df_absolute_distribution['port_share'] = df_absolute_distribution.apply(
        lambda row: row['weight'] / total_weight_product(
            (row['product_category'], row['country'])), axis=1)

    # As we want to make the artificial Outbreaks reproducible, we set the seed for the generation of random numbers
    random.seed(4)

    # Multiprocessing
    df_split = np.array_split(df_all_products, 6)
    pool = mp.Pool(6)

    process = partial(calculate_model, df_all_products=df_all_products,
                      df_absolute_distribution=df_absolute_distribution)

    df_result = pd.concat(pool.map(process,
                                   iterable=[df_split[0], df_split[1], df_split[2], df_split[3], df_split[4],
                                             df_split[5]]), axis=1)

    df_result['No_of_Cases'] = np.array(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
    pool.close()
    pool.join()

    # Define where the results should be saved
    writer = pd.ExcelWriter(
        "C:/Users/srude/OneDrive - Kühne Logistics University/Masterthesis/Reporting/Sensitivity.xlsx")
    df_result.to_excel(writer)
    writer.save()
