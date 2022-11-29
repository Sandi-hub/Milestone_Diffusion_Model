import pandas as pd

from gravity_model import *


def get_flow(all_stores, selected_stores):
    # First we need to get all cells in which there are two stores:
    flow = pd.read_pickle("./Output_Flow/flow.pkl")


    # First we a re selecting all flows from cells where there is a store of the given chain inside
    selected_flow = flow[flow.index.isin(selected_stores.Gitter_ID_)]

    # These flows are correct unless there is more than the one store of the given chain in any cell 
    # First we only selected the cells in which there are more than one store 
    only_multiple = all_stores[all_stores["Markets_Count"] > 1]
 
    # Now we merge it to the existing flow
    selected_flow = selected_flow.merge(only_multiple['production_potential'], on="Gitter_ID_", how="left")
    selected_stores.set_index('Gitter_ID_', inplace= True)
    selected_flow = selected_flow.merge(selected_stores['TotalSales'], on="Gitter_ID_", how="left")

    adjusted_rows = selected_flow.loc[selected_flow['production_potential'].notna()].iloc[:, 0: -2].multiply((selected_flow.loc[selected_flow['production_potential'].notna()].TotalSales / selected_flow.loc[selected_flow['production_potential'].notna()].production_potential), axis = 0)
    
    selected_flow = selected_flow[selected_flow['production_potential'].isnull()].iloc[:, 0: -2]
    
    selected_flow = selected_flow.append(adjusted_rows, verify_integrity = True)
    print(selected_flow)
    return(selected_flow)





def get_stores(chain_name):
    all_stores = import_shop_data()
    selected_stores = all_stores[all_stores["Name"] == chain_name]
    return selected_stores


stores_Aldi = get_stores("Aldi")

all_stores = get_production_potential()

flow = get_flow(all_stores, stores_Aldi)


# get a list of all chain names and their stores count:
print(import_shop_data().groupby(["Name"]).agg({"ID": "count", "TotalSales": "sum"}))


# print(shops_data.groupby(["Gitter_ID_"])
# print(only_multiple.head())
