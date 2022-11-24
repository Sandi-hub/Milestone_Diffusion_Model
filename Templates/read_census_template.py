# -*- coding: utf-8 -*-

import pandas as pd

f1 = "S:\HiDrive\Weitere Projekte\HI_Projekt_Systemische Herausforderungen der Waermewende\Data\Zensus\Geb100m.csv"

'''Raster cells are provided as csv or as shape, there are not needed however, since the ID of each cell, present here in the Geb100m is a concatination of "size_N_value_E_Value" in ETRS89 LAEA EPSG: 3035, where the coords
are of the lower left cell corner. This means 100mN26865E43357 means 100 meter cell with Y = 2686500 abd X = 4335700 as lower left corner. Addint 50m to both equals the cell centre. Note that the order is reversed, first Y than X'''


'''here the bounding box of Hamburg to filter'''
x_max = 4342612.79414699
x_min = 4303113.90442379
y_max = 3403572.91763726
y_min = 3365264.16543474

'''the zensus dataset'''
zensus_df = pd.read_csv(f1, header= 0, usecols=[0,2,3,4,5,6], chunksize=10**6)

ham_data_chunks = []
for chunk in zensus_df:
    #split cell id to get the coords
    chunk["Y"], chunk["X"]  = chunk["Gitter_ID_100m"].str.split("E", 1).str
    chunk["Y"] = chunk["Y"].str[5:] + "00"
    chunk["X"] += "00"
    chunk["X"] = chunk["X"].astype(int, copy=False)
    chunk["Y"] = chunk["Y"].astype(int, copy=False)
    
    ham_filter = (chunk["X"] <= x_max) & (chunk["X"] >= x_min) & (chunk["Y"] <= y_max) & (chunk["Y"] >= y_min)
    print ham_filter
    filtered = chunk[ham_filter]
    if filtered.size > 0:
        ham_data_chunks.append(filtered)

ham_data = pd.concat(ham_data_chunks, ignore_index=True)

#note for some reaosn pd.pivot_table(df..) works, while df.pivot(..) does not. Good to know.
ham_data_pivoted = pd.pivot_table(ham_data, index=['Gitter_ID_100m', "X", "Y"], columns='Auspraegung_Text', values=['Anzahl','Anzahl_q'])

#merge the multindex, so that the columns are uniquely marked.
ham_data_pivoted.columns = [' '.join(col).strip() for col in ham_data_pivoted.columns.values]
ham_data_pivoted.to_csv("S:\HiDrive\Ivan\PhD Thesis\Data\Zensus_geb_Hamburg_ETRS89LAEA.csv")
