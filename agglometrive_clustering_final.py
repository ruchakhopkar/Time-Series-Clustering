#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:11:58 2022

@author: ruchak
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import os 
max_d = 0.7*3000   #EDIT
dist_matrix = np.load('pca_reduced_hierarchial_27154_gain.npy')   #EDIT

#########################################################################
#PLOTTING DENDROGRAM
distArray = ssd.squareform(dist_matrix)
plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
dendrogram = sch.dendrogram(sch.linkage(distArray, method  = "ward"))
plt.axhline(y=max_d, c='k')
plt.title('Dendrogram for gain 27154')    #EDIT
plt.xlabel('Distances')
plt.ylabel('Euclidean distances')
plt.show()

###########################################################################
#HIERARCHIAL CLUSTERING
#EDIT
agc = AgglomerativeClustering(n_clusters = 3, affinity = 'precomputed', linkage = 'average')
clusters = agc.fit_predict(dist_matrix)

df_clustered = pd.read_csv('sample_27154_GAIN.csv')   #EDIT
df_clustered = df_clustered.dropna()
df_clustered['CLUSTER_ID'] = clusters

###########################################################################
#EDIT
col_list = ['PEAK_FREQUENCY', 'GAIN', 'DRIVE_SERIAL_NUM', 'HD_PHYS_PSN']
TRK_NUM = 27154    #EDIT
SPC_ID = 9

#LOADING DATA
df = pd.DataFrame()  # creating an empty dataframe for computation
df_output = pd.DataFrame()  #creating an empty dataframe to store all results
l1 = sorted(os.listdir('P282_BODE_GAIN_PHASE//'))
#concatenating all the dataframes together
for i in l1:
    df_temp = pd.read_csv('P282_BODE_GAIN_PHASE//'+i)
    df_temp = df_temp[(df_temp.SPC_ID == SPC_ID) & (df_temp.TRK_NUM == TRK_NUM)]
    #df_output = pd.concat([df_output, df_temp], axis = 0)
    df_temp = df_temp[col_list]
    df = pd.concat([df, df_temp], axis=0)

#matching the drive serial number and head psn with the already preprocessed data
freqs = df['PEAK_FREQUENCY'].unique().tolist()      #getting unique frequencies to recreate the preprocessed dataframe
freqs_col = []

for i in range(len(freqs)):
    freqs_col.append('freq_'+str(freqs[i]))     #create columns 
    
freqs_col.append('DRIVE_HD')    #add an extra column this time

drive_hd = []       #creating the DRIVE_HD column by concatenating the serial number and hd phys psn
for i in range(len(df)):
    drive_hd.append(str(df.iloc[i, -2]) + '_' + str(df.iloc[i, -1]))
df['DRIVE_HD'] = drive_hd

df_hierarchial = pd.DataFrame(columns = freqs_col)      #reproducing the preprocessed dataframe

unique_drive_hd_combs = df['DRIVE_HD'].unique().tolist()

for i in range(len(unique_drive_hd_combs)):     #creating one row per drive hd combination
    df_temp = df[df['DRIVE_HD'] == unique_drive_hd_combs[i]]
    myDict = {key: None for key in freqs}
    for j in range(len(df_temp)):
        myDict[df_temp.iloc[j, 0]] = df_temp.iloc[j,1]
    X = list(myDict.values())  
    X.append(unique_drive_hd_combs[i])
    df_hierarchial.loc[i,:] = X
df_hierarchial = df_hierarchial.dropna()        #dropping entries having nan values

df_hierarchial['CLUSTER_ID'] = clusters   #copy all the clusters in the new dataframe

final_drives = df_hierarchial['DRIVE_HD'].tolist()  #making a new dataframe with the correct format
df = df[df['DRIVE_HD'].isin(final_drives)]

df_output = pd.DataFrame()
freqs_cols = []
gain = []
drive_srl_num = []
phys_psn = []
clusters = []

for i in range(len(df_hierarchial)):
    print(i)
    freqs_cols.extend(freqs)
    gain.extend(df_hierarchial.iloc[i,:-2])
    drive_srl_num.extend([df_hierarchial.iloc[i, -2].split('_')[0]] * 1941)
    phys_psn.extend([int(df_hierarchial.iloc[i, -2].split('_')[1])] * 1941)
    clusters.extend([df_hierarchial.iloc[i, -1]] * 1941)
df_output['PEAK_FREQUENCY'] = freqs_cols
df_output[col_list[1]] = gain
df_output['DRIVE_SERIAL_NUM'] = drive_srl_num
df_output['HD_PHYS_PSN'] = phys_psn
df_output['CLUSTER_ID'] = clusters

#EDIT
for i in range(3):     #plotting the clusters
    df_temp = df_output[df_output['CLUSTER_ID'] == i]
    plt.scatter(df_temp['PEAK_FREQUENCY'], df_temp[col_list[1]])
    plt.xlabel('PEAK FREQUENCY')
    plt.ylabel(col_list[1])
    plt.title(col_list[1]+' vs PEAK FREQUENCY for '+str(TRK_NUM))
    plt.show()
    
#EDIT
df_output.to_csv('FINAL_OUTPUTS_HIERARCHIAL//gain_27154//hierarchial_clustered_output.csv', index = False)    #EDIT