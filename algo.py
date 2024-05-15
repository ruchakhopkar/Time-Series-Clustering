#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:12:53 2022

@author: ruchak
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
import time

df = pd.read_csv('FINAL_OUTPUTS//phase_27154//pca_reduced_27154_phase.csv') #EDIT
df = df.dropna()
#df = df.iloc[:15, :]
all_dist = np.zeros((len(df),len(df)))
for i in range(len(df)):
    print(i)
    a = time.time()
    print('STarting')
    for j in range(i+1, len(df)):
            print('In loop ', j)
            all_dist[i,j] = all_dist[j,i] = fastdtw(df.iloc[i,:], df.iloc[j,:], dist = euclidean)[0]
    b = time.time()
    print('Time for 1 dataframee ', (b-a)/60)

all_dist = np.array(all_dist)