#!/usr/bin/env python
# coding: utf-8
import concurrent.futures
import multiprocessing as mp
import time
import pandas as pd
import numpy as np
import scipy, os, sys
from statsmodels.stats.multitest import fdrcorrection
start = time.perf_counter()

FLD = sys.argv[1].strip('/')
latent = pd.read_pickle(os.path.join(FLD,'Latent-space.pkl.gz'))
dataset_name = os.path.split(FLD)[-1]
savepath = os.path.join(FLD, f"{dataset_name}-correlations-w-pval-v1.csv")
savepath2 = os.path.join(FLD, f"{dataset_name}-correlations-w-pval-v2.csv")
savepath3 = os.path.join(FLD, f"{dataset_name}-modifications-list.csv")

df = pd.DataFrame(set(latent.index), columns=['PTM_ID'])
df[['Gene','POS','RES','MOD']] = df.PTM_ID.str.split('|', expand=True)
df['res_mod'] = df.RES + '-' + df.MOD
df.to_csv(savepath3, index=False)


def calculate_correlations_w_pval(i, latent_space=latent):
    correlations = []
    print(i+1,'/',len(latent_space)-1)
    ptmA = latent_space.index[i]
    for j in range(i+1, len(latent_space)):
        ptmB = latent.index[j]
        corr, pval = scipy.stats.pearsonr(latent.iloc[i],latent.iloc[j])
        if pval < .05:
            correlations.append([ptmA, ptmB, np.round(corr,2), pval])
    return pd.DataFrame(correlations, columns=['nodeA','nodeB','Score','pval'])

if __name__=='__main__':
    mp.freeze_support()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(calculate_correlations_w_pval, range(len(latent)-1))
        correlations = pd.concat(results, ignore_index=True)

    correlations.sort_values('pval', inplace=True)
    correlations['adj_pval'] = scipy.stats.false_discovery_control(correlations.pval)
    print(savepath)
    correlations.to_csv(savepath, index=False)
    # correlations[correlations.Score>=.50].to_csv(savepath2, index=False)


finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')

print(correlations[correlations.adj_pval<.01].shape)
print(correlations[correlations.Score>=.70].shape)
    