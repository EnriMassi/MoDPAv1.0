#!/usr/bin/env python
# coding: utf-8
import polars as pl
import pandas as pd
import numpy as np
import re, argparse
from PTMmap import Fasta, PTMs_remapping
from string import ascii_uppercase

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("data_folder", type=str, help="Path to processed folder with the data.")
    p.add_argument("date", type=str, help="Date in YYYY-MM-DD format.")
    p.add_argument("target_fasta_path", type=str)
    p.add_argument("contam_fasta_path", type=str)
    return p.parse_args()


# In[ ]:
args = parse_cli()

# In[ ]:
fasta = {}
Fasta.getFasta(args.target_fasta_path, fasta)
# Fasta.getFasta('C:/Users/Enrico/OneDrive - UGent/UniProtKB_2023/human_combined_metamORF_Swiss_Prot.fasta.gz', fasta)
prot_targets = set(fasta.keys())
print('# Target proteins =',len(prot_targets))

cont_fasta = {}
Fasta.getFasta(args.contam_fasta_path, cont_fasta)
prot_contaminants = set(cont_fasta.keys())
print('# Contaminants =',len(prot_contaminants))
del fasta, cont_fasta


# In[ ]:

inpath  = f"{args.data_folder}/{args.date}_PTMs_counts_relative.csv.gz"
outpath = inpath.replace('.csv.gz', '_prefiltered.csv.gz')
# inpath, outpath

relcounts = pl.read_csv(inpath)
print(relcounts.shape)

relcounts = relcounts.filter(
    pl.col('UniAcc').is_in(prot_targets)&
    pl.col('UniAcc').is_in(prot_contaminants).not_()&
    (pl.col('relative_psm_counts') > 0)&
    (pl.col('relative_psm_counts') < 1)    
)
print(relcounts.shape)
print("PTMs in dataset:")
print(relcounts.select('ptm_name').unique())

relcounts.to_pandas().to_csv(outpath, index=False, compression='gzip', encoding='utf-8')
print("Prefiltered data saved to:", outpath)
