import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import time
import json
from archetypes import AA

from archetypes.visualization import stacked_bar, simplex
from archetypes.datasets import sort_by_archetype_similarity, permute_dataset

import sys
import argparse

def main(N_ARCHETYPES=3, group="Genus", FRAC=1, method='braycurtis', NUM_PC=7, otu_file=None):

    # In[3]:
    print("Starting job!")
    timer = time.time()

    print("Reading Data...")
    dataDir = os.path.join("datasets", "tsv")
    outDir = r"results"

    # Use provided file paths or default to existing ones
    otuFile = otu_file or r"datasets/compendium/taxonomic_table.csv/raw_taxa_110.csv"

    try: 
        otu = pd.read_csv(otuFile, index_col="sample")
    except: 
        otu = pd.read_csv(otuFile, index_col=0)
    print(otu.shape)
    
    # Drop all columns starting with "Unnamed"
    otu = otu.loc[:, ~otu.columns.str.startswith("Unnamed")]

    otu_sample = otu.sample(frac=FRAC)
    print(otu_sample.shape)
    print("Done!")
    # print the shape of the otu table
    
    print(otu_sample.shape)

    # In[4]:
    otu_grouped = otu_sample

    print("Log transforming...")
    otu_log = otu_grouped.copy()
    otu_log = otu_log.map(lambda x: np.log2(x+1))

    print("Done!")
    pcoa_coordinates = otu_log

    # In[9]:

    print("Performing Archetypal Analysis...")


    AA_data = pcoa_coordinates.to_numpy()

    method_kwargs = {
        "max_iter_optimizer": 100,
    }

    init = "aa_plus_plus"
    init_kwargs = { "kwargs": {"const": 1000} }

    aa = AA(n_archetypes=N_ARCHETYPES, method="pgd", init=init, init_kwargs=init_kwargs, method_kwargs=method_kwargs, random_state=3407)


    aa.fit(AA_data)

    print(aa.archetypes_)
    
    print("Done!")

    # In[10]:

    data_sorted, info = sort_by_archetype_similarity(AA_data, [aa.similarity_degree_], aa.archetypes_)

    similarity_degree_permuted, _ = permute_dataset(aa.similarity_degree_, info["perms"])

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    simplex(aa.similarity_degree_, alpha=0.5, ax=ax[0], show_vertices=True)
    stacked_bar(similarity_degree_permuted, ax=ax[1])

    # plt.show()


    # In[30]:

    # Make a json file containing all details for the archetypes
    aa_dict = {
        "archetypes": aa.archetypes_.tolist(),
        "similarity_degree": aa.similarity_degree_.tolist(),
        "labels": aa.labels_.tolist(), 
        "index": pcoa_coordinates.index.tolist(),
        "columns": [f"Archetype {i+1}" for i in range(N_ARCHETYPES)], 
        "loss": aa.loss_,
        "info_perms": info["perms"][0].tolist(),
        "info_scores": info["scores"][0].tolist(),
        "info_alphas": info["alphas"][0].tolist(),
}


    # In[31]:

    methodName = method if isinstance(method, str) else "jensenshannon"

    outName = f"otu_AA_K{N_ARCHETYPES}_{FRAC}_{group}_{methodName}.json"
    outPath = os.path.join(outDir, outName)

    json.dump(aa_dict, open(outPath, "w"))
    # otu_aa.to_csv(outPath, sep="\t")

    print("Done!")

    print( f"Job completed for K={N_ARCHETYPES}, FRAC={FRAC}. method='{methodName}', Glom_Group={group}") 
    print(f"Job Completed in {time.time() - timer}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Archetypal Analysis on OTU data.")
    parser.add_argument("N_ARCHETYPES", type=int, help="Number of archetypes.")
    parser.add_argument("FRAC", type=float, help="Fraction of the dataset to sample.")
    parser.add_argument("--group", type=str, default=None, help="Grouping level (e.g., Genus).")
    parser.add_argument("--method", type=str, default=None, help="Distance method (e.g., braycurtis).")
    parser.add_argument("--num_pc", type=int, default=None, help="Number of principal components.")
    parser.add_argument("--otu_file", type=str, default=None, help="Path to the OTU file.")

    args = parser.parse_args()

    main(
        N_ARCHETYPES=args.N_ARCHETYPES,
        FRAC=args.FRAC,
        group=args.group,
        method=args.method,
        NUM_PC=args.num_pc,
        otu_file=args.otu_file,
    )