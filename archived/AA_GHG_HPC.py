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

def main(N_ARCHETYPES=3, group = "Genus", FRAC=1, method='braycurtis', NUM_PC = 7):


    # In[3]:
    print("Starting job!")
    timer = time.time()

    print("Reading Data...")
    dataDir = os.path.join("datasets", "tsv")
    outDir = r"results"

    otuFile = "hadza_otu_nov.tsv"
    taxFile = "hadza_tax_nov.tsv"

    otu = pd.read_table(os.path.join(dataDir, otuFile), index_col=0)
    tax = pd.read_table(os.path.join(dataDir, taxFile), index_col=0)
    
    otu_sample = otu.sample(frac=FRAC)

    print("Done!")

    # In[5]:


    # ------tax_glom() equivalent------

    # group = "Genus" # Change this to "Kingdom", "Phylum", "Class", "Order", "Family", "Genus",to group by that taxonomic level

    # No of uniques => {Kingdom: 1, Phylum: 15, Class: 25, Order: 44, Family: 77, Genus: 305}
    print("Grouping Columns...")
    grouped_cols = tax[group].unique()

    otu_grouped = pd.DataFrame(index=otu_sample.index, columns=grouped_cols)

    i = 0
    for c in tqdm(otu_sample.columns):
        grp_col = tax[group][c]
        
        if(pd.isnull(otu_grouped[grp_col]).all()):
            otu_grouped[grp_col] = otu_sample[c]
            continue

        otu_grouped[grp_col] = otu_grouped[grp_col] + otu_sample[c]
        # print(f"Processed column ({i+1}/{len(otu.columns)}): {grp_col}")

        i += 1

    otu_grouped.columns = otu_grouped.columns.astype(str)
    print("Done!")

    # In[4]:

    print("Log transforming...")
    otu_log = otu_grouped.copy()
    otu_log = otu_log.map(lambda x: np.log2(x+1))

    print("Done!")
    # In[7]:

    print("Performing PCoA...")

    import scipy.spatial
    from skbio.tree import TreeNode
    from skbio.diversity import beta_diversity
    from skbio.stats.ordination import pcoa
    import scipy

    # metadata = pd.read_table('metadata.tsv', index_col=0)
    # FRAC = 0.2

    pcoa_data = otu_log.copy()

    # method = 'braycurtis' #scipy.spatial.distance.jensenshannon  #'jensenshannon' or 'bray_curtis', unifrac is problematic
    bdiv = beta_diversity(
        method, pcoa_data, ids=pcoa_data.index, pairwise_func=scipy.spatial.distance.pdist
    )
    # Perform PCoA
    pcoa_results = pcoa(bdiv, number_of_dimensions=NUM_PC)

    # Extract the coordinates
    pcoa_coordinates = pcoa_results.samples

    print("Done!")


    # In[9]:

    print("Performing Archetypal Analys...is")

    AA_data = pcoa_coordinates.to_numpy()
    AA_data.astype(np.float32)
    method_kwargs = {
        "max_iter_optimizer": 10,
    }

    aa = AA(n_archetypes=N_ARCHETYPES, method="pgd", method_kwargs=method_kwargs, random_state=3)


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
        "dataCols": grouped_cols.tolist()
}


    # In[31]:

    methodName = method if isinstance(method, str) else "jensenshannon"

    outName = f"otu_AA_K{N_ARCHETYPES}_{FRAC}_{group}_{methodName}_{NUM_PC}.json"
    outPath = os.path.join(outDir, outName)

    json.dump(aa_dict, open(outPath, "w"))
    # otu_aa.to_csv(outPath, sep="\t")

    print("Done!")

    print( f"Job completed for K={N_ARCHETYPES}, FRAC={FRAC}. method='{methodName}', Glom_Group={group}") 
    print(f"Job Completed in {time.time() - timer}s")



import sys
# Get system arguments for main: 

if __name__ == "__main__": 
    if len(sys.argv) != 6:
        print("Usage: script.py <N_ARCHETYPES> <FRAC> <group> <method> <NUM_PC>")
        sys.exit(1)

    N_ARCHETYPES = int(sys.argv[1])
    FRAC = float(sys.argv[2])
    group = sys.argv[3]
    method = sys.argv[4]
    NUM_PC = int(sys.argv[5])

    main(N_ARCHETYPES, group, FRAC, method)