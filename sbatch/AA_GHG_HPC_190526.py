import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import json
from archetypes import AA
from sklearn.utils.validation import check_array

from archetypes.visualization import stacked_bar, simplex
from archetypes.datasets import sort_by_archetype_similarity, permute_dataset

import argparse

# Compatibility shim:
# archetypes==0.6.x expects an estimator method `_validate_data`, which is
# not available with newer scikit-learn versions (e.g., 1.8+).
if not hasattr(AA, "_validate_data"):
    print("WARNING: Adding compatibility shim for AA._validate_data to work with newer scikit-learn versions.")
    def _validate_data(self, X, *args, **kwargs):
        return check_array(X, *args, **kwargs)
    AA._validate_data = _validate_data

def read_otu_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        sep = "\t"
    elif ext == ".csv":
        sep = ","
    else:
        sep = None  # pandas auto-detect fallback

    read_kwargs = {"index_col": "sample"}
    if sep is not None:
        read_kwargs["sep"] = sep
    else:
        read_kwargs["sep"] = None
        read_kwargs["engine"] = "python"

    try:
        otu = pd.read_csv(path, **read_kwargs)
    except Exception:
        read_kwargs["index_col"] = 0
        otu = pd.read_csv(path, **read_kwargs)
    return otu

def main(N_ARCHETYPES=3, FRAC=1, otu_file=None, out_dir="results", log_transform=True):

    print("Starting job!")
    timer = time.time()

    print("Reading Data...")
    outDir = out_dir
    os.makedirs(outDir, exist_ok=True)

    # Use provided file paths or default to existing ones
    otuFile = otu_file

    otu = read_otu_table(otuFile)
    print(otu.shape)
    
    # Drop all columns starting with "Unnamed"
    otu = otu.loc[:, ~otu.columns.str.startswith("Unnamed")]

    otu_sample = otu.sample(frac=FRAC)
    print(otu_sample.shape)
    print("Done!")
    # print the shape of the otu table
    
    print(otu_sample.shape)

    if log_transform:
        print("Log transforming...")
        pcoa_coordinates = otu_sample.map(lambda x: np.log2(x + 1))
        print("Done!")
    else:
        print("Skipping log transformation (--log_transform false).")
        pcoa_coordinates = otu_sample


    print("Performing Archetypal Analysis...")


    AA_data = pcoa_coordinates.to_numpy()

    method_kwargs = {
        "max_iter_optimizer": 10000,
    }

    init = "aa_plus_plus"
    init_kwargs = { "kwargs": {"const": 1000} }

    aa = AA(n_archetypes=N_ARCHETYPES, method="pgd", init=init, init_kwargs=init_kwargs, method_kwargs=method_kwargs, random_state=3407)


    aa.fit(AA_data)

    print(aa.archetypes_)
    
    print("Done!")


    _, info = sort_by_archetype_similarity(AA_data, [aa.similarity_degree_], aa.archetypes_)

    similarity_degree_permuted, _ = permute_dataset(aa.similarity_degree_, info["perms"])

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    simplex(aa.similarity_degree_, alpha=0.5, ax=ax[0], show_vertices=True)
    stacked_bar(similarity_degree_permuted, ax=ax[1])

    # plt.show()


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



    outName = f"otu_AA_K{N_ARCHETYPES}_{FRAC}.json"
    outPath = os.path.join(outDir, outName)

    with open(outPath, "w") as f:
        json.dump(aa_dict, f)
    # otu_aa.to_csv(outPath, sep="\t")

    print("Done!")

    print(f"Job completed for K={N_ARCHETYPES}, FRAC={FRAC}.")
    print(f"Job Completed in {time.time() - timer}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Archetypal Analysis on OTU data.")
    parser.add_argument("--n_archetypes", type=int, required=True, help="Number of archetypes.")
    parser.add_argument("--frac", type=float, default=1.0, help="Fraction of the dataset to sample.")
    parser.add_argument("--otu_file", type=str, required=True, help="Path to the OTU file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to store all results.")
    parser.add_argument(
        "--log_transform",
        type=lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y"},
        default=True,
        help="Apply log2(x+1) transform before AA (default: true). Pass false to disable.",
    )

    args = parser.parse_args()

    main(
        N_ARCHETYPES=args.n_archetypes,
        FRAC=args.frac,
        otu_file=args.otu_file,
        out_dir=args.out_dir,
        log_transform=args.log_transform,
    )
