import sys
import os
import pandas as pd
import numpy as np
import scipy.spatial
from skbio.tree import TreeNode
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm

def extract_metadata_compendium(index, tags, sample_meta, projects, extract_cols = [], verbose=False):
    PROJECT_DATA_COLS = ['link', 'amplicon', 'avg_length', 'bead_beating', 'kit', 'sample_type','condition', 'subjects', 'notes']
    SAMPLE_METADATA_COLS = ['project', 'srr', 'library_strategy', 'library_source', 'pubdate','total_bases', 'instrument', 'geo_loc_name', 'iso', 'region']
    project_cols = [col for col in extract_cols if col in PROJECT_DATA_COLS]
    sample_cols =  [col for col in extract_cols if col in SAMPLE_METADATA_COLS]
    tags_cols =    [col for col in extract_cols if col not in project_cols and col not in sample_cols]
    metadata = pd.DataFrame(index=index, columns=extract_cols)
    for col in project_cols:
        metadata[col] = metadata.index.map(lambda x: projects.loc[x.split("_")[0], col] if x.split("_")[0] in projects.index and col in projects.columns else np.nan)
    for col in sample_cols:
        metadata[col] = metadata.index.map(lambda x: sample_meta.loc[x, col] if x in sample_meta.index and col in sample_meta.columns else np.nan)
    for col in tags_cols:
        metadata[col] = metadata.index.map(lambda x: tags.loc[tags['sample'] == x, col].values[0] if col in tags.columns and (tags['sample'] == x).any() else np.nan)
    return metadata

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pcoa_plotter.py <otu_file>")
        sys.exit(1)
    otuFile = sys.argv[1]
    # For Compendium Dataset
    otu = pd.read_csv(otuFile, index_col="sample")
    projects = pd.read_csv("datasets/compendium/projects.csv", index_col=0)
    meta = pd.read_table("datasets/compendium/sample_metadata.tsv", index_col=0)
    tags = pd.read_table("datasets/compendium/tags.tsv")

    pcoa_data = otu.copy()
    print(f"Before shape: {pcoa_data.shape}")
    pcoa_data = pcoa_data.map(lambda x: np.log2(x+1))
    pcoa_data = pcoa_data.replace([np.inf, -np.inf], np.nan)
    pcoa_data = pcoa_data.dropna()
    pcoa_data = pcoa_data[(pcoa_data > 0).sum(axis=1) > 0]
    print(f"After shape: {pcoa_data.shape}")

    method = scipy.spatial.distance.jensenshannon
    bdiv = beta_diversity(
        method, pcoa_data, ids=pcoa_data.index, pairwise_func=scipy.spatial.distance.pdist
    )
    print(np.isnan(bdiv.data).sum(), np.isinf(bdiv.data).sum())
    pcoa_results = pcoa(bdiv, number_of_dimensions=5)
    pcoa_coordinates = pcoa_results.samples

    def add_discrete_legend(ax, unique_colors, cmap):
        legend_patches = [
            Patch(color=cmap(unique_colors[c]), label=c)
            for c in unique_colors
        ]
        ax.legend(handles=legend_patches, title=filter_col, loc='lower left')

    def add_continuous_legend(ax, filter_labels, cmap):
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, orientation='vertical')
        cbar.set_label(f"{filter_col}: min: {str(min(filter_labels))}; max: {str(max(filter_labels))}")

    filter_col = "region"
    legend_type = "discrete" # "discrete" or "continuous"

    metadata = extract_metadata_compendium(pcoa_coordinates.index, tags, meta, projects, [filter_col])
    filter_labels = pd.Series(index=pcoa_coordinates.index, dtype=str)

    for col in [filter_col]:
        filter_labels = filter_labels.fillna(metadata.loc[pcoa_coordinates.index, col])

    if(legend_type == "discrete"):
        unique_colors = {c: i for i, c in enumerate(sorted(filter_labels.unique(), reverse=True) )}
        color = filter_labels.map(unique_colors)
        cmap = "tab20"
    elif(legend_type == "continuous"):
        unique_colors = filter_labels
        color = filter_labels.astype(float)
        cmap = "viridis"

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    pcs = [
        ('PC1', 'PC2'),
        ('PC2', 'PC3'),
        ('PC1', 'PC3')
    ]

    for i, (x_pc, y_pc) in enumerate(pcs):
        axs[i].scatter(
            pcoa_coordinates[x_pc],
            pcoa_coordinates[y_pc],
            s=10,
            c=color,
            cmap=cmap,
            alpha=0.5
        )
        axs[i].set_xlabel(f"{x_pc} ({pcoa_results.proportion_explained[int(x_pc[-1])-1]:.2%} variance explained)")
        axs[i].set_ylabel(f"{y_pc} ({pcoa_results.proportion_explained[int(y_pc[-1])-1]:.2%} variance explained)")
        axs[i].set_title(f"{x_pc} vs {y_pc}")
        if i == 0:
            if legend_type == "continuous":
                plt.colorbar(cm.ScalarMappable(cmap="viridis"), ax=axs[i], label=filter_col)
            else:
                add_discrete_legend(axs[i], unique_colors, plt.cm.get_cmap("tab20", len(unique_colors)))

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    out_png = os.path.join('results', f"pcoa_plot_{os.path.splitext(os.path.basename(otuFile))[0]}.png")
    plt.savefig(out_png)
    print(f"Plot saved to {out_png}")