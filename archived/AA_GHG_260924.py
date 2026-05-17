import pandas as pd

otu = pd.read_csv("hadza_otu.tsv", sep="\t")
tax = pd.read_csv('hadza_tax.tsv', sep="\t")


# In[22]:


from tqdm import tqdm
import time

group = "Family" # Change this to "Kingdom", "Phylum", "Class", "Order", "Family", "Genus",to group by that taxonomic level

# No of uniques => {Kingdom: 1, Phylum: 15, Class: 25, Order: 44, Family: 77, Genus: 305}
grouped_cols = tax[group].unique()

# otu_grouped = pd.DataFrame(np.zeros((otu.shape[0], len(grouped_cols))), columns=grouped_cols)

otu_grouped = pd.DataFrame(index=otu.index, columns=grouped_cols)

i = 0
for c in tqdm(otu.columns):
    grp_col = tax[group][c]
    
    if(pd.isnull(otu_grouped[grp_col]).all()):
        otu_grouped[grp_col] = otu[c]
        continue

    otu_grouped[grp_col] = otu_grouped[grp_col] + otu[c]

    # print(f"Processed column ({i+1}/{len(otu.columns)}): {grp_col}")

    i += 1

otu_grouped.columns = otu_grouped.columns.astype(str)


# In[23]:


import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


pca_data = otu_grouped # Or otu itself

otu_norm = StandardScaler().fit_transform(otu)

pca = PCA(n_components=2)
pca.fit(pca_data)

otu_pca = pca.transform(pca_data)

variance_explained = sum(pca.explained_variance_ratio_) * 100

print(f"Percentage of variance explained by the first two components: {variance_explained:.2f}%")

plt.scatter(otu_pca[:, 0], otu_pca[:, 1])
plt.show()



# In[25]:


from archetypes.numpy._aa_3 import AA_3

data = otu_pca[:1000]

method_kwargs = {
    "max_iter_optimizer": 10,
}

N_ARCHETYPES = 3
aa = AA_3(n_archetypes=N_ARCHETYPES, method="pgd", method_kwargs=method_kwargs, random_state=3)


aa.fit(data)

print(aa.archetypes_)


# In[28]:


import matplotlib.pyplot as plt
from archetypes.visualization import stacked_bar, simplex
from archetypes.datasets import sort_by_archetype_similarity, permute_dataset

data_sorted, info = sort_by_archetype_similarity(data, [aa.similarity_degree_], aa.archetypes_)

similarity_degree_permuted, _ = permute_dataset(aa.similarity_degree_, info["perms"])

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
simplex(similarity_degree_permuted, alpha=0.5, ax=ax[0])
stacked_bar(similarity_degree_permuted, ax=ax[1])

plt.show()


# In[31]:


plt.scatter(otu_pca[:, 0], otu_pca[:, 1])
plt.scatter(aa.archetypes_[:, 0], aa.archetypes_[:, 1], c="red", s=25)

plt.show()

