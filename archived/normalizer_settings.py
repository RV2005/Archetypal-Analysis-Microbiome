def normalize_data(control, normalize, otu, meta, tags, projects, ignore=None, verbose=False, tol=0):
    pass
def quality_control(norm_otu, saveDir=None, verbose=False, min_taxa_level=3, min_nonzero_samples=0.001, min_counts=5000):
    pass
otu, meta, tags, projects = None, None, None, None  # Placeholder for actual data loading
# Settings for the normalized datasets created from the compendium


# ----- Normalize Region 30k ------- 
control = {"sample_type": ["stool"]} # "condition": ["healthy"], "subjects": subjects_NoCondition_Adults_only
normalize = {"region": None}
ignore = ["unknown", "Oceania"]
tol = 6.5
saveDir = r"datasets/compendium/normalized"
norm_otu, _ = normalize_data(control, normalize, otu, meta, tags, projects, ignore=ignore, verbose=True, tol=tol)
norm_otu = quality_control(norm_otu, saveDir=saveDir, verbose=True, min_taxa_level=3, min_nonzero_samples=0.001, min_counts=5000)

# ------ Normalize Region 7k -------
control = {"sample_type": ["stool"]} # "condition": ["healthy"], "subjects": subjects_NoCondition_Adults_only
normalize = {"region": None}
ignore = ["unknown", "Oceania"]
tol = 0
saveDir = r"datasets/compendium/normalized"
norm_otu, _ = normalize_data(control, normalize, otu, meta, tags, projects, ignore=ignore, verbose=True, tol=tol)
norm_otu = quality_control(norm_otu, saveDir=saveDir, verbose=True, min_taxa_level=3, min_nonzero_samples=0.001, min_counts=5000)

# ------ Normalize Sample_Type -------
control = {"sample_type": ["stool"]} # "condition": ["healthy"], "subjects": subjects_NoCondition_Adults_only
normalize = {"sample_type": None}
ignore = ["unknown", "Oceania"]
tol = 0 # 6.5 for region30k
saveDir = r"datasets/compendium/normalized"
norm_otu, _ = normalize_data(control, normalize, otu, meta, tags, projects, ignore=ignore, verbose=True, tol=tol)
norm_otu = quality_control(norm_otu, saveDir=saveDir, verbose=True, min_taxa_level=3, min_nonzero_samples=0.001, min_counts=5000)

# Normalize by Sample_Type
control = {"region": ["Europe and Northern America"]} 
normalize = {"sample_type": ['cervicovaginal swab', 'appendiceal tissue', 'stool', 'ileal mucosal sample', 'fixed tissue samples', 'esophageal biopsy', 'tumor samples']}
ignore = ["unknown"]
tol = 1.5 # 6.5 for region30k
saveDir = r"datasets/compendium/normalized"
norm_otu, _ = normalize_data(control, normalize, otu, meta, tags, projects, ignore=ignore, verbose=True, tol=tol)
norm_otu = quality_control(norm_otu, saveDir=saveDir, verbose=True, min_taxa_level=3, min_nonzero_samples=0.01, min_counts=5000)

