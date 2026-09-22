import numpy as np

from archetypal_analysis.analysis.experiments import archetype_taxa_values


def test_taxa_values_default_to_fitted_space_without_inverse_log_rescaling():
    archetypes = np.array([[10.0, 9.0, 8.0, 1.0]])
    transform = {
        "mode": "log",
        "log_transform": True,
        "log_base": 2.0,
        "pseudocount": 1.0,
        "standard_scale": False,
    }

    fitted = archetype_taxa_values(archetypes, transform, "fitted")
    inverse = archetype_taxa_values(archetypes, transform, "inverse")

    assert np.allclose(fitted, archetypes)
    assert fitted[0, 0] / fitted.sum() < inverse[0, 0] / inverse.sum()