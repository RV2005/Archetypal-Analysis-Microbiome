"""Pipeline stage package.

Concrete stages are intentionally not imported here. The runner loads only the stages
needed for a request, so e.g. load/filter/QC do not require plotting dependencies.
"""