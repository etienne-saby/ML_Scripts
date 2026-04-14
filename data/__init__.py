"""
MetAIsAFe — data package
==========================
Data loading, preparation, diagnostics, preprocessing, and splitting.

Modules
-------
loader      : load_data(), encode_categoricals().
preparation : add_derived_columns(), filter_crops(), clean(),
              compute_effective_vars(), compute_carbon_deltas().
diagnostics : analyze_problematic_simulations(), apply_exclusions().
preprocessing : apply_winsorization(), compute_ratios_from_stocks(),
                build_dataset().
splitter    : split_by_simid(), stratified_split_by_rotation(), has_rotations().
"""
