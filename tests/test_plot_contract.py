from pathlib import Path


def test_pipeline_no_longer_emits_separate_taxa_bars_png():
    root = Path(__file__).resolve().parents[1]
    source = (root / "archetypal_analysis/pipeline/stages/plot.py").read_text()
    assert 'taxa_bars.{ext}' not in source
    assert 'taxa_stacked_bar.{ext}' in source
    assert 'taxa_heatmap.{ext}' in source
    assert 'taxa_heatmap.tsv' in source
    assert 'visual_top_taxa.tsv' in source
    assert 'summary.{ext}' in source


def test_plot_stage_records_embedding_projection_methods():
    root = Path(__file__).resolve().parents[1]
    source = (root / "archetypal_analysis/pipeline/stages/plot.py").read_text()
    assert 'embedding_projection_methods' in source
    assert 'resolved_archetype_orders' in source
    assert 'visual_archetype_labels' in source


def test_plot_stage_relabels_ordered_archetypes_for_visualization():
    root = Path(__file__).resolve().parents[1]
    source = (root / "archetypal_analysis/pipeline/stages/plot.py").read_text()
    assert "relabel_result_for_visual_order" in source
    assert "visual_results[k]" in source
    assert "display_orders[k]" in source
    assert "display_name_maps[k]" in source


def test_stacked_similarity_legend_uses_explicit_archetype_order():
    root = Path(__file__).resolve().parents[1]
    source = (root / "archetypal_analysis/visualization/stacked_bar.py").read_text()
    assert "legend_patches = [" in source
    assert "Patch(color=colors[i], label=vertices_labels[i])" in source
    assert "ax.legend(handles=legend_patches" in source


def test_projected_embedding_marker_options_are_configurable():
    root = Path(__file__).resolve().parents[1]
    source = (root / "archetypal_analysis/visualization/pipeline_plots.py").read_text()
    assert "numeric_marker" in source
    assert "marker_mode" in source
    assert "marker_legend" in source
    assert "fig.legend(" in source


def test_taxa_stacked_plot_uses_explicit_taxon_colors_and_visual_table():
    root = Path(__file__).resolve().parents[1]
    source = (root / "archetypal_analysis/visualization/pipeline_plots.py").read_text()
    assert "def _taxa_colors" in source
    assert "color=colors[column]" in source
    assert "def visual_top_taxa_frame" in source
    assert "archetype_taxa_values" in source
    assert "Percentage Contribution (%)" in source
    assert "_simplify_taxon(features[int(idx)], level, max_len).lower()" in source


def test_k_summary_supports_taxa_heatmap_and_row_width_ratios():
    root = Path(__file__).resolve().parents[1]
    source = (root / "archetypal_analysis/visualization/pipeline_plots.py").read_text()
    defaults = (root / "configs/defaults.yaml").read_text()
    assert "top_width_ratios" in source
    assert "bottom_width_ratios" in source
    assert "taxa_panel" in source
    assert "_draw_taxa_heatmap" in source
    assert "top_width_ratios:" in defaults
    assert "bottom_width_ratios:" in defaults
    assert "taxa_panel:" in defaults


def test_pcoa_axes_display_pico_labels():
    root = Path(__file__).resolve().parents[1]
    pcoa_source = (root / "archetypal_analysis/visualization/pcoa.py").read_text()
    pipeline_source = (root / "archetypal_analysis/visualization/pipeline_plots.py").read_text()
    assert "PiCo {match.group(1)}" in pcoa_source
    assert "set_title(f\"{x_label} vs {y_label}\")" in pcoa_source
    assert "PiCo {match.group(1)}" in pipeline_source
    assert "set_title(f\"{x_label} vs {y_label}\")" in pipeline_source
