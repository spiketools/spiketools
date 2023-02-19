.. _api_documentation:

=================
API Documentation
=================

API reference for the module.

Table of Contents
=================

.. contents::
   :local:
   :depth: 2

Measures
--------

Spike related measures and conversions, available in the `measures` sub-module.

Spikes
~~~~~~

Measures for spike data.

.. currentmodule:: spiketools.measures.spikes
.. autosummary::
   :toctree: generated/

   compute_firing_rate
   compute_isis
   compute_cv
   compute_fano_factor
   compute_presence_ratio

Conversions
~~~~~~~~~~~

Conversions between spike representations.

.. currentmodule:: spiketools.measures.conversions
.. autosummary::
   :toctree: generated/

   convert_times_to_train
   convert_train_to_times
   convert_isis_to_times
   convert_times_to_counts
   convert_times_to_rates

Circular
~~~~~~~~

Measures for working with circular data.

.. currentmodule:: spiketools.measures.circular
.. autosummary::
   :toctree: generated/

   bin_circular

Trials
~~~~~~

Measures related to trial-epoched data.

.. currentmodule:: spiketools.measures.trials
.. autosummary::
   :toctree: generated/

   compute_trial_frs
   compute_segment_frs
   compute_pre_post_rates
   compute_pre_post_averages
   compute_pre_post_diffs

Collections
~~~~~~~~~~~

Measures that can be applied to collections (groups) of neurons.

.. currentmodule:: spiketools.measures.collections
.. autosummary::
   :toctree: generated/

   detect_empty_time_ranges

Objects
-------

Object for managing spike-related data, available in the `objects` sub-module.

.. currentmodule:: spiketools.objects
.. autosummary::
   :toctree: generated/

   Unit
   Session

Spatial
-------

Functionality for processing spatial data, available in the `spatial` sub-module.

Position
~~~~~~~~

Measures and methods related to positions.

.. currentmodule:: spiketools.spatial.position
.. autosummary::
   :toctree: generated/

   compute_distance
   compute_distances
   compute_cumulative_distances
   compute_speed

Occupancy
~~~~~~~~~

Measures and methods related to computing spatial occupancy.

.. currentmodule:: spiketools.spatial.occupancy
.. autosummary::
   :toctree: generated/

   compute_bin_edges
   compute_bin_assignment
   compute_bin_counts_pos
   compute_bin_counts_assgn
   normalize_bin_counts
   create_position_df
   compute_occupancy_df
   compute_occupancy

Place
~~~~~

Measures and methods related to place cell analyses.

.. currentmodule:: spiketools.spatial.place
.. autosummary::
   :toctree: generated/

   compute_place_bins
   compute_trial_place_bins

Target
~~~~~~

Measures and methods related to spatial target cell analyses.

.. currentmodule:: spiketools.spatial.target
.. autosummary::
   :toctree: generated/

   compute_target_bins

Information
~~~~~~~~~~~

Measures and methods related to computing spatial information.

.. currentmodule:: spiketools.spatial.information
.. autosummary::
   :toctree: generated/

   compute_spatial_information

Utilities
~~~~~~~~~

Utilities related to spatial data.

.. currentmodule:: spiketools.spatial.utils
.. autosummary::
   :toctree: generated/

   compute_nbins
   compute_sample_durations
   compute_bin_width
   compute_pos_ranges
   convert_2dindices
   convert_1dindices

Statistics
----------

Statistical analyses, available in the `stats` sub-module.

Generators
~~~~~~~~~~

Statistical generators.

.. currentmodule:: spiketools.stats.generators
.. autosummary::
   :toctree: generated/

   poisson_generator

Trials
~~~~~~

Statistical measures for trial-epoched data.

.. currentmodule:: spiketools.stats.trials
.. autosummary::
   :toctree: generated/

   compute_pre_post_ttest
   compare_pre_post_activity
   compare_trial_frs

Shuffle
~~~~~~~

Methods for shuffling data.

.. currentmodule:: spiketools.stats.shuffle
.. autosummary::
   :toctree: generated/

   shuffle_spikes
   shuffle_isis
   shuffle_bins
   shuffle_poisson
   shuffle_circular

Permutations
~~~~~~~~~~~~

Methods and measures related to permutation statistics.

.. currentmodule:: spiketools.stats.permutations
.. autosummary::
   :toctree: generated/

   permute_vector
   compute_surrogate_pvalue
   compute_surrogate_zscore
   compute_surrogate_stats

ANOVA
~~~~~

Methods and measures related to computing ANOVAs.

.. currentmodule:: spiketools.stats.anova
.. autosummary::
   :toctree: generated/

   create_dataframe
   create_dataframe_bins
   fit_anova

Simulations
-----------

Functionality for simulating spiking data, available in the `sim` sub-module.

.. currentmodule:: spiketools.sim
.. autosummary::
   :toctree: generated/

   sim_spiketimes
   sim_spiketrain

Spike Times
~~~~~~~~~~~

Simulate spike times.

.. currentmodule:: spiketools.sim.times
.. autosummary::
   :toctree: generated/

   sim_spiketimes_poisson

Spike Trains
~~~~~~~~~~~~

Simulate spike trains.

.. currentmodule:: spiketools.sim.train
.. autosummary::
   :toctree: generated/

   sim_spiketrain_prob
   sim_spiketrain_binom
   sim_spiketrain_poisson

Utilities
~~~~~~~~~

Simulation related utilities.

.. currentmodule:: spiketools.sim.utils
.. autosummary::
   :toctree: generated/

   apply_refractory_times
   apply_refractory_train

Plots
-----

Functions in the `plts` sub-module for visualizing spiking data and related measures.

Spikes
~~~~~~

Plots for spike data.

.. currentmodule:: spiketools.plts.spikes
.. autosummary::
   :toctree: generated/

   plot_waveform
   plot_waveforms3d
   plot_waveform_density
   plot_isis
   plot_firing_rates

Spatial
~~~~~~~

Plots for spatial data.

.. currentmodule:: spiketools.plts.spatial
.. autosummary::
   :toctree: generated/

   plot_positions
   plot_position_by_time
   plot_heatmap
   plot_trial_heatmaps
   create_heat_title

Task
~~~~

Plots for visualizing task structure.

.. currentmodule:: spiketools.plts.task
.. autosummary::
   :toctree: generated/

   plot_task_structure

Trials
~~~~~~

Plots for trial-epoched data.

.. currentmodule:: spiketools.plts.trials
.. autosummary::
   :toctree: generated/

   plot_rasters
   plot_rate_by_time
   create_raster_title

Stats
~~~~~

Plots for statistical measures and methods.

.. currentmodule:: spiketools.plts.stats
.. autosummary::
   :toctree: generated/

   plot_surrogates

Data
~~~~

Basic plot function for different data types.

.. currentmodule:: spiketools.plts.data
.. autosummary::
   :toctree: generated/

   plot_lines
   plot_dots
   plot_points
   plot_hist
   plot_bar
   plot_polar_hist
   plot_text

Annotate
~~~~~~~~

Helper functions to annotate plot axes.

.. currentmodule:: spiketools.plts.annotate
.. autosummary::
   :toctree: generated/

   color_pvalue
   add_vlines
   add_hlines
   add_vshades
   add_hshades
   add_box_shade
   add_box_shades
   add_dots
   add_significance
   add_text_labels

Style
~~~~~

Helper functions for managing plot style.

.. currentmodule:: spiketools.plts.utils
.. autosummary::
   :toctree: generated/

   drop_spines
   invert_axes

Utils
~~~~~

Plot related utilities.

.. currentmodule:: spiketools.plts.utils
.. autosummary::
   :toctree: generated/

   check_ax
   save_figure
   make_axes
   make_grid
   get_grid_subplot

Utilities
---------

Utility functions, in the `utils` sub-module.

Base
~~~~

General utilities for basic data types.

.. currentmodule:: spiketools.utils.base
.. autosummary::
   :toctree: generated/

   flatten
   lower_list
   select_from_list
   count_elements
   combine_dicts
   add_key_prefix

Data
~~~~

Utilities for working with arrays of data.

.. currentmodule:: spiketools.utils.data
.. autosummary::
   :toctree: generated/

   compute_range
   smooth_data
   drop_nans
   assign_data_to_bins

Extract
~~~~~~~

Utilities for extracting data segments of interest.

.. currentmodule:: spiketools.utils.extract
.. autosummary::
   :toctree: generated/

   create_mask
   get_range
   get_value_range
   get_ind_by_time
   get_inds_by_times
   get_value_by_time
   get_values_by_times
   get_values_by_time_range
   threshold_spikes_by_times
   threshold_spikes_by_values
   drop_range
   reinstate_range

Timestamps
~~~~~~~~~~

Utilities for working with timestamps.

.. currentmodule:: spiketools.utils.timestamps
.. autosummary::
   :toctree: generated/

   infer_time_unit
   convert_ms_to_sec
   convert_sec_to_min
   convert_min_to_hour
   convert_ms_to_min
   convert_nsamples_to_time
   convert_time_to_nsamples
   create_bin_times
   split_time_value
   format_time_string

Epoch
~~~~~

Utilities for epoching data.

.. currentmodule:: spiketools.utils.epoch
.. autosummary::
   :toctree: generated/

   epoch_spikes_by_event
   epoch_spikes_by_range
   epoch_spikes_by_segment
   epoch_data_by_time
   epoch_data_by_event
   epoch_data_by_range
   epoch_data_by_segment

Trials
~~~~~~

Utilities for working with trial-level data.

.. currentmodule:: spiketools.utils.trials
.. autosummary::
   :toctree: generated/

   split_trials_by_condition
   split_trials_by_condition_list
   split_trials_by_condition_array
   recombine_trial_data

Run
~~~

Utilities for helping with running analyses.

.. currentmodule:: spiketools.utils.run
.. autosummary::
   :toctree: generated/

   create_methods_list

Utils
~~~~~

Other utilities.

.. currentmodule:: spiketools.utils.utils
.. autosummary::
   :toctree: generated/

   set_random_seed
   set_random_state
