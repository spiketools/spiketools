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

Measures
~~~~~~~~

Spike measures.

.. currentmodule:: spiketools.measures.measures
.. autosummary::
   :toctree: generated/

   compute_firing_rate
   compute_isis
   compute_cv
   compute_fano_factor

Conversions
~~~~~~~~~~~

Conversions between spike representations.

.. currentmodule:: spiketools.measures.conversions
.. autosummary::
   :toctree: generated/

   convert_times_to_train
   convert_train_to_times
   convert_isis_to_times

Circular
~~~~~~~~

Measures for working with circular data.

.. currentmodule:: spiketools.measures.circular
.. autosummary::
   :toctree: generated/

   bin_circular

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

Occupancy
~~~~~~~~~

.. currentmodule:: spiketools.spatial.occupancy
.. autosummary::
   :toctree: generated/

    compute_nbins
    compute_bin_edges
    compute_bin_assignment
    compute_bin_firing
    normalize_bin_firing
    compute_bin_time
    compute_occupancy

Position
~~~~~~~~

.. currentmodule:: spiketools.spatial.position
.. autosummary::
   :toctree: generated/

    compute_distance
    compute_distances
    compute_cumulative_distances
    compute_speed

Information
~~~~~~~~~~~

.. currentmodule:: spiketools.spatial.information
.. autosummary::
   :toctree: generated/

    compute_spatial_information

Utilities
~~~~~~~~~

.. currentmodule:: spiketools.spatial.utils
.. autosummary::
   :toctree: generated/

    get_pos_ranges
    get_bin_width

Statistics
----------

Statistical analyses, available in the `stats` sub-module.

Generators
~~~~~~~~~~

.. currentmodule:: spiketools.stats.generators
.. autosummary::
   :toctree: generated/

    poisson_generator

Shuffle
~~~~~~~

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

.. currentmodule:: spiketools.stats.permutations
.. autosummary::
   :toctree: generated/

    vec_perm
    compute_empirical_pvalue
    zscore_to_surrogates
    compute_surrogate_stats

ANOVA
~~~~~

.. currentmodule:: spiketools.stats.anova
.. autosummary::
   :toctree: generated/

    create_dataframe
    fit_anova

Simulations
-----------

Functionality for simulating spiking data, available in the `sim` sub-module.

.. currentmodule:: spiketools.sim
.. autosummary::
   :toctree: generated/

Spike Times
~~~~~~~~~~~

.. currentmodule:: spiketools.sim.times
.. autosummary::
   :toctree: generated/

   sim_spiketimes
   sim_spiketimes_poisson

Spike Trains
~~~~~~~~~~~~

.. currentmodule:: spiketools.sim.train
.. autosummary::
   :toctree: generated/

    sim_spiketrain
    sim_spiketrain_prob
    sim_spiketrain_binom
    sim_spiketrain_poisson

Utilities
~~~~~~~~~

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

.. currentmodule:: spiketools.plts.spikes
.. autosummary::
   :toctree: generated/

    plot_waveform
    plot_waveforms3d
    plot_waveform_density
    plot_isis
    plot_firing_rates

Space
~~~~~

.. currentmodule:: spiketools.plts.spatial
.. autosummary::
   :toctree: generated/

    plot_positions
    plot_heatmap

Task
~~~~

.. currentmodule:: spiketools.plts.task
.. autosummary::
   :toctree: generated/

   plot_task_structure

Trials
~~~~~~
.. currentmodule:: spiketools.plts.trials
.. autosummary::
   :toctree: generated/

    plot_rasters
    plot_rate_by_time

Stats
~~~~~

.. currentmodule:: spiketools.plts.stats
.. autosummary::
   :toctree: generated/

    plot_surrogates

Data
~~~~

.. currentmodule:: spiketools.plts.data
.. autosummary::
   :toctree: generated/

   plot_lines
   plot_dots
   plot_bar
   plot_hist
   plot_polar_hist
   plot_text

Utilities
---------

Utility functions, in the `utils` sub-module.

Data
~~~~

.. currentmodule:: spiketools.utils.data
.. autosummary::
   :toctree: generated/

   get_range
   restrict_range
   get_value_by_time
   get_value_by_time_range

Timestamps
~~~~~~~~~~

.. currentmodule:: spiketools.utils.timestamps
.. autosummary::
   :toctree: generated/

   convert_ms_to_sec
   convert_sec_to_min
   convert_min_to_hour
   convert_ms_to_min
   split_time_value
   format_time_string

Trials
~~~~~~

.. currentmodule:: spiketools.utils.trials
.. autosummary::
   :toctree: generated/

  epoch_spikes_by_event
  epoch_spikes_by_range
  epoch_data_by_event
  epoch_data_by_range

Utils
~~~~~

.. currentmodule:: spiketools.utils.utils
.. autosummary::
   :toctree: generated/

   set_random_seed
   set_random_state
