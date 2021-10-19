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

   compute_spike_rate
   compute_isis
   compute_cv
   compute_fano_factor

Conversions
~~~~~~~~~~~

Conversions between spike representations.

.. currentmodule:: spiketools.measures.conversions
.. autosummary::
   :toctree: generated/

   create_spike_train
   convert_train_to_times
   convert_isis_to_spikes

Objects
-------

Object for managing spike-related data, available in the `objects` sub-module.

.. currentmodule:: spiketools.objects
.. autosummary::
   :toctree: generated/

   Cell
   Session

Spatial
-------

Functionality for processing spatial data, available in the `spatial` sub-module.

Occupancy
~~~~~~~~~

.. currentmodule:: spiketools.spatial.occupancy
.. autosummary::
   :toctree: generated/

    compute_spatial_bin_edges
    compute_spatial_bin_assignment
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

    compute_spatial_information_1d
    compute_spatial_information_2d

Statistics
----------

Statistical analyses, available in the `stats` sub-module.

Generators
~~~~~~~~~~

.. currentmodule:: spiketools.stats.generators
.. autosummary::
   :toctree: generated/

    poisson_train

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

Simulations
-----------

Functionality for simulating spiking data, available in the `sims` sub-module.

.. currentmodule:: spiketools.sims
.. autosummary::
   :toctree: generated/

Plots
-----

Functions in the `plts` sub-module for visualizing spiking data and related measures.

Spikes
~~~~~~

.. currentmodule:: spiketools.plts.spikes
.. autosummary::
   :toctree: generated/

    plot_waveform
    plot_isis
    plot_firing_rates

Space
~~~~~

.. currentmodule:: spiketools.plts.space
.. autosummary::
   :toctree: generated/

    plot_positions
    plot_space_heat

Trials
~~~~~~
.. currentmodule:: spiketools.plts.trials
.. autosummary::
   :toctree: generated/

    plot_trial_rasters

Stats
~~~~~

.. currentmodule:: spiketools.plts.stats
.. autosummary::
   :toctree: generated/

    plot_surrogates


Utilities
---------

Utility functions, in the `utils` sub-module.

.. currentmodule:: spiketools.utils
.. autosummary::
   :toctree: generated/

   restrict_range
   set_random_seed
