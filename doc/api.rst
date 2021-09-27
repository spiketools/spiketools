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

Words, words, words.

Measures
~~~~~~~~

.. currentmodule:: spiketools.measures.measures
.. autosummary::
   :toctree: generated/

   compute_spike_rate
   compute_isis
   compute_cv
   compute_fano_factor

Conversions
~~~~~~~~~~~

Words, words, words.

.. currentmodule:: spiketools.measures.conversions
.. autosummary::
   :toctree: generated/

   create_spike_train
   convert_train_to_times
   convert_isis_to_spikes

Objects
-------

Words, words, words.

.. currentmodule:: spiketools.objects
.. autosummary::
   :toctree: generated/

   Cell
   Session

Spatial
-------

Words, words, words.

Occupancy
~~~~~~~~~

.. currentmodule:: spiketools.spatial.occupancy
.. autosummary::
   :toctree: generated/

    compute_spatial_bin_edges
    compute_spatial_bin_assignment
    compute_bin_width
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

Words, words, words.

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

Words, words, words.

.. currentmodule:: spiketools.sims
.. autosummary::
   :toctree: generated/

Plots
-----

Words, words, words.

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

    plot_space_heat

Stats
~~~~~

.. currentmodule:: spiketools.plts.stats
.. autosummary::
   :toctree: generated/

    plot_surrogates


Utilities
---------

Words, words, words.

.. currentmodule:: spiketools.utils
.. autosummary::
   :toctree: generated/

   restrict_range
   set_random_seed
