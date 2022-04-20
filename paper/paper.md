---
title: 'spiketools: a Python package for analyzing single-unit neural activity'
tags:
  - Python
  - neuroscience
  - electrophysiology
  - single unit activity
authors:
  - name: Thomas Donoghue
    orcid: 0000-0001-5911-0472
    affiliation: "1"
  - name: Sandra Maestra
    orcid: XX
    affiliation: "1"
  - name: Zhixian Han
    orcid: XX
    affiliation: "1"
  - name: Salman Qasim
    orcid: XX
    affiliation: "1"
  - name: Joshua Jacobs
    orcid: XX
    affiliation: "1"
affiliations:
 - name: Department of Biomedical Engineering, Columbia University
   index: 1
date: XX XX XX
bibliography: paper.bib
---

# Summary

A common method of collecting and analyzing neural activity is to implant electrodes that record the electrical activity of the brain, from which action potentials of individual neurons can be recorded [@buzsaki_origin_2012]. After pre-processing to detect spike waveforms and cluster them into groups representing putative single neurons [@rey_past_2015], this data can be used to investigate how neurons in the brain encode and process information. Analyzing single-unit activity requires dedicated analysis approaches, including representing spiking activity as spike times and/or binary spike trains, and analysis tools that allow for associating this activity to features of interest, for example the position of the animal in space or the properties of presented visual stimuli. To assist in this process, ``spiketools`` is a package designed to be used by neuroscientists for analyzing spiking activity.

``spiketools`` is accompanied by a [documentation site](https://spiketools.github.io/) that includes detailed [tutorials](https://spiketools.github.io/spiketools/auto_tutorials/index.html) for each of the modules, which are described below, as well as suggested workflows for combining them.

Modules in ``spiketools`` include:

* measures : measures and conversions that can be applied to spiking data
* objects : objects that can be used to manage spiking data
* spatial : space related functionality and measures
* stats : statistical measures for analyzing spiking data
* sim : simulations of spiking activity and related functionality
* plts : plotting functions for visualizing spike data and related measures
* utils : additional utilities for working with spiking data

``spiketools`` is built on existing tools in the scientific Python ecosystem, and has the following dependencies:

* numpy : which is used for managing and computing with arrays [@harris_array_2020]
* scipy : which is used for some existing algorithms [@virtanen_scipy_2020]
* pandas : which is used for managing heterogeneous data [@mckinney_pandas_2011]
* matplotlib : which is used for plotting [@hunter_matplotlib_2007]

# Statement of Need

``spiketools`` is an open-source Python package for analyzing spiking neural data. Spiking neural activity is an idiosyncratic data stream with specific properties that requires specialized analysis tools and dedicated algorithms and statistical tools. Despite the popularity of this kind of data, there is currently a lack of openly available and maintained tools for this kind of data, especially within the Python ecosystem. ``spiketools`` therefore fills a niche, leveraging the power of the scientific power ecosystem, while providing dedicated implementations for the specific requirements of spiking data.

``spiketools`` complements related tools that support other functionality in the ecosystem, including `neo` [@neo14], which supports loading and working with electrophysiological data, and `spike interface` [@buccino_spikeinterface_2020], which implements and supports spike-sorting related functionality. ``spiketools`` also offers a module for simulating spiking activity, which is used for testing the properties of methods against synthetic data for which ground truth parameters are known. Note that these simulations are designed to mimic the statistics of single unit spiking activity, but are not designed to investigate biophysical properties of neurons. Nevertheless, this

Benefits of ``spiketools`` include that it follows modular organization, includes a test suite, follows a release cycle with versioned updates, and includes documentation and tutorials. ``spiketools`` is designed with a lightweight architecture in which functions take in arrays of spike times or spike times, thus offering a flexible toolbox for custom analysis of spiking data. This approach also makes the tool flexible to be able to be integrated into existing codebases and workflows that use other tools.

# Acknowledgments

We thank ....

# References