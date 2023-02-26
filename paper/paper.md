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
  - name: Sandra Maesta-Pereira
    orcid: 0000-0001-6522-8311
    affiliation: "1"
  - name: Claire Zhixian Han
    orcid: 0000-0001-9710-8381
    affiliation: "1"
  - name: Salman Ehtesham Qasim
    orcid: 0000-0001-8739-5962
    affiliation: "2"
  - name: Joshua Jacobs
    orcid: 0000-0003-1807-6882
    affiliation: "1"
affiliations:
  - name: Department of Biomedical Engineering, Columbia University
    index: 1
  - name: Department of Psychiatry. Icahn School of Medicine at Mount Sinai
    index: 2
  - name: Department of Neurological Surgery, Columbia University
    index: 3
date: XX XX XX
bibliography: paper.bib
---

# Summary

A common method of collecting and analyzing neural activity is to implant electrodes that record the electrical activity of the brain, from which action potentials of individual neurons can be recorded [@buzsaki_origin_2012]. After pre-processing to detect spike waveforms and cluster them into groups representing putative single neurons [@rey_past_2015], this data can be used to investigate how neurons in the brain encode and process information. Analyzing single-unit activity requires dedicated analysis approaches, including representing spiking activity as spike times and/or binary spike trains, and analysis tools that allow for associating this activity to features of interest, for example the position of the animal in space or the properties of presented visual stimuli. To assist in this process, ``spiketools`` is a package designed to be used by neuroscientists for analyzing spiking activity.

``spiketools`` is written in the Python programming language, built on existing tools in the scientific Python ecosystem, and developed using best-practice procedures. The module is accompanied by a [documentation site](https://spiketools.github.io/) that includes detailed [tutorials](https://spiketools.github.io/spiketools/auto_tutorials/index.html) for each of the modules, which are described below, as well as suggested workflows for combining them.

Modules in ``spiketools`` include:

* measures : measures and conversions that can be applied to spiking data
* objects : objects that can be used to manage spiking data
* spatial : space related functionality and measures
* stats : statistical measures for analyzing spiking data
* sim : simulations of spiking activity and related functionality
* plts : plotting functions for visualizing spike data and related measures
* utils : additional utilities for working with spiking data

``spiketools`` is has the following required dependencies:

* numpy : used for managing and computing with arrays [@harris_array_2020]
* scipy : used for some existing algorithms [@virtanen_scipy_2020]
* pandas : used for managing heterogeneous data [@mckinney_pandas_2011]
* matplotlib : used for plotting [@hunter_matplotlib_2007]

``spiketools`` also has some optional dependencies that offer extra functionality:

* statsmodels : used for statistically testing responses [@seabold_statsmodels_2010]

# Statement of Need

``spiketools`` is an open-source Python package for analyzing spiking neural data. Spiking neural activity is an idiosyncratic data stream with specific properties that requires specialized analysis tools and dedicated algorithms and statistical tools. Despite the popularity of this kind of data, there is currently a lack of openly available and maintained tools for this kind of data, especially within the Python ecosystem. ``spiketools`` therefore fills a niche, leveraging the power of the scientific power ecosystem, while providing dedicated implementations for the specific requirements of spiking data.

Benefits of ``spiketools`` include that it follows modular organization, includes a test suite, follows a release cycle with versioned updates, and includes documentation and tutorials. ``spiketools`` is designed with a lightweight architecture in which functions take in arrays of spike times or spike trains, thus offering a flexible toolbox for custom analysis of spiking data. This approach also makes the tool flexible to be able to be integrated into existing codebases and workflows that use other tools.

``spiketools`` complements related tools that support other functionality in the ecosystem, including `neo` [@garcia_neo_2014], which supports loading and working with electrophysiological data, and `spike interface` [@buccino_spikeinterface_2020], which implements and supports spike-sorting related functionality. It's lightweight architecture also allows for integration with other related tools, for example, it could be used in combination with `NeuroDSP` [@cole_neurodsp_2019], which provides functionality for analyzing neural time series, in order to examine relationships between spiking activity and the local field potential.

``spiketools`` also offers a module for simulating spiking activity, offering several methods for simulating spiking activity with specified parameters. Note that these simulations are designed to mimic the statistics of single unit spiking activity, but are not designed to replicate or reflect biophysical properties of neurons, and therefore should not be over-interpreted as biophysically realistic. Nevertheless, this simulation system allows for method testing, as new methods and implementations can be tested against synthetic data for which ground truth parameters are known.

# Acknowledgments

We thank like to thank the Jacobs Lab for useful discussions throughout the development of this toolbox. This work was supported by the National Institute of Health (NIH) grants U01-NS121472 and 2R01-MH104606, as well as funding from the National Science Foundation (NSF).

# References