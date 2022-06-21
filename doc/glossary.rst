Glossary
========

A glossary of terms used in the module, with brief descriptions of what they mean and how they are used, as well as a list of common abbreviations.

Spiking Data
------------

.. glossary::

    action potential / spike
        An action potential is an event in which a neuron rapidly depolarizes, initiating a cascade of events that can influence the activity of connected neurons.
        In general, we will refer to action potentials as 'spikes'.

    unit
        A general term used to refer to a collection of spikes that are putatively from the same neuron.

    spike times
        A representation of spiking activity based on listing the times at which spikes occur.

    spike train
        A representation of spiking activity in which each sample is either a 0 or 1, with 1 representing a spike.

    refractory period
        The amount of time after an action potential happens during which the same neuron cannot yet initiate another action potential.

Spike Measures
--------------

.. glossary::

    firing rate
        How fast an individual unit is firing, typically measured in Hz, as spikes per second.

    coefficient of variation
        A measure of the variability of unit firing, reflecting the dispersion of the data around the mean.

    fano factor
        A measure of the variability of unit firing, as a variant of the coefficient of variation.

    interspike interval (ISI)
        A measure of the time interval(s) between successive spikes.

    waveform
        The spike waveform is the shape of the action potential.

    raster
        A raster plot is a visualization of spike activity across time, representing each spike as a vertical line.


Recording
---------

.. glossary::

    electrode
        A recording device that can measure electrical activity, that is used to collect the raw data.

    extracellular recordings
        Recordings in which the electrode sits in the extracellular fluid, outside of any individual cells.

    micro-electrode
        An electrode that is very small, such that if it is close enough, it can record activity from single-units that are nearby.

    tetrode
        A recording device with four small electrodes that can record single unit activity.
        The multiple wires help to differentiate different units.

    sampling rate
        The rate at which samples are collected on a recording device.

    neurodata without borders
        A standardized file type for storing neuro-electrophysiological data.

Pre-Processing
--------------

.. glossary::

    spike sorting
        The process of identifying spikes and grouping them into clusters that represent putative single-neuron activity.

Measurement Units
-----------------

.. glossary::

    Hertz (Hz)
        A unit of frequency, used to measure the number of spikes per second.

Abbreviations
-------------

.. glossary::

    AP
        Action potential.

    NWB
        Neurodata without borders.

    ISIs
        Interspike intervals.

    SUA
        Single-unit activity.

    MUA
        Multi-unit activity.

    LFP
        Local field potential.

    ANOVA
        ANalysis Of VAriance.
