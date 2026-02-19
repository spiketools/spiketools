"""Simulate place cells."""

from spiketools.sim.trials import (sim_trial_placefield, sim_skew_trial_placefield,
                                sim_trial_multi_placefields, sim_trial_multi_skew_placefields)

###################################################################################################
###################################################################################################

def sim_neuron_placefield(param_gen, vary_height=False, vary_width=False, vary_place_loc=False):
    """Simulate cell place fields for a given set of parameters.

    Parameters
    ----------
    param_gen : generator
        Generator of parameters.
    vary_height : bool, optional, default=False
        Whether to vary height.
    vary_width : bool, optional, default=False
        Whether to vary width.
    vary_place_loc : bool, optional, default=False
        Whether to vary place location.

    Returns
    -------
    cell_place_bins : list
        List of cell place bins : (num_cells, num_trials, num_bins).
    """

    cell_place_bins = []
    for cur_params in param_gen:
        trial_place_bins = sim_trial_placefield(**cur_params, vary_height=vary_height,
                                                vary_width=vary_width,
                                                vary_place_loc=vary_place_loc)
        cell_place_bins.append(trial_place_bins)
    return cell_place_bins


def sim_neuron_skew_placefield(param_gen, vary_height=False,vary_width=False,
                               vary_place_loc=False, vary_skewness=True):
    """Simulate cell place fields for a given set of parameters.

    Parameters
    ----------
    param_gen : generator
        Generator of parameters.
    vary_height : bool, optional, default=False
        Whether to vary height.
    vary_width : bool, optional, default=False
        Whether to vary width.
    vary_place_loc : bool, optional, default=False
        Whether to vary place location.
    vary_skewness : bool, optional, default=True
        Whether to vary skewness.

    Returns
    -------
    cell_place_bins : list
        List of cell place bins : (num_cells, num_trials, num_bins)
    """

    cell_place_bins = []
    for cur_params in param_gen:

        trial_place_bins = sim_skew_trial_placefield(**cur_params, vary_height=vary_height,
                                                     vary_width=vary_width,
                                                     vary_place_loc=vary_place_loc,
                                                     vary_skewness=vary_skewness)
        cell_place_bins.append(trial_place_bins)
    return cell_place_bins


def sim_neuron_multi_placefield(param_gen, vary_height=False,
                                vary_width=False, vary_place_loc=False):
    """Simulate cell place fields for a given set of parameters.

    Parameters
    ----------
    param_gen : generator
        Generator of parameters.
    vary_height : bool, optional, default=False
        Whether to vary height.
    vary_width : bool, optional, default=False
        Whether to vary width.
    vary_place_loc : bool, optional, default=False
        Whether to vary place location.

    Returns
    -------
    cell_place_bins : list
        List of cell place bins : (num_cells, num_trials, num_bins).
    """

    cell_place_bins = []
    for cur_params in param_gen:

        trial_place_bins = sim_trial_multi_placefields(**cur_params, vary_height=vary_height,
                                                       vary_width=vary_width,
                                                       vary_place_loc=vary_place_loc)
        cell_place_bins.append(trial_place_bins)
    return cell_place_bins


def sim_neuron_multi_skew_placefield(param_gen, vary_height=False,
                                     vary_width=False, vary_place_loc=False, vary_skewness=False):
    """Simulate cell skew place fields for a given set of parameters.

    Parameters
    ----------
    param_gen : generator
        Generator of parameters.
    vary_height : bool, optional, default=False
        Whether to vary height.
    vary_width : bool, optional, default=False
        Whether to vary width.
    vary_place_loc : bool, optional, default=False
        Whether to vary place location.
    vary_skewness : bool, optional, default=False
        Whether to vary skewness.

    Returns
    -------
    cell_place_bins : list
        List of cell place bins : (num_cells, num_trials, num_bins).
    """

    cell_place_bins = []
    for cur_params in param_gen:
        trial_place_bins = sim_trial_multi_skew_placefields(**cur_params, vary_height=vary_height,
                                                            vary_width=vary_width,
                                                            vary_place_loc=vary_place_loc,
                                                            vary_skewness=vary_skewness)
        cell_place_bins.append(trial_place_bins)
    return cell_place_bins
