"""Default settings for plots."""

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

# Define a list of arguments that will be caught and applied with 'set'
SET_KWARGS = ['title', 'xlim', 'ylim', 'xlabel', 'ylabel']

# Collect a list of the default matplotlib color cycle
DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
