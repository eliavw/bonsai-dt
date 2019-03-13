import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corr(corr):
    """
    Plot a nice plot of the correlation matrix

    Args:
        corr:   Correlation matrix

    Returns:

    """
    fig = plt.figure(figsize=((10, 10)))  # slightly larger
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='seismic')
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center', color='w')
    return


def plot_summary_grid(df, samples=1000, random_state=997, replace=False):
    """
    Grid of plots for overview of dataset.

    Plot a grid of plots about pairwise relationships between attributes.
    Bottom triangle: density plots
    Top triangle: scatterplots
    Diagonal: Histogram

    Explicit correlations or special distributions should be very visible here.

    Args:
        df:             pd.DataFrame
                        Data to visualize
        samples:        int, default=1000
                        One in every <samples> point is looked at.
                        For efficiency reasons.
        random_state:   int, default=997
                        For reproducibility in the subsampling. Does not matter
                        too much probably.
        replace:        Bool, default=False
                        Whether or not to sample with replacement.

    Returns:

    """

    df_subsample = df.sample(n=samples,
                             replace=replace,
                             random_state=random_state)

    sns.set(style="ticks", color_codes=True)
    g = sns.PairGrid(df_subsample)  # slice every 10 otherwise a bit slow
    g = g.map_diag(plt.hist)  # histograms on the diagonal
    g = g.map_lower(sns.kdeplot, cmap="Blues_d")  # density plot on the lower plots
    g = g.map_upper(plt.scatter)  # scatter plots on the upper plots
    return
