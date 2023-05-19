import matplotlib.pyplot as plt

def plot_figures(figures, nrows=1, ncols=1):
    """
    Plots a dictionary of figures in a grid layout.

    Args:
        figures (dict): A dictionary containing the figures to plot, where the keys are the titles and
            the values are the corresponding images.
        nrows (int): Number of rows in the grid layout (default: 1).
        ncols (int): Number of columns in the grid layout (default: 1).

    Returns:
        matplotlib.pyplot: The matplotlib.pyplot object.

    """

    fig, axes_list = plt.subplots(ncols=ncols, nrows=nrows)

    for index, title in enumerate(figures):
        axes_list.ravel()[index].imshow(figures[title], aspect='auto')
        axes_list.ravel()[index].set_title(title)

    plt.tight_layout()
    return plt
