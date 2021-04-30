import numpy as np
import matplotlib.pyplot as plt


def plot():
    fig, ax = plt.subplots()

    prcptn = np.load('prcptn_data.npy')

    heatmap = ax.pcolor(prcptn[0], vmin=np.nanmin(prcptn), vmax=np.nanmax(prcptn))
    heatmap.cmap.set_under('black')

    bar = fig.colorbar(heatmap, extend='both')


    # want a more natural, table-like display
    ax.invert_yaxis()
    # ax.xaxis.tick_top()

    # ax.set_xticklabels(row_labels, minor=False)
    # ax.set_yticklabels(column_labels, minor=False)
    plt.show()


def plot_hatch():
    fig, ax = plt.subplots()

    prcptn = np.load('prcptn_data.npy')

    heatmap = ax.pcolor(prcptn[1], vmin=np.nanmin(prcptn), vmax=np.nanmax(prcptn))
    ax.patch.set(hatch='x', edgecolor='black')
    fig.colorbar(heatmap)


    # want a more natural, table-like display
    ax.invert_yaxis()
    # ax.xaxis.tick_top()

    plt.show()

# plot_hatch()

