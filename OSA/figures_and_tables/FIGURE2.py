import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import matplotlib as mpl

plt.style.use('nature.mplstyle')

def world_map_plot():
    world = gpd.read_file('metadata/figure2a.json')
    world['count'] = pd.Categorical(world['count'],ordered=True,
                   categories=['<100', '100-200', '200-300', '300-500', '500-1000', '1000-5000', '5000+'])
    cmap = mpl.cm.YlGnBu(np.linspace(0, 1, 15))
    cmap = mpl.colors.ListedColormap(cmap[2:-2, :-1])
    fig, ax = plt.subplot_mosaic([['a1', 'a1', 'a1','a1'],
                                    ['a1', 'a1','a1','a1'],
                                  ['a2', 'a2','a3','a3']],
                                  figsize=(6, 5))
    world.plot(column='count', ax = ax['a1'],
            cmap = cmap, legend = True,edgecolor="#737373", linewidth=0.5, missing_kwds={'color': '#d9d9d9'},
               legend_kwds={'loc': "lower left"})
    world.plot(column='count', ax = ax['a2'],
            cmap = cmap, legend = False,edgecolor="#737373", linewidth=0.5, missing_kwds={'color': '#d9d9d9'},
               legend_kwds={'loc': "lower left"})
    ax['a2'].set_xlim(-15, 35)
    ax['a2'].set_ylim(35, 75)
    ax['a2'].spines.right.set_visible(False)
    ax['a2'].spines.top.set_visible(False)
    ax['a1'].spines.right.set_visible(False)
    ax['a1'].spines.top.set_visible(False)
    ax['a1'].spines.bottom.set_visible(False)
    ax['a1'].spines.left.set_visible(False)
    ax['a1'].tick_params(  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=False,
        labelbottom=False
    )
    ax['a3'].spines.right.set_visible(False)
    ax['a3'].spines.top.set_visible(False)
    ax['a3'].spines.bottom.set_visible(False)
    ax['a3'].spines.left.set_visible(False)
    ax['a3'].tick_params(  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=False,
        labelbottom=False
    )
    ax['a1'].axhline(y=0, color='k', alpha=0.1, linestyle='--')
    ax['a1'].axhline(y=30, color='k', alpha=0.1, linestyle='--')
    ax['a1'].axhline(y=-30, color='k', alpha=0.1, linestyle='--')
    plt.subplots_adjust(left = 0.05, bottom = 0.045, right = 0.97, top = 0.97, wspace = 0, hspace = 0)
    plt.show()

def OSA_prevalence_countries():
    dfs = pd.read_csv('metadata/figure2b.csv')
    fig, axs= plt.subplots(1,2, figsize=(3.7,2.7))
    for n,ax in enumerate(axs):
        data = dfs.loc[dfs['panel']==n,:]
        ax = sns.barplot(x="mod_OSA",y="country", data=data,ax=ax,
                    label="Total", color="#829cbc", legend=False)
        ax = sns.barplot(x="sev_OSA",y="country", data=data,ax=ax,
                    color="#6290c8", legend=False)
        pos = ax.get_yticks()
        label= ax.get_yticklabels()
        ax.set_ylabel(' ')
        ax.set_yticks(pos, label)
        ax.spines.right.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.tick_params(  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=True,  # ticks along the top edge are off
            labeltop=True, labelbottom=False,
            right=False,
            left=True,
            labelleft=True)
        ax.grid(axis='x', which='major')
        ax.set_xlabel('OSA prevalence, %')
        ax.xaxis.set_label_position('top')
        ax.set_xlim(None, right=0.35)
    plt.tight_layout()
    plt.show()



OSA_prevalence_countries()

world_map_plot()
