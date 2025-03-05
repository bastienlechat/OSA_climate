import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./nature.mplstyle')
wdatemp = pd.read_csv('metadata/figure1a.csv')

fig, (ax,ax2) = plt.subplots(1,2)
ax.plot(wdatemp['day_of_year'].values, wdatemp['t2m_mean'].values-273.15, label='2023', color='k', alpha= 1)

ax.plot(wdatemp['day_of_year'].values, wdatemp['t2m_historical'].values, label='1950 to 1990',
         linestyle='--', color='r', alpha=0.5)
y2 = wdatemp['t2m_mean'].values-273.15
y1 = wdatemp['t2m_historical'].values
ax.fill_between(x=wdatemp['day_of_year'].values, y1 = y1,
                 y2 = y2, where=y2 >= y1, alpha=0.2, color='k', label='Global warming')
ax.set_xticks(np.linspace(0,365,13)[:-1] + 15, ('Jan', ' ','March',' ','May',' ','July',' ','Sep',' ','Nov',' '))

ax.grid()
ax.legend(loc='upper left')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_visible(False)
ax.spines.left.set_visible(False)
ax.tick_params( # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    )
ax.set_ylabel('Temperature (C)')

wsalogit = pd.read_csv('metadata/figure1b.csv')

ax2.plot(wsalogit['xvar'].values, wsalogit['est'].values, color='#606c38', label='WSA')
ax2.fill_between(wsalogit['xvar'].values, y1=wsalogit['lb'].values, y2=wsalogit['hb'].values, color='#606c38',
                      alpha=0.2)

ax2.grid()
ax2.legend(loc='upper left')
ax2.spines.right.set_visible(False)
ax2.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)
ax2.spines.left.set_visible(False)
ax2.tick_params( # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    )
ax2.set_xlabel('Temperature (C)')
ax2.set_ylabel('Risk ratio of OSA')
plt.show()