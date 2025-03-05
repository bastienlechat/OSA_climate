import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
import numpy as np
import warnings
import seaborn as sns
import mpl_toolkits.axisartist as axisartist
warnings.filterwarnings("ignore")

np.random.seed(seed=0)
plt.style.use('./nature.mplstyle')

def plot_test(hist, ssps):
    fig, ax = plt.subplot_mosaic([['a1', 'a0','m'],
                                  ['a1', 'a0','m'],
                                  ['a1', 'a0','m']],
                                 # layout='constrained',
                                 #width_ratios=[2 / 5, 2 / 5, 1 / 5],
                                 figsize=(3.8, 5))
    ssps = ssps.rename(columns={'climate_absenteism_mean_lowess_corrected': 'climate_absenteism_mean_lowess',
                                'climate_presenteism_mean_lowess_corrected': 'climate_presenteism_mean_lowess',
                                'labor_loss_mean_lowess_corrected':'labor_loss_mean_lowess'})
    df = pd.concat([hist, ssps], join='inner', ignore_index=True)
    df = df.loc[~df['country'].isin(
        ['Austria', 'Brazil', 'China', 'Croatia', 'India', 'Israel', 'Korea, South', 'New Zealand', 'Singapore',
         'Thailand', 'Turkey', 'United Arab Emirates',
         ]), :]

    df['abs_per'] = 100000 * df['climate_absenteism_mean_lowess'] / df['lbf2023']
    df['pres_per'] = 100000 * df['climate_presenteism_mean_lowess'] / df['lbf2023']
    df['money_per'] = (100000 * df['labor_loss_mean_lowess'] / df['lbf2023']) /1e6

    f = df.loc[df['year'].isin([2023,2098])]

    palette = {'historical': '#073b4c',
               'ssp126': '#003049',
               'ssp245': '#fcbf49',
               'ssp370': '#f77f00',
               'ssp585': '#d62828'}
    # Draw a pointplot to show pulse as a function of three categorical factors

    f = f.loc[f['city'] != 'Hong Kong (China)']
    f = f.replace({'United Kingdom': 'UK',
                               'Korea, South': 'S.Korea',
                               'United Arab Emirates': 'U.A.E',
                               'New Zealand': 'N.Z',
                               'United States': 'USA',
                                'Czech Republic': 'Czechia'})

    f = f.sort_values(by=['pres_per','city'], ascending=False)

    hst = f.loc[f['year'].isin([2023])]
    f = f.loc[f['year'].isin([2098])]

    fa = f.loc[f['country']=='Australia',:]
    fa.to_csv('testing.csv')

    palette = {'historical': '#073b4c',
               'ssp126': '#003049',
               'ssp245': '#fcbf49',
               'ssp370': '#f77f00',
               'ssp585': '#d62828'}

    ssp585 = f.loc[f['exp'] == 'ssp585',:]
    hist = hst.loc[hst['exp'] == 'historical', :]
    ax['a0'].axvspan(xmin=240, xmax=300, color='grey', alpha=0.2)
    sns.barplot(x="abs_per", y="country", data=ssp585, ax=ax['a0'],
                     color=palette['ssp585'], legend=False, alpha=0.8)

    sns.barplot(x="pres_per", y="country", data=ssp585, ax=ax['a1'],
                     color=palette['ssp585'], legend=False, alpha=0.8)

    sns.barplot(x="money_per", y="country", data=ssp585, ax=ax['m'],
                     color=palette['ssp585'], legend=False, alpha=0.8)

    ssp370 = f.loc[f['exp'] == 'ssp370',:]
    sns.barplot(x="abs_per", y="country", data=ssp370, ax=ax['a0'],
                     color=palette['ssp370'], legend=False)
    sns.barplot(x="pres_per", y="country", data=ssp370, ax=ax['a1'],
                     color=palette['ssp370'], legend=False)
    sns.barplot(x="money_per", y="country", data=ssp370, ax=ax['m'],
                     color=palette['ssp370'], legend=False)


    ssp245 = f.loc[f['exp'] == 'ssp245',:]
    sns.barplot(x="abs_per", y="country", data=ssp245, ax=ax['a0'],
                     color=palette['ssp245'], legend=False)
    sns.barplot(x="pres_per", y="country", data=ssp245, ax=ax['a1'],
                     color=palette['ssp245'], legend=False)
    sns.barplot(x="money_per", y="country", data=ssp245, ax=ax['m'],
                     color=palette['ssp245'], legend=False)

    exp = f.loc[f['exp']=='ssp126',:]
    sns.barplot(x="abs_per", y="country", data=exp, ax=ax['a0'],
                     label="Total", color=palette['ssp126'], legend=False)
    sns.barplot(x="pres_per", y="country", data=exp, ax=ax['a1'],
                     color=palette['ssp126'], legend=False)
    sns.barplot(x="money_per", y="country", data=exp, ax=ax['m'],
                     color=palette['ssp126'], legend=False)
    pos = ax['a0'].get_yticks()
    label = ax['a0'].get_yticklabels()
    ax['a0'].set_ylabel(' ')
    ax['a0'].set_yticks(pos, label)
    ax['a0'].spines.right.set_visible(False)
    ax['a0'].spines.bottom.set_visible(False)


    ax['a0'].tick_params(  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labeltop=True, labelbottom=False,
        right=False,
        left=True,
        labelleft=True)
    ax['a0'].grid(axis='x', which='major')
    #ax['a0'].set_xlabel('OSA prevalence, %')
    #ax['a0'].xaxis.set_label_position('top')


    ax['a1'].set_ylabel('')
    ax['a1'].invert_xaxis()

    ax['a1'].grid(axis='x', which='major')
    ax['a1'].spines.right.set_visible(True)
    ax['a1'].spines.left.set_visible(False)
    ax['a1'].spines.bottom.set_visible(False)
    ax['a1'].tick_params(  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labeltop=True, labelbottom=False,
        right=True,
        left=False,
        labelleft=False)

    ax['m'].invert_xaxis()
    ax['m'].set_ylabel('')
    ax['m'].set_xlabel('Cost (millions USD) \n per 100,000/year')
    ax['m'].grid(axis='x', which='major')
    ax['m'].spines.right.set_visible(False)
    ax['m'].spines.left.set_visible(False)
    ax['m'].spines.bottom.set_visible(False)
    ax['m'].tick_params(  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labeltop=True, labelbottom=False,
        right=False,
        left=False,
        labelleft=False)

    ax['a1'].set_xlabel('Absenteeism days \n per 100,000 persons/year')
    ax['a1'].xaxis.set_label_position('top')
    ax['a0'].set_xlabel('Presenteism days \n per 100,000 persons/year')
    ax['a0'].xaxis.set_label_position('top')
    ax['m'].set_xlabel('Labor loss (millions USD) \n per 100,000 persons/year')
    ax['m'].xaxis.set_label_position('top')
    plt.show()

def plot_figures(hist, ssps):

    fig, ax = plt.subplot_mosaic([['a0', 'a0', 'a0'],
                                    ['a1', 'a1', 'a1'],
                                    ['a2', 'a2','a2'],
                                  ['b1', 'b2', 'b3']],
                                  #layout='constrained',
                                height_ratios = [1/10,3/10,3/10,3/10],
                                  figsize=(3.8, 5))

    ax['a0'].spines.right.set_visible(False)
    ax['a0'].spines.top.set_visible(False)
    ax['a0'].spines.bottom.set_visible(False)
    ax['a0'].spines.left.set_visible(False)
    ax['a0'].tick_params(  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=False,
        labelbottom=False,
    )
    ssps = ssps.rename(columns={'climate_absenteism_mean_lowess_corrected': 'climate_absenteism_mean_lowess',
                                'climate_presenteism_mean_lowess_corrected': 'climate_presenteism_mean_lowess',
                                'labor_loss_mean_lowess_corrected': 'labor_loss_mean_lowess'})

    df = pd.concat([hist, ssps], join='inner', ignore_index=True)
    df = df.loc[~df['country'].isin(
        ['Austria', 'Brazil', 'China', 'Croatia', 'India', 'Israel', 'Korea, South', 'New Zealand', 'Singapore',
         'Thailand', 'Turkey', 'United Arab Emirates',
         ]), :]

    df['abs_per'] = 100000 * df['climate_absenteism_mean_lowess'] / df['lbf2023']
    df['pres_per'] = 100000 * df['climate_presenteism_mean_lowess'] / df['lbf2023']
    df['money_per'] = (100000 * df['labor_loss_mean_lowess'] / df['lbf2023']) / 1e6

    f = df.loc[df['year'].isin([2023, 2098])]

    palette = {'historical': '#073b4c',
               'ssp126': '#003049',
               'ssp245': '#fcbf49',
               'ssp370': '#f77f00',
               'ssp585': '#d62828'}
    # Draw a pointplot to show pulse as a function of three categorical factors

    f = df.loc[df['year'].isin([1950, 2000, 2023, 2050, 2098])]

    palette = {'historical': '#073b4c',
               'ssp126': '#003049',
               'ssp245': '#fcbf49',
               'ssp370': '#f77f00',
               'ssp585': '#d62828'}
    # Draw a pointplot to show pulse as a function of three categorical factors
    sns.pointplot(
        data=f, x="year", y="money_per", hue="exp",
        palette=palette, ax=ax['a1'], legend=False,
        dodge=0.2,
    )
    ax['a1'].grid(which='major', axis='y', linestyle='--')
    ax['a1'].set_ylabel('')

    ax['a1'].set_title('Labor loss per 100,000 \n (millions USD)', loc='left', y=1.0, pad=-20, fontsize=8)

    ax['a1'].spines.right.set_visible(False)
    ax['a1'].spines.top.set_visible(False)
    ax['a1'].spines.bottom.set_visible(False)
    ax['a1'].spines.left.set_visible(False)
    ax['a1'].tick_params(  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=True,
    )
    ax['a1'].set_xlabel('')

    dft = df.groupby(['year', 'exp'])[
        ['climate_absenteism_mean_lowess', 'climate_presenteism_mean_lowess',
         'labor_loss_mean_lowess']].sum().reset_index()

    print(dft)

    dft['labor_loss_mean_lowess'] = dft['labor_loss_mean_lowess']/ 1e9

    # Draw a pointplot to show pulse as a function of three categorical factors
    ax['a2'] = sns.lineplot(
        data=dft, x="year", y="labor_loss_mean_lowess", hue="exp",
        palette=palette, ax=ax['a2'], legend=False,
    )
    ax['a2'].grid(which='both', axis='y', linestyle='--')
    ax['a2'].set_ylabel('')

    ax['a2'].set_title('Labor loss per year \n (billions USD)', loc='left', y=1.0, pad=-12, fontsize=8)
    ax['a2'].set_xlabel('')

    ax['a2'].spines.right.set_visible(False)
    ax['a2'].spines.top.set_visible(False)
    ax['a2'].spines.bottom.set_visible(False)
    ax['a2'].spines.left.set_visible(False)
    ax['a2'].tick_params(  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=True,
    )

    dfscenarios = df.loc[df['year'].values>2024,:]
    dfeveryone = dfscenarios.groupby(['year','exp'])[['climate_absenteism_mean_lowess', 'climate_presenteism_mean_lowess', 'labor_loss_mean_lowess']].sum().reset_index()

    dfeveryone['cumsum_ABS'] = dfeveryone.groupby('exp')['climate_absenteism_mean_lowess'].transform('cumsum')/1e6
    dfeveryone['cumsum_PRES'] = dfeveryone.groupby('exp')['climate_presenteism_mean_lowess'].transform('cumsum')/1e6
    dfeveryone['cumsum_labor'] = dfeveryone.groupby('exp')['labor_loss_mean_lowess'].transform('cumsum')/1e12

    #ax['b1']. x="idx", y="cumsum", hue="hue_val", data=df

    sns.lineplot(data=dfeveryone, x="year", y="cumsum_ABS", hue="exp",
        palette=palette, ax=ax['b1'], legend=False)
    ax['b1'].set_ylabel('ABS (millions)')

    sns.lineplot(data=dfeveryone, x="year", y="cumsum_PRES", hue="exp",
        palette=palette, ax=ax['b2'], legend=False)
    ax['b2'].set_ylabel('PRES (millions)')

    sns.lineplot(data=dfeveryone, x="year", y="cumsum_labor", hue="exp",
        palette=palette, ax=ax['b3'], legend=False)
    ax['b3'].set_ylabel('Cost (trillions)')
    #sns.lineplot(x="idx", y="cumsum", hue="hue_val", data=df)

    for a in [ax['b1'], ax['b2'], ax['b3']]:
        a.grid(which='major', axis='y', linestyle='--')
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        a.spines.bottom.set_visible(False)
        a.spines.left.set_visible(False)
        a.tick_params(  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            labelleft=True,
        )

    plt.subplots_adjust(bottom=0.08,wspace=0.65, hspace=0.505, right=0.97, left=0.12, top=0.95)
    print(dfeveryone.loc[dfeveryone['year'].values==2098,:])

    plt.show()

def ssp_table(dhist, year = 2050, correction=1e3):
    clist = list(dhist['country'].value_counts().index)

    daly_table = {}
    daly_table_numbers = {}
    print(dhist.keys())

    dhist = dhist.rename(columns={'climate_absenteism_mean_lowess_corrected': 'climate_absenteism_mean_lowess',
                                'climate_presenteism_mean_lowess_corrected': 'climate_presenteism_mean_lowess',
                                'labor_loss_mean_lowess_corrected': 'labor_loss_mean_lowess',
                               'climate_absenteism_min_lowess_corrected': 'climate_absenteism_min_lowess',
                               'climate_presenteism_min_lowess_corrected': 'climate_presenteism_min_lowess',
                               'labor_loss_min_lowess_corrected': 'labor_loss_min_lowess'
                                ,'climate_absenteism_max_lowess_corrected': 'climate_absenteism_max_lowess',
                                                                'climate_presenteism_max_lowess_corrected': 'climate_presenteism_max_lowess',
                                                                'labor_loss_max_lowess_corrected': 'labor_loss_max_lowess'                               })

    #df['abs_per'] = 100000 * df['climate_absenteism_mean_lowess'] / df['lbf2023']
    #df['pres_per'] = 100000 * df['climate_presenteism_mean_lowess'] / df['lbf2023']
    #df['money_per'] = (100000 * df['labor_loss_mean_lowess'] / df['lbf2023']) / 1e6
    for var in ['labor_loss_mean_lowess','labor_loss_min_lowess','labor_loss_max_lowess',
                'climate_absenteism_mean_lowess', 'climate_absenteism_min_lowess', 'climate_absenteism_max_lowess',
                'climate_presenteism_mean_lowess','climate_presenteism_min_lowess','climate_presenteism_max_lowess']:
        dhist[var] = dhist[var] /correction

    for cname in clist:

        historical = dhist.loc[dhist['country'] == cname]

        vartable = {}
        vartablenumbers = {}
        vars = []
        for var in [['climate_presenteism_mean_lowess','climate_presenteism_min_lowess','climate_presenteism_max_lowess'],
                    ['climate_absenteism_mean_lowess', 'climate_absenteism_min_lowess', 'climate_absenteism_max_lowess'],
                    ['labor_loss_mean_lowess','labor_loss_min_lowess','labor_loss_max_lowess']]:

            meanvar = var[0]
            minvar = var[1]
            maxvar = var[2]
            # print(historical)
            pop = historical.loc[historical['year'].isin([year]), 'lbf2023'].values[0]
            v = historical.loc[historical['year'].isin([year]), meanvar].values[0]
            vmin = historical.loc[historical['year'].isin([year]), minvar].values[0]
            vmax = historical.loc[historical['year'].isin([year]), maxvar].values[0]
            vper = 100000 * v / int(pop)
            vpermin = 100000 * vmin / int(pop)
            vpermax = 100000 * vmax / int(pop)
            dalyt = '{:.0f} \n ({:.0f}, {:.0f})'.format(v,
                                                        vmin,
                                                        vmax)

            if var[0] in ['labor_loss_mean_lowess','labor_loss_min_lowess','labor_loss_max_lowess']:
                yld_pert = '{:.1f} \n ({:.1f}, {:.1f})'.format(vper,
                                                               vpermin,
                                                               vpermax)
            else:
                yld_pert = '{:.0f} \n ({:.0f}, {:.0f})'.format(vper,
                                                               vpermin,
                                                               vpermax)

            vartable[var[0]+'_c'] = dalyt
            vartable[var[0] + '_per'] = yld_pert

            vartablenumbers[var[0]+'_c_mean'] = v
            vartablenumbers[var[0] + '_c_min'] = vmin
            vartablenumbers[var[0] + '_c_max'] = vmax
            vartablenumbers[var[0] + '_per_mean'] = vper
            vartablenumbers[var[0] + '_per_min'] = vpermin
            vartablenumbers[var[0] + '_per_max'] = vpermax

        daly_table[cname] = vartable
        daly_table_numbers[cname] = vartablenumbers

    data =  pd.DataFrame.from_dict(daly_table_numbers, orient='index')
    data['country'] = data.index
    data = data.reset_index(drop=True)
    data = data.sort_values(by='country')


    data = data.loc[~data['country'].isin(
        ['Austria', 'Brazil', 'China', 'Croatia', 'India', 'Israel', 'Korea, South', 'New Zealand', 'Singapore',
         'Thailand', 'Turkey', 'United Arab Emirates',
         ]), :]


    climate_presenteism_mean_lowess_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['climate_presenteism_mean_lowess_c_mean'].sum(), data['climate_presenteism_mean_lowess_c_min'].sum(), data['climate_presenteism_mean_lowess_c_max'].sum())
    climate_presenteism_mean_lowess_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['climate_presenteism_mean_lowess_per_mean'].mean(), data['climate_presenteism_mean_lowess_per_min'].mean(),
                                                 data['climate_presenteism_mean_lowess_per_max'].mean())

    climate_absenteism_mean_lowess_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['climate_absenteism_mean_lowess_c_mean'].sum(), data['climate_absenteism_mean_lowess_c_min'].sum(), data['climate_absenteism_mean_lowess_c_max'].sum())
    climate_absenteism_mean_lowess_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['climate_absenteism_mean_lowess_per_mean'].mean(), data['climate_absenteism_mean_lowess_per_min'].mean(),
                                                 data['climate_absenteism_mean_lowess_per_max'].mean())

    labor_loss_mean_lowess_c = '{:.1f} \n ({:.1f}, {:.1f})'.format(data['labor_loss_mean_lowess_c_mean'].sum(), data['labor_loss_mean_lowess_c_min'].sum(), data['labor_loss_mean_lowess_c_max'].sum())
    labor_loss_mean_lowess_per = '{:.1f} \n ({:.1f}, {:.1f})'.format(data['labor_loss_mean_lowess_per_mean'].mean(), data['labor_loss_mean_lowess_per_min'].mean(),
                                                 data['labor_loss_mean_lowess_per_max'].mean())

    daly_table['total'] = {'climate_presenteism_mean_lowess_c':climate_presenteism_mean_lowess_c,
                           'climate_presenteism_mean_lowess_per':climate_presenteism_mean_lowess_per,
                           'climate_absenteism_mean_lowess_c':climate_absenteism_mean_lowess_c,
                           'climate_absenteism_mean_lowess_per':climate_absenteism_mean_lowess_per,
                           'labor_loss_mean_lowess_c':labor_loss_mean_lowess_c,
                           'labor_loss_mean_lowess_per':labor_loss_mean_lowess_per}

    d = pd.DataFrame.from_dict(daly_table, orient='index')
    d['country'] = d.index
    d = d.reset_index(drop=True)
    d = d.sort_values(by='country')
    return d

def t2023_table(dhist, year = 2023):
    clist = list(dhist['country'].value_counts().index)
    hists = []
    daly_table = {}
    daly_table_numbers = {}
    for cname in clist:

        historical = dhist.loc[dhist['country'] == cname]
        historical['daly'] = historical['yld_prediction_mean'] + historical['yll_prediction_mean']
        historical['daly_min'] = historical['yll_prediction_min'] + historical['yld_prediction_min']
        historical['daly_max'] = historical['yll_prediction_max'] + historical['yld_prediction_max']

        historical['money_cost_corrected'] = historical['money_cost_corrected']/1e6
        historical['money_cost_min_corrected'] = historical['money_cost_min_corrected']/1e6
        historical['money_cost_max_corrected'] = historical['money_cost_max_corrected']/1e6

        vartable = {}
        vartablenumbers = {}
        vars = []
        for var in [['daly','daly_min','daly_max'], ['yll_prediction_mean','yll_prediction_min','yll_prediction_max'],
                    ['yld_prediction_mean', 'yld_prediction_min', 'yld_prediction_max'],
                    ['money_cost_corrected','money_cost_min_corrected','money_cost_max_corrected']]:

            meanvar = var[0]
            minvar = var[1]
            maxvar = var[2]
            # print(historical)
            pop = historical.loc[historical['year'].isin([year]), 'population_adult'].values[0]
            v = historical.loc[historical['year'].isin([year]), meanvar].values[0]
            vmin = historical.loc[historical['year'].isin([year]), minvar].values[0]
            vmax = historical.loc[historical['year'].isin([year]), maxvar].values[0]
            vper = 100000 * v / int(pop)
            vpermin = 100000 * vmin / int(pop)
            vpermax = 100000 * vmax / int(pop)

            dalyt = '{:.0f} \n ({:.0f}, {:.0f})'.format(v,
                                                        vmin,
                                                        vmax)
            yld_pert = '{:.0f} \n ({:.0f}, {:.0f})'.format(vper,
                                                           vpermin,
                                                           vpermax)

            vartable[var[0]+'_c'] = dalyt
            vartable[var[0] + '_per'] = yld_pert

            vartablenumbers[var[0]+'_c_mean'] = v
            vartablenumbers[var[0] + '_c_min'] = vmin
            vartablenumbers[var[0] + '_c_max'] = vmax
            vartablenumbers[var[0] + '_per_mean'] = vper
            vartablenumbers[var[0] + '_per_min'] = vpermin
            vartablenumbers[var[0] + '_per_max'] = vpermax

        daly_table[cname] = vartable
        daly_table_numbers[cname] = vartablenumbers

    data =  pd.DataFrame.from_dict(daly_table_numbers, orient='index')
    data['country'] = data.index
    data = data.reset_index(drop=True)
    data = data.sort_values(by='country')
    data = data.loc[~data['country'].isin(['Austria','Singapore','United Arab Emirates','Korea, South']),:]
    print(data.keys())
    daly_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['daly_c_mean'].sum(), data['daly_c_min'].sum(), data['daly_c_max'].sum())
    daly_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['daly_per_mean'].mean(), data['daly_per_min'].mean(),
                                                 data['daly_per_max'].mean())

    yll_prediction_mean_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['yll_prediction_mean_c_mean'].sum(), data['yll_prediction_mean_c_min'].sum(), data['yll_prediction_mean_c_max'].sum())
    yll_prediction_mean_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['yll_prediction_mean_per_mean'].mean(), data['yll_prediction_mean_per_min'].mean(),
                                                 data['yll_prediction_mean_per_max'].mean())

    yld_prediction_mean_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['yld_prediction_mean_c_mean'].sum(), data['yld_prediction_mean_c_min'].sum(), data['yld_prediction_mean_c_max'].sum())
    yld_prediction_mean_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['yld_prediction_mean_per_mean'].mean(), data['yld_prediction_mean_per_min'].mean(),
                                                 data['yld_prediction_mean_per_max'].mean())

    money_cost_corrected_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['money_cost_corrected_c_mean'].sum(), data['money_cost_corrected_c_min'].sum(), data['money_cost_corrected_c_max'].sum())
    money_cost_corrected_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(data['money_cost_corrected_per_mean'].mean(), data['money_cost_corrected_per_min'].mean(),
                                                 data['money_cost_corrected_per_max'].mean())

    daly_table['total'] = {'daly_c':daly_c, 'daly_per':daly_per,
                           'yll_prediction_mean_c':yll_prediction_mean_c,
                           'yll_prediction_mean_per':yll_prediction_mean_per,
                           'yld_prediction_mean_c':yld_prediction_mean_c,
                           'yld_prediction_mean_per':yld_prediction_mean_per,
                           'money_cost_corrected_c':money_cost_corrected_c,
                           'money_cost_corrected_c_per':money_cost_corrected_per}

    d = pd.DataFrame.from_dict(daly_table, orient='index')
    d['country'] = d.index
    d = d.reset_index(drop=True)
    d = d.sort_values(by='country')
    return d

def historical_table(dhist):
    clist = list(dhist['country'].value_counts().index)
    hists = []
    daly_table = {}
    daly_table_numbers = {}
    for cname in clist:
        historical = dhist.loc[dhist['country'] == cname]
        historical['daly'] = historical['yld_prediction_mean'] + historical['yll_prediction_mean']
        historical['daly_min'] = historical['yll_prediction_min'] + historical['yld_prediction_min']
        historical['daly_max'] = historical['yll_prediction_max'] + historical['yld_prediction_max']
        historical = historical.loc[historical['year'].isin([1990, 2000, 2010, 2023])]

        #print(historical)
        pop = historical.loc[historical['year'].isin([2000]), 'population_adult'].values[0]
        daly = historical.loc[historical['year'].isin([2000]), 'daly'].values[0]
        daly_min = historical.loc[historical['year'].isin([2000]), 'daly_min'].values[0]
        daly_max = historical.loc[historical['year'].isin([2000]), 'daly_max'].values[0]
        daly_per = 100000 * daly / int(pop)
        daly_per_min = 100000 * daly_min / int(pop)
        daly_per_max = 100000 * daly_max / int(pop)

        dalyt = '{:.1f} \n ({:.1f}, {:.1f})'.format(daly / 1000,
                                                    daly_min / 1000,
                                                    daly_max / 1000)
        yld_pert = '{:.0f} \n ({:.0f}, {:.0f})'.format(daly_per,
                                                       daly_per_min,
                                                       daly_per_max)

        #######
        pop2023 = historical.loc[historical['year'].isin([2023]), 'population_adult'].values[0]
        daly2023 = historical.loc[historical['year'].isin([2023]), 'daly'].values[0]
        daly_min2023 = historical.loc[historical['year'].isin([2023]), 'daly_min'].values[0]
        daly_max2023 = historical.loc[historical['year'].isin([2023]), 'daly_max'].values[0]
        daly_per2023 = 100000 * daly2023 / int(pop2023)
        daly_per_min2023 = 100000 * daly_min2023 / int(pop2023)
        daly_per_max2023 = 100000 * daly_max2023 / int(pop2023)

        daly2023t = '{:.1f} \n ({:.1f}, {:.1f})'.format(daly2023 / 1000,
                                                        daly_min2023 / 1000,
                                                        daly_max2023 / 1000)
        yld_per2023t = '{:.0f} \n ({:.0f}, {:.0f})'.format(daly_per2023,
                                                           daly_per_min2023,
                                                           daly_per_max2023)

        rate = 100 * (daly_per2023 - daly_per) / (daly_per)
        rate_min = 100 * (daly_per_min2023 - daly_per) / (daly_per)
        rate_high = 100 * (daly_per_max2023 - daly_per) / (daly_per)
        rate_change = '{:.0f} \n ({:.0f}, {:.0f})'.format(rate, rate_min, rate_high)
        pops = '{:.1f}'.format(pop2023 / 10e5)
        daly_table[cname] = {'daly_' + str(2020): dalyt, 'daly_per_' + str(2020): yld_pert,
                             'daly_' + str(2023): daly2023t, 'daly_per_' + str(2023): yld_per2023t,
                             'rate_change': rate_change,
                             'pop2023':pops}

        daly_table_numbers[cname] = {'daly' + str(2020): daly, 'dalymin' + str(2020): daly_min, 'dalymax' + str(2020): daly_max,
                                     'daly_per' + str(2020): daly_per, 'daly_permin' + str(2020): daly_per_min,
                                     'daly_permax' + str(2020): daly_per_max,

                                     'daly2023' + str(2023): daly2023, 'dalymin2023' + str(2023): daly_min2023, 'daly2023max' + str(2023): daly_max2023,
                                     'daly_per2023' + str(2023): daly_per2023, 'daly_permin2023' + str(2023): daly_per_min2023,
                                     'daly_permax2023' + str(2023): daly_per_max2023,


                                     'pop2023': pop2023,'rate_change': rate, 'rate_min': rate_min, 'rate_high': rate_high}

    d =  pd.DataFrame.from_dict(daly_table_numbers, orient='index')
    d = d.loc[~d.index.isin(['Austria','Singapore','United Arab Emirates','Korea, South']), :]
    dsum = d.sum()
    dmean = d.mean()

    #################
    daly2020global = '{:.1f} \n ({:.1f}, {:.1f})'.format(dsum['daly2020'] / 1000,
                                                    dsum['dalymin2020'] / 1000,
                                                    dsum['dalymax2020'] / 1000)
    daly_per2020global = '{:.0f} \n ({:.0f}, {:.0f})'.format(dmean['daly_per2020'],
                                                       dmean['daly_permin2020'],
                                                       dmean['daly_permax2020'])

    daly2023global = '{:.1f} \n ({:.1f}, {:.1f})'.format(dsum['daly20232023'] / 1000,
                                                    dsum['dalymin20232023'] / 1000,
                                                    dsum['daly2023max2023'] / 1000)
    daly_per2023global = '{:.0f} \n ({:.0f}, {:.0f})'.format(dmean['daly_per20232023'],
                                                       dmean['daly_permin20232023'],
                                                       dmean['daly_permax20232023'])

    rate_change = '{:.0f} \n ({:.0f}, {:.0f})'.format(dmean['rate_change'], dmean['rate_min'],
                                                      dmean['rate_high'])
    pops = '{:.1f}'.format(dsum['pop2023']/10e5)
    daly_table['total'] = {'daly_' + str(2020): daly2020global, 'daly_per_' + str(2020): daly_per2020global,
                         'daly_' + str(2023): daly2023global, 'daly_per_' + str(2023): daly_per2023global,
                         'rate_change': rate_change,
                           'pop2023':pops}

    d = pd.DataFrame.from_dict(daly_table, orient='index')
    d['country'] = d.index
    d = d.reset_index(drop=True)
    d = d.sort_values(by='country')


    return d

if __name__ == "__main__":
    dhist = pd.read_csv('metadata/figure_table_productivity/historical_predictions_prod.csv')
    ssp126 = pd.read_csv('metadata/figure_table_productivity/ssp126_prod.csv')
    ssp245 = pd.read_csv('metadata/figure_table_productivity/ssp245_prod.csv')
    ssp370 = pd.read_csv('metadata/figure_table_productivity/ssp370_prod.csv')
    ssp585 = pd.read_csv('metadata/figure_table_productivity/ssp585_prod.csv')

    ssps = pd.concat([ssp126, ssp245, ssp370,ssp585], ignore_index=True)

    plot_figures(dhist, ssps)

    plot_test(hist=dhist, ssps=ssps)

# t2023 = ssp_table(dhist.copy(),year=2023)
# t2023.to_excel('tables/hist2023_prod.xlsx', index=False)
#
# t2000 = ssp_table(dhist.copy(),year=2000)
# t2000.to_excel('tables/hist2000_prod.xlsx', index=False)
#
# #ssptable126.to_excel('tables/ssptable126_2050_prod.xlsx', index=False)
#
# ssptable126 = ssp_table(ssp126.copy(),year=2050, correction=1e6)
# ssptable126.to_excel('tables/ssptable126_2050_prod.xlsx', index=False)
#
# ssptable126_2100 = ssp_table(ssp126.copy(),year=2098, correction=1e6)
# ssptable126_2100.to_excel('tables/ssptable126_2100_prod.xlsx', index=False)
#
# ssptable245 = ssp_table(ssp245.copy(),year=2050, correction=1e6)
# ssptable245.to_excel('tables/ssptable245_2050_prod.xlsx', index=False)
# ssptable245_2100 = ssp_table(ssp245.copy(),year=2098, correction=1e6)
# ssptable245_2100.to_excel('tables/ssptable245_2100_prod.xlsx', index=False)
#
# ssptable370 = ssp_table(ssp370.copy(),year=2050, correction=1e6)
# ssptable370.to_excel('tables/ssptable370_2050_prod.xlsx', index=False)
# ssptable370_2100 = ssp_table(ssp370.copy(),year=2098, correction=1e6)
# ssptable370_2100.to_excel('tables/ssptable370_2100_prod.xlsx', index=False)
#
# ssptable585 = ssp_table(ssp585.copy(),year=2050, correction=1e6)
# ssptable585.to_excel('tables/ssptable585_2050_prod.xlsx', index=False)
# ssptable585_2100 = ssp_table(ssp585.copy(),year=2098, correction=1e6)
# ssptable585_2100.to_excel('tables/ssptable585_2100_prod.xlsx', index=False)
#
