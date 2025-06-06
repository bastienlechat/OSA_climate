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

    t2023 = ssp_table(dhist.copy(),year=2023)
    t2000 = ssp_table(dhist.copy(),year=2000)

    ssptable126 = ssp_table(ssp126.copy(),year=2050, correction=1e6)
    ssptable126_2100 = ssp_table(ssp126.copy(),year=2098, correction=1e6)

    ssptable245 = ssp_table(ssp245.copy(),year=2050, correction=1e6)
    ssptable245_2100 = ssp_table(ssp245.copy(),year=2098, correction=1e6)

    ssptable370 = ssp_table(ssp370.copy(),year=2050, correction=1e6)
    ssptable370_2100 = ssp_table(ssp370.copy(),year=2098, correction=1e6)

    ssptable585 = ssp_table(ssp585.copy(),year=2050, correction=1e6)
    ssptable585_2100 = ssp_table(ssp585.copy(),year=2098, correction=1e6)

    ####################
    ## Table S9
    t2023 = t2023.merge(t2000, on='country', how='left', copy=False, suffixes=('', '_2000'))
    last_row = t2023.iloc[-1].copy()
    new_first_row = pd.DataFrame([last_row], columns=t2023.columns)
    # Drop the last row from the original DataFrame
    df_without_last = t2023.iloc[:-1].copy()
    # Concatenate the new first row with the DataFrame without the last row
    histdatanew = pd.concat([new_first_row, df_without_last], ignore_index=True)
    tables9 = histdatanew[['labor_loss_mean_lowess_per', 'labor_loss_mean_lowess_per_2000',
                           'climate_presenteism_mean_lowess_c', 'climate_absenteism_mean_lowess_c', 'labor_loss_mean_lowess_c']]
    pd.DataFrame(tables9).to_excel('TABLE_S9.xlsx')

    ###################
    # Table S10 and S11
    import pandas as pd
    import os
    lloss = {}
    lloss_p = {}
    yll_c = {}
    money_c = {}
    for n, df in enumerate([ssptable126, ssptable245, ssptable370,
                              ssptable585,
                              ssptable126_2100, ssptable245_2100, ssptable370_2100,
                              ssptable585_2100]):
        # Get the last row of the DataFrame
        last_row = df.iloc[-1].copy()
        # Create a new DataFrame with the last row
        new_first_row = pd.DataFrame([last_row], columns=df.columns)
        # Drop the last row from the original DataFrame
        df_without_last = df.iloc[:-1].copy()
        # Concatenate the new first row with the DataFrame without the last row
        df_reordered = pd.concat([new_first_row, df_without_last], ignore_index=True)
        print(df_reordered.keys())

        if n == 0:
            lloss['country'] = df_reordered['country'].values
            lloss_p['country'] = df_reordered['country'].values

        lloss[str(n)] = df_reordered['labor_loss_mean_lowess_c'].values
        lloss_p[str(n)] = df_reordered['labor_loss_mean_lowess_per'].values

    pd.DataFrame(lloss).to_excel('TABLE_S10.xlsx')
    pd.DataFrame(lloss_p).to_excel('TABLE_S11.xlsx')