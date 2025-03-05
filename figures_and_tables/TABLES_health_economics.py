import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import os
plt.style.use('./nature.mplstyle')

def load_processed_data(input_dir, sensitvity=False):
    """
    Load processed data from CSV files.

    Parameters:
    -----------
    input_dir : str, optional
        Directory containing the processed data

    Returns:
    --------
    tuple
        (historical_data, ssp126_data, ssp245_data, ssp370_data, ssp585_data, combined_ssp)
    """
    # Load each dataset
    if sensitvity:
        historical_data = None
        ssp126_data = pd.read_csv(f"{input_dir}/ssp126_sens.csv")
        ssp245_data = pd.read_csv(f"{input_dir}/ssp245_sens.csv")
        ssp370_data = pd.read_csv(f"{input_dir}/ssp370_sens.csv")
        ssp585_data = pd.read_csv(f"{input_dir}/ssp585_sens.csv")
    else:
        historical_data = pd.read_csv(f"{input_dir}/historical_predictions.csv")
        ssp126_data = pd.read_csv(f"{input_dir}/ssp126.csv")
        ssp245_data = pd.read_csv(f"{input_dir}/ssp245.csv")
        ssp370_data = pd.read_csv(f"{input_dir}/ssp370.csv")
        ssp585_data = pd.read_csv(f"{input_dir}/ssp585.csv")

    combined_ssp = pd.concat([ssp126_data, ssp245_data, ssp370_data, ssp585_data], ignore_index=True)

    return historical_data, ssp126_data, ssp245_data, ssp370_data, ssp585_data, combined_ssp


def ssp_table(scenario_data, year=2050):
    """
    Generate summary table for a specific SSP scenario and year.

    Parameters:
    -----------
    scenario_data : pandas.DataFrame
        Processed data for a specific SSP scenario
    year : int, optional
        Target year for the table

    Returns:
    --------
    pandas.DataFrame
        Summary table with health and economic metrics
    """
    # Filter out specific countries
    excluded_countries = [
        'Austria', 'Singapore', 'United Arab Emirates', 'Korea, South', 'India', 'Israel', 'Brazil', 'China',
        'Croatia', 'New Zealand', 'Thailand', 'Turkey'
    ]

    df = scenario_data.copy()

    # Rename columns for consistency if needed
    if 'yld_diff_mean_corrected' in df.columns:
        df = df.rename(columns={
            'yld_diff_mean_corrected': 'yld_prediction_mean',
            'yld_diff_max_corrected': 'yld_prediction_max',
            'yld_diff_min_corrected': 'yld_prediction_min',
            'yll_diff_corrected': 'yll_prediction_mean',
            'yll_diff_min_corrected': 'yll_prediction_min',
            'yll_diff_max_corrected': 'yll_prediction_max'
        })

    # Filter countries
    dv = df.loc[~df['country'].isin(excluded_countries)]
    clist = list(dv['country'].unique())

    # Initialize tables
    daly_table = {}
    daly_table_numbers = {}

    # Process data for each country
    for cname in clist:
        country_data = df.loc[df['country'] == cname]

        # Calculate DALYs (Disability-Adjusted Life Years)
        country_data['daly'] = country_data['yld_prediction_mean'] + country_data['yll_prediction_mean']
        country_data['daly_min'] = country_data['yll_prediction_min'] + country_data['yld_prediction_min']
        country_data['daly_max'] = country_data['yll_prediction_max'] + country_data['yld_prediction_max']

        # Convert monetary costs to millions
        country_data['money_cost_corrected'] = country_data['money_cost_corrected'] / 1e6
        country_data['money_cost_min_corrected'] = country_data['money_cost_min_corrected'] / 1e6
        country_data['money_cost_max_corrected'] = country_data['money_cost_max_corrected'] / 1e6

        # Initialize tables for this country
        vartable = {}
        vartablenumbers = {}

        # Process each metric type
        metrics = [
            ['daly', 'daly_min', 'daly_max'],
            ['yll_prediction_mean', 'yll_prediction_min', 'yll_prediction_max'],
            ['yld_prediction_mean', 'yld_prediction_min', 'yld_prediction_max'],
            ['money_cost_corrected', 'money_cost_min_corrected', 'money_cost_max_corrected']
        ]

        for var in metrics:
            try:
                meanvar, minvar, maxvar = var[0], var[1], var[2]

                # Try both population column names (for compatibility)
                pop_col = 'adult_population' if 'adult_population' in country_data.columns else 'population_adult'
                year_data = country_data.loc[country_data['year'] == year]

                if len(year_data) == 0:
                    print(f"No data for {cname} in year {year}")
                    continue

                pop = year_data[pop_col].values[0]
                v = year_data[meanvar].values[0]
                vmin = year_data[minvar].values[0]
                vmax = year_data[maxvar].values[0]

                # Calculate per capita metrics
                vper = 100000 * v / int(pop)
                vpermin = 100000 * vmin / int(pop)
                vpermax = 100000 * vmax / int(pop)

                # Format strings for display
                abs_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(v, vmin, vmax)
                per_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(vper, vpermin, vpermax)

                # Store formatted values
                vartable[f"{meanvar}_c"] = abs_str
                vartable[f"{meanvar}_per"] = per_str
                vartable['pop'] = "{:.1f}".format(pop / 1e6)

                # Store numbers for calculations
                vartablenumbers[f"{meanvar}_c_mean"] = v
                vartablenumbers[f"{meanvar}_c_min"] = vmin
                vartablenumbers[f"{meanvar}_c_max"] = vmax
                vartablenumbers[f"{meanvar}_per_mean"] = vper
                vartablenumbers[f"{meanvar}_per_min"] = vpermin
                vartablenumbers[f"{meanvar}_per_max"] = vpermax
                vartablenumbers[f"{meanvar}_population"] = pop / 1e6

            except Exception as e:
                print(f"Error processing {cname}: {e}")

        daly_table[cname] = vartable
        daly_table_numbers[cname] = vartablenumbers

    # Create a dataframe from the numbers dict for easier aggregation
    data = pd.DataFrame.from_dict(daly_table_numbers, orient='index')
    data['country'] = data.index
    data = data.reset_index(drop=True)
    data = data.sort_values(by='country')

    # Filter out excluded countries again (just to be safe)
    data = data.loc[~data['country'].isin(excluded_countries)]

    # Calculate totals and averages
    try:
        # DALY totals and averages
        daly_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['daly_c_mean'].sum(),
            data['daly_c_min'].sum(),
            data['daly_c_max'].sum()
        )
        daly_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['daly_per_mean'].mean(),
            data['daly_per_min'].mean(),
            data['daly_per_max'].mean()
        )

        # YLL totals and averages
        yll_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['yll_prediction_mean_c_mean'].sum(),
            data['yll_prediction_mean_c_min'].sum(),
            data['yll_prediction_mean_c_max'].sum()
        )
        yll_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['yll_prediction_mean_per_mean'].mean(),
            data['yll_prediction_mean_per_min'].mean(),
            data['yll_prediction_mean_per_max'].mean()
        )

        # YLD totals and averages
        yld_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['yld_prediction_mean_c_mean'].sum(),
            data['yld_prediction_mean_c_min'].sum(),
            data['yld_prediction_mean_c_max'].sum()
        )
        yld_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['yld_prediction_mean_per_mean'].mean(),
            data['yld_prediction_mean_per_min'].mean(),
            data['yld_prediction_mean_per_max'].mean()
        )

        # Cost totals and averages
        cost_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['money_cost_corrected_c_mean'].sum(),
            data['money_cost_corrected_c_min'].sum(),
            data['money_cost_corrected_c_max'].sum()
        )
        cost_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['money_cost_corrected_per_mean'].mean(),
            data['money_cost_corrected_per_min'].mean(),
            data['money_cost_corrected_per_max'].mean()
        )

        # Add totals row
        pop_sum = data['daly_population'].sum() if 'daly_population' in data.columns else 0
        daly_table['total'] = {
            'daly_c': daly_c,
            'daly_per': daly_per,
            'yll_prediction_mean_c': yll_c,
            'yll_prediction_mean_per': yll_per,
            'yld_prediction_mean_c': yld_c,
            'yld_prediction_mean_per': yld_per,
            'money_cost_corrected_c': cost_c,
            'money_cost_corrected_per': cost_per,
            'pop': pop_sum
        }
    except Exception as e:
        print(f"Error calculating totals: {e}")

    # Convert to dataframe
    result = pd.DataFrame.from_dict(daly_table, orient='index')
    result['country'] = result.index
    result = result.reset_index(drop=True)
    result = result.sort_values(by='country')

    return result

def historical_table(historical_data, base_year=2000, target_year=2023):
    """
    Generate a comparison table between a base year and a target year.

    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Processed historical data
    base_year : int, optional
        Base year for comparison (default 2000)
    target_year : int, optional
        Target year for comparison (default 2023)

    Returns:
    --------
    pandas.DataFrame
        Summary table with health metrics comparison
    """
    # Filter out specific countries
    excluded_countries = [
        'Austria', 'Singapore', 'United Arab Emirates', 'Korea, South', 'India', 'Israel', 'Brazil', 'China',
        'Croatia', 'New Zealand', 'Thailand', 'Turkey'
    ]

    dv = historical_data.loc[~historical_data['country'].isin(excluded_countries)]
    clist = list(dv['country'].unique())

    # Initialize tables
    daly_table = {}
    daly_table_numbers = {}

    # Process each country
    for cname in clist:
        country_data = historical_data.loc[historical_data['country'] == cname].copy()

        # Calculate DALYs
        country_data['daly'] = country_data['yld_prediction_mean'] + country_data['yll_prediction_mean']
        country_data['daly_min'] = country_data['yll_prediction_min'] + country_data['yld_prediction_min']
        country_data['daly_max'] = country_data['yll_prediction_max'] + country_data['yld_prediction_max']

        # Filter for relevant years
        country_data = country_data.loc[country_data['year'].isin([1990, base_year, 2010, target_year])]

        try:
            # Base year values
            base_pop = country_data.loc[country_data['year'] == base_year, 'population_adult'].values[0]
            base_daly = country_data.loc[country_data['year'] == base_year, 'daly'].values[0]
            base_daly_min = country_data.loc[country_data['year'] == base_year, 'daly_min'].values[0]
            base_daly_max = country_data.loc[country_data['year'] == base_year, 'daly_max'].values[0]

            # Per capita values for base year
            base_daly_per = 100000 * base_daly / int(base_pop)
            base_daly_per_min = 100000 * base_daly_min / int(base_pop)
            base_daly_per_max = 100000 * base_daly_max / int(base_pop)

            # Format base year values
            base_daly_str = '{:.1f} \n ({:.1f}, {:.1f})'.format(
                base_daly / 1000, base_daly_min / 1000, base_daly_max / 1000
            )
            base_daly_per_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(
                base_daly_per, base_daly_per_min, base_daly_per_max
            )

            # Target year values
            target_pop = country_data.loc[country_data['year'] == target_year, 'population_adult'].values[0]
            target_daly = country_data.loc[country_data['year'] == target_year, 'daly'].values[0]
            target_daly_min = country_data.loc[country_data['year'] == target_year, 'daly_min'].values[0]
            target_daly_max = country_data.loc[country_data['year'] == target_year, 'daly_max'].values[0]

            # Per capita values for target year
            target_daly_per = 100000 * target_daly / int(target_pop)
            target_daly_per_min = 100000 * target_daly_min / int(target_pop)
            target_daly_per_max = 100000 * target_daly_max / int(target_pop)

            # Format target year values
            target_daly_str = '{:.1f} \n ({:.1f}, {:.1f})'.format(
                target_daly / 1000, target_daly_min / 1000, target_daly_max / 1000
            )
            target_daly_per_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(
                target_daly_per, target_daly_per_min, target_daly_per_max
            )

            # Calculate percent change
            pct_change = 100 * (target_daly_per - base_daly_per) / base_daly_per
            pct_change_min = 100 * (target_daly_per_min - base_daly_per) / base_daly_per
            pct_change_max = 100 * (target_daly_per_max - base_daly_per) / base_daly_per

            # Format percent change
            pct_change_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(
                pct_change, pct_change_min, pct_change_max
            )

            # Store results
            daly_table[cname] = {
                f'daly_{base_year}': base_daly_str,
                f'daly_per_{base_year}': base_daly_per_str,
                f'daly_{target_year}': target_daly_str,
                f'daly_per_{target_year}': target_daly_per_str,
                'rate_change': pct_change_str,
                'pop2023': '{:.1f}'.format(target_pop / 10e5)
            }

            # Store raw numbers
            daly_table_numbers[cname] = {
                f'daly{base_year}': base_daly,
                f'dalymin{base_year}': base_daly_min,
                f'dalymax{base_year}': base_daly_max,
                f'daly_per{base_year}': base_daly_per,
                f'daly_permin{base_year}': base_daly_per_min,
                f'daly_permax{base_year}': base_daly_per_max,

                f'daly2023{target_year}': target_daly,
                f'dalymin2023{target_year}': target_daly_min,
                f'daly2023max{target_year}': target_daly_max,
                f'daly_per2023{target_year}': target_daly_per,
                f'daly_permin2023{target_year}': target_daly_per_min,
                f'daly_permax2023{target_year}': target_daly_per_max,

                'pop2023': target_pop,
                'rate_change': pct_change,
                'rate_min': pct_change_min,
                'rate_high': pct_change_max
            }

        except Exception as e:
            print(f"Error processing {cname}: {e}")

    # Convert to dataframe for analysis
    data = pd.DataFrame.from_dict(daly_table_numbers, orient='index')

    # Filter out excluded countries again
    data = data.loc[~data.index.isin(excluded_countries)]

    # Calculate sums and means
    dsum = data.sum()
    dmean = data.mean()

    # Calculate global totals and averages
    try:
        # Base year global values
        base_global_str = '{:.1f} \n ({:.1f}, {:.1f})'.format(
            dsum[f'daly{base_year}'] / 1000,
            dsum[f'dalymin{base_year}'] / 1000,
            dsum[f'dalymax{base_year}'] / 1000
        )
        base_global_per_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            dmean[f'daly_per{base_year}'],
            dmean[f'daly_permin{base_year}'],
            dmean[f'daly_permax{base_year}']
        )

        # Target year global values
        target_global_str = '{:.1f} \n ({:.1f}, {:.1f})'.format(
            dsum[f'daly2023{target_year}'] / 1000,
            dsum[f'dalymin2023{target_year}'] / 1000,
            dsum[f'daly2023max{target_year}'] / 1000
        )
        target_global_per_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            dmean[f'daly_per2023{target_year}'],
            dmean[f'daly_permin2023{target_year}'],
            dmean[f'daly_permax2023{target_year}']
        )

        # Global rate change
        rate_change_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            dmean['rate_change'], dmean['rate_min'], dmean['rate_high']
        )

        # Add global row
        daly_table['total'] = {
            f'daly_{base_year}': base_global_str,
            f'daly_per_{base_year}': base_global_per_str,
            f'daly_{target_year}': target_global_str,
            f'daly_per_{target_year}': target_global_per_str,
            'rate_change': rate_change_str,
            'pop2023': '{:.1f}'.format(dsum['pop2023'] / 10e5)
        }
    except Exception as e:
        print(f"Error calculating global totals: {e}")

    # Convert to dataframe
    result = pd.DataFrame.from_dict(daly_table, orient='index')
    result['country'] = result.index
    result = result.reset_index(drop=True)
    result = result.sort_values(by='country')

    return result


def t2023_table(historical_data, year=2023):
    """
    Generate table with metrics for the current (2023) scenario.

    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Processed historical data
    year : int, optional
        Target year for the table (default 2023)

    Returns:
    --------
    pandas.DataFrame
        Summary table with health and economic metrics for the specified year
    """
    # Filter out specific countries
    excluded_countries = [
        'Austria', 'Singapore', 'United Arab Emirates', 'Korea, South', 'India', 'Israel', 'Brazil', 'China',
        'Croatia', 'New Zealand', 'Thailand', 'Turkey'
    ]

    dv = historical_data.loc[~historical_data['country'].isin(excluded_countries)]
    clist = list(dv['country'].unique())

    # Initialize tables
    daly_table = {}
    daly_table_numbers = {}

    # Process each country
    for cname in clist:
        country_data = historical_data.loc[historical_data['country'] == cname].copy()

        # Calculate DALYs
        country_data['daly'] = country_data['yld_prediction_mean'] + country_data['yll_prediction_mean']
        country_data['daly_min'] = country_data['yll_prediction_min'] + country_data['yld_prediction_min']
        country_data['daly_max'] = country_data['yll_prediction_max'] + country_data['yld_prediction_max']

        # Convert monetary costs to millions
        country_data['money_cost_corrected'] = country_data['money_cost_corrected'] / 1e6
        country_data['money_cost_min_corrected'] = country_data['money_cost_min_corrected'] / 1e6
        country_data['money_cost_max_corrected'] = country_data['money_cost_max_corrected'] / 1e6

        # Initialize tables for this country
        vartable = {}
        vartablenumbers = {}

        # Process each metric type
        metrics = [
            ['daly', 'daly_min', 'daly_max'],
            ['yll_prediction_mean', 'yll_prediction_min', 'yll_prediction_max'],
            ['yld_prediction_mean', 'yld_prediction_min', 'yld_prediction_max'],
            ['money_cost_corrected', 'money_cost_min_corrected', 'money_cost_max_corrected']
        ]

        for var in metrics:
            try:
                meanvar, minvar, maxvar = var[0], var[1], var[2]

                # Get population and values
                year_data = country_data.loc[country_data['year'] == year]

                if len(year_data) == 0:
                    print(f"No data for {cname} in year {year}")
                    continue

                pop = year_data['population_adult'].values[0]
                v = year_data[meanvar].values[0]
                vmin = year_data[minvar].values[0]
                vmax = year_data[maxvar].values[0]

                # Calculate per capita metrics
                vper = 100000 * v / int(pop)
                vpermin = 100000 * vmin / int(pop)
                vpermax = 100000 * vmax / int(pop)

                # Format strings for display
                abs_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(v, vmin, vmax)
                per_str = '{:.0f} \n ({:.0f}, {:.0f})'.format(vper, vpermin, vpermax)

                # Store formatted values and raw numbers
                vartable[f"{meanvar}_c"] = abs_str
                vartable[f"{meanvar}_per"] = per_str

                vartablenumbers[f"{meanvar}_c_mean"] = v
                vartablenumbers[f"{meanvar}_c_min"] = vmin
                vartablenumbers[f"{meanvar}_c_max"] = vmax
                vartablenumbers[f"{meanvar}_per_mean"] = vper
                vartablenumbers[f"{meanvar}_per_min"] = vpermin
                vartablenumbers[f"{meanvar}_per_max"] = vpermax

            except Exception as e:
                print(f"Error processing {cname}: {e}")

        daly_table[cname] = vartable
        daly_table_numbers[cname] = vartablenumbers

    # Create a dataframe from the numbers dict
    data = pd.DataFrame.from_dict(daly_table_numbers, orient='index')
    data['country'] = data.index
    data = data.reset_index(drop=True)
    data = data.sort_values(by='country')

    # Filter out excluded countries again
    data = data.loc[~data['country'].isin(excluded_countries)]

    # Calculate totals and averages
    try:
        # DALY totals and averages
        daly_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['daly_c_mean'].sum(),
            data['daly_c_min'].sum(),
            data['daly_c_max'].sum()
        )
        daly_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['daly_per_mean'].mean(),
            data['daly_per_min'].mean(),
            data['daly_per_max'].mean()
        )

        # YLL totals and averages
        yll_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['yll_prediction_mean_c_mean'].sum(),
            data['yll_prediction_mean_c_min'].sum(),
            data['yll_prediction_mean_c_max'].sum()
        )
        yll_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['yll_prediction_mean_per_mean'].mean(),
            data['yll_prediction_mean_per_min'].mean(),
            data['yll_prediction_mean_per_max'].mean()
        )

        # YLD totals and averages
        yld_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['yld_prediction_mean_c_mean'].sum(),
            data['yld_prediction_mean_c_min'].sum(),
            data['yld_prediction_mean_c_max'].sum()
        )
        yld_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['yld_prediction_mean_per_mean'].mean(),
            data['yld_prediction_mean_per_min'].mean(),
            data['yld_prediction_mean_per_max'].mean()
        )

        # Cost totals and averages
        cost_c = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['money_cost_corrected_c_mean'].sum(),
            data['money_cost_corrected_c_min'].sum(),
            data['money_cost_corrected_c_max'].sum()
        )
        cost_per = '{:.0f} \n ({:.0f}, {:.0f})'.format(
            data['money_cost_corrected_per_mean'].mean(),
            data['money_cost_corrected_per_min'].mean(),
            data['money_cost_corrected_per_max'].mean()
        )

        # Add totals row
        daly_table['total'] = {
            'daly_c': daly_c,
            'daly_per': daly_per,
            'yll_prediction_mean_c': yll_c,
            'yll_prediction_mean_per': yll_per,
            'yld_prediction_mean_c': yld_c,
            'yld_prediction_mean_per': yld_per,
            'money_cost_corrected_c': cost_c,
            'money_cost_corrected_per': cost_per
        }
    except Exception as e:
        print(f"Error calculating totals: {e}")

    # Convert to dataframe
    result = pd.DataFrame.from_dict(daly_table, orient='index')
    result['country'] = result.index
    result = result.reset_index(drop=True)
    result = result.sort_values(by='country')

    return result


if __name__ == "__main__":
    historical_data, ssp126_data, ssp245_data, ssp370_data, ssp585_data, combined_ssp = load_processed_data(input_dir='metadata/figure_table_health_economics',
                                                                                                            sensitvity=False)

    _, _, ssp245_data_sens, ssp370_data_sens, _, _ = load_processed_data(
        input_dir='metadata/figure_table_health_economics', sensitvity=True)

    histdata = historical_table(historical_data=historical_data, base_year=2000, target_year=2023)
    t2023 = t2023_table(historical_data=historical_data, year=2023)
    histdata = histdata.merge(t2023, on='country', how='left', copy=False)

    ssp126_2050 = ssp_table(scenario_data=ssp126_data, year=2050)
    ssp245_2050 = ssp_table(scenario_data=ssp245_data, year=2050)
    ssp370_2050 = ssp_table(scenario_data=ssp370_data, year=2050)
    ssp585_2050 = ssp_table(scenario_data=ssp585_data, year=2050)

    ssp126_2100 = ssp_table(scenario_data=ssp126_data, year=2098)
    ssp245_2100 = ssp_table(scenario_data=ssp245_data, year=2098)
    ssp370_2100 = ssp_table(scenario_data=ssp370_data, year=2098)
    ssp585_2100 = ssp_table(scenario_data=ssp585_data, year=2098)

    # sensitivity
    ssp245_2050_sens = ssp_table(scenario_data=ssp245_data_sens, year=2050)
    ssp370_2050_sens = ssp_table(scenario_data=ssp370_data_sens, year=2050)
    ssp245_2100_sens = ssp_table(scenario_data=ssp245_data_sens, year=2098)
    ssp370_2100_sens = ssp_table(scenario_data=ssp370_data_sens, year=2098)

    ### Table S2
    # Get the last row of the DataFrame
    last_row = histdata.iloc[-1].copy()
    new_first_row = pd.DataFrame([last_row], columns=histdata.columns)
    # Drop the last row from the original DataFrame
    df_without_last = histdata.iloc[:-1].copy()
    # Concatenate the new first row with the DataFrame without the last row
    histdatanew = pd.concat([new_first_row, df_without_last], ignore_index=True)
    print(histdatanew.keys())
    tables2 = histdatanew[['daly_per_2000','daly_per_2023','daly_c','yll_prediction_mean_c','yld_prediction_mean_c','money_cost_corrected_c','pop2023']]
    pd.DataFrame(tables2).to_excel('TABLE_S2.xlsx')
    ###

    import pandas as pd
    import os

    daly_rate = {}
    yld_c = {}
    yll_c = {}
    money_c = {}

    for n, df in enumerate(
            [ssp126_2050, ssp245_2050, ssp370_2050, ssp585_2050,
             ssp126_2100, ssp245_2100, ssp370_2100, ssp585_2100]):

        # Get the last row of the DataFrame
        last_row = df.iloc[-1].copy()
        # Create a new DataFrame with the last row
        new_first_row = pd.DataFrame([last_row], columns=df.columns)
        # Drop the last row from the original DataFrame
        df_without_last = df.iloc[:-1].copy()
        # Concatenate the new first row with the DataFrame without the last row
        df_reordered = pd.concat([new_first_row, df_without_last], ignore_index=True)

        if n == 0:
            daly_rate['country'] = df_reordered['country'].values
            yld_c['country'] = df_reordered['country'].values
            yll_c['country'] = df_reordered['country'].values
            money_c['country'] = df_reordered['country'].values

        daly_rate[str(n)] = df_reordered['daly_per'].values
        yld_c[str(n)] = df_reordered['yld_prediction_mean_c'].values
        yll_c[str(n)] = df_reordered['yll_prediction_mean_c'].values
        money_c[str(n)] = df_reordered['money_cost_corrected_c'].values

    pd.DataFrame(daly_rate).to_excel('TABLE_S3.xlsx')

    pd.DataFrame(yll_c).to_excel('TABLE_S4.xlsx')  # yll

    pd.DataFrame(yld_c).to_excel('TABLE_S5.xlsx')  # yld

    pd.DataFrame(money_c).to_excel('TABLE_S6.xlsx')  # cost

    #####################################################
    ## SENSITIVITY TABLES

    import pandas as pd
    import os

    daly_rate = {}
    yld_c = {}
    yll_c = {}
    money_c = {}
    daly_c = {}

    for n, df in enumerate([ssp245_2050, ssp245_2050_sens,
                              ssp370_2050, ssp370_2050_sens,
                              ssp245_2100, ssp245_2100_sens,
                              ssp370_2050, ssp370_2100_sens]):
        # Get the last row of the DataFrame
        last_row = df.iloc[-1].copy()
        # Create a new DataFrame with the last row
        new_first_row = pd.DataFrame([last_row], columns=df.columns)
        # Drop the last row from the original DataFrame
        df_without_last = df.iloc[:-1].copy()
        # Concatenate the new first row with the DataFrame without the last row
        df_reordered = pd.concat([new_first_row, df_without_last], ignore_index=True)

        if n == 0:
            daly_rate['country'] = df_reordered['country'].values
            yld_c['country'] = df_reordered['country'].values
            yll_c['country'] = df_reordered['country'].values
            money_c['country'] = df_reordered['country'].values
            daly_c['country'] = df_reordered['country'].values

        daly_rate[str(n)] = df_reordered['daly_per'].values
        yld_c[str(n)] = df_reordered['yld_prediction_mean_c'].values
        yll_c[str(n)] = df_reordered['yll_prediction_mean_c'].values
        money_c[str(n)] = df_reordered['money_cost_corrected_c'].values
        daly_c[str(n)] = df_reordered['daly_c'].values

    pd.DataFrame(daly_rate).to_excel('TABLE_S7.xlsx')
    pd.DataFrame(daly_c).to_excel('TABLE_S8.xlsx')





