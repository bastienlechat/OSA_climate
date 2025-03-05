import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import os
plt.style.use('./nature.mplstyle')

def plot_country_from_processed_data(historical_data, ssp_data, country_name, output_dir=None):
    """
    Plot time series data for a specific country using pre-processed data.

    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Processed historical data
    ssp_data : pandas.DataFrame
        Combined SSP scenario data (ssp126, ssp245, ssp370, ssp585)
    country_name : str
        Name of the country to plot
    output_dir : str, optional
        Directory to save the plot. If None, plot is displayed but not saved.
    """
    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, sharex='col', figsize=(3.8, 5))
    axs = list(axs.flatten())

    # Filter data for the specific country
    historical = historical_data[historical_data['country'] == country_name].copy()
    ssp126 = ssp_data[(ssp_data['country'] == country_name) & (ssp_data['exp'] == 'ssp126')].copy()
    ssp245 = ssp_data[(ssp_data['country'] == country_name) & (ssp_data['exp'] == 'ssp245')].copy()
    ssp370 = ssp_data[(ssp_data['country'] == country_name) & (ssp_data['exp'] == 'ssp370')].copy()
    ssp585 = ssp_data[(ssp_data['country'] == country_name) & (ssp_data['exp'] == 'ssp585')].copy()

    # Apply smoothing function for visualization
    lowess = sm.nonparametric.lowess
    nyears = 74
    smooth_param = 20
    frac = smooth_param / nyears

    # Plot historical data
    axs[0].plot(historical['year'].values, historical['t2m_mean'].values, 'k', alpha=0.2, label='Historical',
                linestyle='--')
    axs[0].plot(historical['year'].values, historical['t2m_mean_corrected'].values, 'k', alpha=0.9, label='Historical')

    axs[1].plot(historical['year'].values, historical['osa_person_days'].values / 1e6, 'k', alpha=0.2,
                label='Historical', linestyle='--')
    axs[1].plot(historical['year'].values, historical['osa_person_days_corrected'].values / 1e6, 'k', alpha=0.9,
                label='Historical')

    axs[2].plot(historical['year'].values, historical['yld_prediction_mean'].values / 1000, 'k', alpha=0.8)
    axs[2].plot(historical['year'].values, historical['yld_diff_mean'].values / 1000, 'k', alpha=0.2,
                label='Historical', linestyle='--')

    axs[3].plot(historical['year'].values, historical['yll_diff'].values / 1000, 'k', alpha=0.2, label='Historical',
                linestyle='--')
    axs[3].plot(historical['year'].values, historical['yll_prediction_mean'].values / 1000, 'k', alpha=0.8)

    # Define colors for SSP scenarios
    exp_colors = ['#003049', '#fcbf49', '#f77f00', '#d62828']

    # Process and plot each SSP scenario
    for n, (exp, exp_data) in enumerate([
        ('ssp126', ssp126), ('ssp245', ssp245), ('ssp370', ssp370), ('ssp585', ssp585)
    ]):
        if len(exp_data) == 0:
            print(f"No {exp} data for {country_name}")
            continue

        # Plot temperature
        axs[0].plot(
            exp_data['year'].astype('int').values,
            lowess(exp_data['t2m_mean_corrected'].values, exp_data['year'].astype('int').values, frac=frac)[:, 1],
            alpha=0.8, label=exp, color=exp_colors[n]
        )

        # Plot OSA person-days
        axs[1].plot(
            exp_data['year'].astype('int').values,
            lowess(exp_data['osa_person_days_corrected'].values / 1e6, exp_data['year'].astype('int').values,
                   frac=frac)[:, 1],
            alpha=0.8, label=exp, color=exp_colors[n]
        )

        # For YLD and YLL, show uncertainty ranges for extreme scenarios
        if exp in ['ssp126', 'ssp585']:
            # Handle column name differences
            yld_mean_col = 'yld_diff_mean_corrected' if 'yld_diff_mean_corrected' in exp_data.columns else 'yld_prediction_mean'
            yld_min_col = 'yld_diff_min_corrected' if 'yld_diff_min_corrected' in exp_data.columns else 'yld_prediction_min'
            yld_max_col = 'yld_diff_max_corrected' if 'yld_diff_max_corrected' in exp_data.columns else 'yld_prediction_max'
            yll_mean_col = 'yll_diff_corrected' if 'yll_diff_corrected' in exp_data.columns else 'yll_prediction_mean'
            yll_min_col = 'yll_diff_min_corrected' if 'yll_diff_min_corrected' in exp_data.columns else 'yll_prediction_min'
            yll_max_col = 'yll_diff_max_corrected' if 'yll_diff_max_corrected' in exp_data.columns else 'yll_prediction_max'

            # Plot YLD with uncertainty
            axs[2].plot(
                exp_data['year'].astype('int').values,
                lowess(exp_data[yld_mean_col].values / 1000, exp_data['year'].astype('int').values, frac=frac)[:, 1],
                label=exp, color=exp_colors[n], alpha=0.9
            )

            axs[2].fill_between(
                x=exp_data['year'].astype('int').values,
                y1=lowess(exp_data[yld_min_col].values / 1000, exp_data['year'].astype('int').values, frac=frac)[:, 1],
                y2=lowess(exp_data[yld_max_col].values / 1000, exp_data['year'].astype('int').values, frac=frac)[:, 1],
                color=exp_colors[n], alpha=0.1
            )

            # Plot YLL with uncertainty
            axs[3].plot(
                exp_data['year'].astype('int').values,
                lowess(exp_data[yll_mean_col].values / 1000, exp_data['year'].astype('int').values, frac=frac)[:, 1],
                label=exp, color=exp_colors[n], alpha=0.9
            )

            axs[3].fill_between(
                x=exp_data['year'].astype('int').values,
                y1=lowess(exp_data[yll_min_col].values / 1000, exp_data['year'].astype('int').values, frac=frac)[:, 1],
                y2=lowess(exp_data[yll_max_col].values / 1000, exp_data['year'].astype('int').values, frac=frac)[:, 1],
                color=exp_colors[n], alpha=0.1
            )
        else:
            # Handle column name differences
            yld_mean_col = 'yld_diff_mean_corrected' if 'yld_diff_mean_corrected' in exp_data.columns else 'yld_prediction_mean'
            yll_mean_col = 'yll_diff_corrected' if 'yll_diff_corrected' in exp_data.columns else 'yll_prediction_mean'

            # Plot mean lines only for intermediate scenarios
            axs[2].plot(
                exp_data['year'].astype('int').values,
                lowess(exp_data[yld_mean_col].values / 1000, exp_data['year'].astype('int').values, frac=frac)[:, 1],
                label=exp, color=exp_colors[n], alpha=0.9
            )

            axs[3].plot(
                exp_data['year'].astype('int').values,
                lowess(exp_data[yll_mean_col].values / 1000, exp_data['year'].astype('int').values, frac=frac)[:, 1],
                label=exp, color=exp_colors[n], alpha=0.9
            )

    # Style all axes
    for n, a in enumerate([axs[0], axs[1], axs[3], axs[2]]):
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        a.spines.bottom.set_visible(False)
        a.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=True,
        )

    # Add titles
    axs[2].set_title('Years lived with disability \n (x1000)', loc='left', y=1.0, pad=-19, fontsize=8)
    axs[0].set_title('Temperature, °C', loc='left', y=1.0, pad=-12, fontsize=8)
    axs[1].set_title('OSA person-days \n (millions)', loc='left', y=1.0, pad=-18, fontsize=8)
    axs[3].set_title('Years of life lost \n (x1000)', loc='left', y=1.0, pad=-18, fontsize=8)

    # Add legend to first plot
    axs[0].legend(fontsize=6, loc='upper left')

    # Add grid to all axes
    for a in axs:
        a.grid(axis='y', which='major')

    # Add main title
    plt.suptitle(f"{country_name}", fontsize=10)

    # Adjust layout
    plt.subplots_adjust(left=0.125, right=0.864, bottom=0.11, top=0.848, wspace=0.2, hspace=0.16)
    plt.show()


def plot_country_timeseries(country_data, country_name, output_dir=None):
    """
    Plot time series data for a specific country.

    Parameters:
    -----------
    country_data : dict
        Dictionary containing processed data for historical and SSP scenarios
    country_name : str
        Name of the country being plotted
    output_dir : str, optional
        Directory to save the plot. If None, plot is displayed but not saved.

    Returns:
    --------
    dict
        Dictionary containing processed data for the country
    """
    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, sharex='col', figsize=(3.8, 5))
    axs = list(axs.flatten())

    # Color definitions
    exp_colors = ['#003049', '#fcbf49', '#f77f00', '#d62828']

    # Extract historical data
    historical = country_data['historical']

    # Apply smoothing function for visualization
    lowess = sm.nonparametric.lowess
    nyears = 74
    smooth_param = 20
    frac = smooth_param / nyears

    # Plot historical temperature data
    axs[0].plot(historical['year'].values, historical['t2m_mean'].values, 'k', alpha=0.2, label='Historical',
                linestyle='--')
    axs[0].plot(historical['year'].values, historical['t2m_mean_corrected'], 'k', alpha=0.9, label='Historical')

    # Plot historical OSA person-days
    axs[1].plot(historical['year'].values, historical['osa_person_days'].values / 1e6, 'k', alpha=0.2,
                label='Historical', linestyle='--')
    axs[1].plot(historical['year'].values, historical['osa_person_days_corrected'] / 1e6, 'k', alpha=0.9,
                label='Historical')

    # Plot historical YLD (Years Lived with Disability)
    axs[2].plot(historical['year'].values, historical['yld_prediction_mean'] / 1000, 'k', alpha=0.8)
    axs[2].plot(historical['year'].values, historical['yld_diff_mean'].values / 1000, 'k', alpha=0.2,
                label='Historical', linestyle='--')

    # Plot historical YLL (Years of Life Lost)
    axs[3].plot(historical['year'].values, historical['yll_diff'].values / 1000, 'k', alpha=0.2, label='Historical',
                linestyle='--')
    axs[3].plot(historical['year'].values, historical['yll_prediction_mean'] / 1000, 'k', alpha=0.8)

    # Plot each SSP scenario
    for n, exp in enumerate(['ssp126', 'ssp245', 'ssp370', 'ssp585']):
        exp_data = country_data[exp]

        # Plot temperature for all scenarios
        axs[0].plot(
            exp_data['year'].astype('int').values,
            lowess(exp_data['t2m_mean_corrected'].values, exp_data['year'].astype('int').values, frac=frac)[:, 1],
            alpha=0.8, label=exp, color=exp_colors[n]
        )

        # Plot OSA person-days for all scenarios
        axs[1].plot(
            exp_data['year'].astype('int').values,
            lowess(exp_data['osa_person_days_corrected'].values / 1e6, exp_data['year'].astype('int').values,
                   frac=frac)[:, 1],
            alpha=0.8, label=exp, color=exp_colors[n]
        )

        # For YLD and YLL, show details only for extreme scenarios
        if exp in ['ssp126', 'ssp585']:
            # Plot YLD with uncertainty range
            axs[2].plot(
                exp_data['year'].astype('int').values,
                lowess(exp_data['yld_diff_mean_corrected'].values / 1000, exp_data['year'].astype('int').values,
                       frac=frac)[:, 1],
                label=exp, color=exp_colors[n], alpha=0.9
            )

            axs[2].fill_between(
                x=exp_data['year'].astype('int').values,
                y1=lowess(exp_data['yld_diff_min_corrected'].values / 1000, exp_data['year'].astype('int').values,
                          frac=frac)[:, 1],
                y2=lowess(exp_data['yld_diff_max_corrected'].values / 1000, exp_data['year'].astype('int').values,
                          frac=frac)[:, 1],
                color=exp_colors[n], alpha=0.1
            )

            # Plot YLL with uncertainty range
            axs[3].plot(
                exp_data['year'].astype('int').values,
                lowess(exp_data['yll_diff_corrected'].values / 1000, exp_data['year'].astype('int').values, frac=frac)[
                :, 1],
                label=exp, color=exp_colors[n], alpha=0.9
            )

            axs[3].fill_between(
                x=exp_data['year'].astype('int').values,
                y1=lowess(exp_data['yll_diff_min_corrected'].values / 1000, exp_data['year'].astype('int').values,
                          frac=frac)[:, 1],
                y2=lowess(exp_data['yll_diff_max_corrected'].values / 1000, exp_data['year'].astype('int').values,
                          frac=frac)[:, 1],
                color=exp_colors[n], alpha=0.1
            )
        else:
            # Plot mean lines only for intermediate scenarios
            axs[2].plot(
                exp_data['year'].astype('int').values,
                lowess(exp_data['yld_diff_mean_corrected'].values / 1000, exp_data['year'].astype('int').values,
                       frac=frac)[:, 1],
                label=exp, color=exp_colors[n], alpha=0.9
            )

            axs[3].plot(
                exp_data['year'].astype('int').values,
                lowess(exp_data['yll_diff_corrected'].values / 1000, exp_data['year'].astype('int').values, frac=frac)[
                :, 1],
                label=exp, color=exp_colors[n], alpha=0.9
            )

    # Style adjustments for all axes
    for n, a in enumerate([axs[0], axs[1], axs[3], axs[2]]):
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        a.spines.bottom.set_visible(False)
        a.tick_params(which='both', bottom=False, top=False, left=True)
        a.grid(axis='y', which='major')

    # Add titles
    axs[2].set_title('Years lived with disability \n (x1000)', loc='left', y=1.0, pad=-19, fontsize=8)
    axs[0].set_title('Temperature, °C', loc='left', y=1.0, pad=-12, fontsize=8)
    axs[1].set_title('OSA person-days \n (millions)', loc='left', y=1.0, pad=-18, fontsize=8)
    axs[3].set_title('Years of life lost \n (x1000)', loc='left', y=1.0, pad=-18, fontsize=8)

    # Add legend to first plot
    axs[0].legend(fontsize=6, loc='upper left')

    # Adjust layout
    plt.subplots_adjust(left=0.125, right=0.864, bottom=0.11, top=0.848, wspace=0.2, hspace=0.16)


    plt.show()

    return country_data


def plot_country_comparison(comparison_data, year_start=2023, year_end=2098, output_dir=None):
    """
    Create a comparison plot showing health impacts across countries.

    Parameters:
    -----------
    comparison_data : pandas.DataFrame
        Processed and filtered data for comparison
    year_start : int, optional
        Starting year for comparison
    year_end : int, optional
        Ending year for comparison
    output_dir : str, optional
        Directory to save the plot. If None, plot is displayed but not saved.
    """
    # Create figure with subplot mosaic
    fig, ax = plt.subplot_mosaic(
        [['a1', 'a0', 'm']],
        figsize=(3.8, 5)
    )

    # Filter data for specific years
    f = comparison_data.loc[comparison_data['year'].isin([year_start, year_end])]

    # Remove city from Hong Kong if present
    f = f.loc[f['city'] != 'Hong Kong (China)']

    # Sort by YLD per capita
    f = f.sort_values(by=['yld_per', 'country'], ascending=False)

    # Define color palette for scenarios
    palette = {
        'historical': '#073b4c',
        'ssp126': '#003049',
        'ssp245': '#fcbf49',
        'ssp370': '#f77f00',
        'ssp585': '#d62828'
    }

    # Filter data by year and scenario
    historical = f.loc[f['year'] == year_start]
    historical = historical.loc[historical['exp'] == 'historical']

    # Create plots for each scenario from most severe to least
    for scenario in ['ssp585', 'ssp370', 'ssp245', 'ssp126']:
        scenario_data = f.loc[(f['year'] == year_end) & (f['exp'] == scenario)]

        # Plot Years Lived with Disability
        alpha = 1.0 if scenario == 'ssp126' else (0.8 if scenario == 'ssp585' else 0.6)

        sns.barplot(
            x="yld_per", y="country", data=scenario_data, ax=ax['a0'],
            color=palette[scenario], alpha=alpha
        )

        # Plot Years of Life Lost
        sns.barplot(
            x="yll_per", y="country", data=scenario_data, ax=ax['a1'],
            color=palette[scenario], alpha=alpha
        )

        # Plot monetary cost
        sns.barplot(
            x="money_per", y="country", data=scenario_data, ax=ax['m'],
            color=palette[scenario], alpha=alpha
        )

    # Add marks for historical values
    for n, row in historical.iterrows():
        ax['a0'].plot(row['yld_per'], row['country'], "|", markersize=10, color="white")
        ax['a1'].plot(row['yll_per'], row['country'], "|", markersize=10, color="white")
        ax['m'].plot(row['money_per'], row['country'], "|", markersize=10, color="white")

    # Style YLD plot (right side)
    ax['a0'].set_ylabel('')
    ax['a0'].set_yticks(ax['a0'].get_yticks(), ax['a0'].get_yticklabels())
    ax['a0'].spines.right.set_visible(False)
    ax['a0'].spines.bottom.set_visible(False)
    ax['a0'].tick_params(
        which='both', bottom=False, top=True,
        labeltop=True, labelbottom=False, right=False, left=True, labelleft=True
    )
    ax['a0'].grid(axis='x', which='major')
    ax['a0'].set_xlabel('Years lived with disability \nper 100,000/year')
    ax['a0'].xaxis.set_label_position('top')

    # Style YLL plot (left side)
    ax['a1'].set_ylabel('')
    ax['a1'].invert_xaxis()
    ax['a1'].grid(axis='x', which='major')
    ax['a1'].spines.right.set_visible(True)
    ax['a1'].spines.left.set_visible(False)
    ax['a1'].spines.bottom.set_visible(False)
    ax['a1'].tick_params(
        which='both', bottom=False, top=True,
        labeltop=True, labelbottom=False, right=True, left=False, labelleft=False
    )
    ax['a1'].set_xlabel('Years of life lost \nper 100,000/year')
    ax['a1'].xaxis.set_label_position('top')

    # Style monetary cost plot
    ax['m'].invert_xaxis()
    ax['m'].set_ylabel('')
    ax['m'].grid(axis='x', which='major')
    ax['m'].spines.right.set_visible(False)
    ax['m'].spines.left.set_visible(False)
    ax['m'].spines.bottom.set_visible(False)
    ax['m'].tick_params(
        which='both', bottom=False, top=True,
        labeltop=True, labelbottom=False, right=False, left=False, labelleft=False
    )
    ax['m'].set_xlabel('Cost (millions USD) \nper 100,000/year')
    ax['m'].xaxis.set_label_position('top')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=palette['historical'], marker='|', linestyle='None',
               markersize=10, markeredgewidth=2, label=f'{year_start} (Historical)'),
        Line2D([0], [0], color=palette['ssp126'], marker='s', linestyle='None',
               markersize=8, label='SSP1-2.6 (Low emissions)'),
        Line2D([0], [0], color=palette['ssp245'], marker='s', linestyle='None',
               markersize=8, label='SSP2-4.5 (Medium emissions)'),
        Line2D([0], [0], color=palette['ssp370'], marker='s', linestyle='None',
               markersize=8, label='SSP3-7.0 (High emissions)'),
        Line2D([0], [0], color=palette['ssp585'], marker='s', linestyle='None',
               markersize=8, label='SSP5-8.5 (Very high emissions)')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
               ncol=2, fontsize=6)

    # Adjust layout
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.2, top=0.85)
    plt.show()


def plot_cumulative_impact(cumulative_data, output_dir=None):
    """
    Plot cumulative health and economic impacts over time.

    Parameters:
    -----------
    cumulative_data : pandas.DataFrame
        Data frame with cumulative metrics for each scenario
    output_dir : str, optional
        Directory to save the plot. If None, plot is displayed but not saved.
    """
    print(cumulative_data)
    # Create figure with subplot mosaic
    fig, ax = plt.subplot_mosaic(
        [['a0', 'a0', 'a0'],
         ['a1', 'a1', 'a1'],
         ['a2', 'a2', 'a2']],
        figsize=(3.8, 5)
    )

    # Define color palette for scenarios
    palette = {
        'historical': '#073b4c',
        'ssp126': '#003049',
        'ssp245': '#fcbf49',
        'ssp370': '#f77f00',
        'ssp585': '#d62828'
    }

    # Plot cumulative YLL
    sns.lineplot(
        data=cumulative_data, x="year", y="cumsum_yll", hue="exp",
        palette=palette, ax=ax['a0'], legend=False
    )
    ax['a0'].set_ylabel('Cumulative YLL\n(millions)')

    # Plot cumulative YLD
    sns.lineplot(
        data=cumulative_data, x="year", y="cumsum_yld", hue="exp",
        palette=palette, ax=ax['a1'], legend=False
    )
    ax['a1'].set_ylabel('Cumulative YLD\n(millions)')

    # Plot cumulative cost
    sns.lineplot(
        data=cumulative_data, x="year", y="cumsum_money", hue="exp",
        palette=palette, ax=ax['a2'], legend=False
    )
    ax['a2'].set_ylabel('Cumulative Cost\n(trillion USD)')

    # Style all plots
    for axis_key in ['a0', 'a1', 'a2']:
        a = ax[axis_key]
        a.grid(which='major', axis='y', linestyle='--')
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)

        # Only show x-axis label on bottom plot
        if axis_key == 'a2':
            a.set_xlabel('Year')
        else:
            a.set_xlabel('')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=palette['ssp126'], lw=2, label='SSP1-2.6 (Low emissions)'),
        Line2D([0], [0], color=palette['ssp245'], lw=2, label='SSP2-4.5 (Medium emissions)'),
        Line2D([0], [0], color=palette['ssp370'], lw=2, label='SSP3-7.0 (High emissions)'),
        Line2D([0], [0], color=palette['ssp585'], lw=2, label='SSP5-8.5 (Very high emissions)')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.04),
               ncol=2, fontsize=7)

    # Adjust layout
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9, hspace=0.3)


    plt.show()


def plot_point_comparison(comparison_data, selected_years=[2000, 2023, 2050, 2098], output_dir=None):
    """
    Create pointplot comparing health metrics across different years and scenarios.

    Parameters:
    -----------
    comparison_data : pandas.DataFrame
        Processed and filtered data for comparison
    selected_years : list, optional
        Years to include in the point plot
    output_dir : str, optional
        Directory to save the plot. If None, plot is displayed but not saved.
    """
    # Create figure with subplot mosaic
    fig, ax = plt.subplot_mosaic(
        [['a0', 'a0', 'a0'],
         ['a1', 'a1', 'a1'],
         ['a2', 'a2', 'a2']],
        figsize=(3.8, 5),
        height_ratios=[1 / 10, 3 / 10, 6 / 10]
    )

    # Hide the placeholder axis
    ax['a0'].axis('off')

    # Define color palette for scenarios
    palette = {
        'historical': '#073b4c',
        'ssp126': '#003049',
        'ssp245': '#fcbf49',
        'ssp370': '#f77f00',
        'ssp585': '#d62828'
    }

    # Filter data for selected years
    f = comparison_data.loc[comparison_data['year'].isin(selected_years)]

    # Plot YLL per capita
    sns.pointplot(
        data=f, x="year", y="yll_per", hue="exp",
        palette=palette, ax=ax['a1'], legend=False, dodge=0.2
    )
    ax['a1'].grid(which='major', axis='y', linestyle='--')
    ax['a1'].set_ylabel('YLLs per 100,000')
    ax['a1'].set_xlabel('')

    # Style YLL plot
    ax['a1'].spines.right.set_visible(False)
    ax['a1'].spines.top.set_visible(False)

    # Plot YLD per capita
    sns.pointplot(
        data=f, x="year", y="yld_per", hue="exp",
        palette=palette, ax=ax['a2'], legend=False, dodge=0.2
    )
    ax['a2'].grid(which='major', axis='y', linestyle='--')
    ax['a2'].set_ylabel('YLDs per 100,000')
    ax['a2'].set_xlabel('Year')

    # Style YLD plot
    ax['a2'].spines.right.set_visible(False)
    ax['a2'].spines.top.set_visible(False)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=palette['historical'], marker='o', linestyle='-',
               markersize=8, label='Historical'),
        Line2D([0], [0], color=palette['ssp126'], marker='o', linestyle='-',
               markersize=8, label='SSP1-2.6 (Low emissions)'),
        Line2D([0], [0], color=palette['ssp245'], marker='o', linestyle='-',
               markersize=8, label='SSP2-4.5 (Medium emissions)'),
        Line2D([0], [0], color=palette['ssp370'], marker='o', linestyle='-',
               markersize=8, label='SSP3-7.0 (High emissions)'),
        Line2D([0], [0], color=palette['ssp585'], marker='o', linestyle='-',
               markersize=8, label='SSP5-8.5 (Very high emissions)')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.97),
               ncol=2, fontsize=7)

    # Adjust layout
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.85, hspace=0.3)


    plt.show()


def create_all_plots(historical_data, combined_ssp, output_dir="plots"):
    """
    Create and save all plots to the specified directory.

    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Processed historical data
    combined_ssp : pandas.DataFrame
        Combined SSP scenario data
    output_dir : str, optional
        Directory to save all plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Preparing comparison data...")
    # Rename columns for consistent column names
    ssp_data_renamed = combined_ssp.rename(columns={
        'yld_diff_mean_corrected': 'yld_prediction_mean',
        'yld_diff_max_corrected': 'yld_prediction_max',
        'yld_diff_min_corrected': 'yld_prediction_min',
        'yll_diff_corrected': 'yll_prediction_mean',
        'yll_diff_min_corrected': 'yll_prediction_min',
        'yll_diff_max_corrected': 'yll_prediction_max'
    })

    # Combine and prepare data for comparison plots
    comparison_data = pd.concat([historical_data, ssp_data_renamed], join='inner', ignore_index=True)

    # Filter out specific countries
    excluded_countries = [
        'Austria', 'Singapore', 'United Arab Emirates', 'Korea, South', 'India', 'Israel', 'Brazil',
        'China', 'Croatia', 'New Zealand', 'Thailand', 'Turkey'
    ]

    comparison_data = comparison_data.loc[~comparison_data['country'].isin(excluded_countries), :]

    # Add per capita metrics
    comparison_data['yll_per'] = 100000 * comparison_data['yll_prediction_mean'] / comparison_data['population_adult']
    comparison_data['yld_per'] = 100000 * comparison_data['yld_prediction_mean'] / comparison_data['population_adult']
    comparison_data['money_per'] = (100000 * comparison_data['money_cost_corrected'] / comparison_data[
        'population_adult']) / 1e6

    # Replace country names for better display
    country_name_replacements = {
        'United Kingdom': 'UK',
        'Korea, South': 'S.Korea',
        'United Arab Emirates': 'U.A.E',
        'New Zealand': 'N.Z',
        'United States': 'USA',
        'Czech Republic': 'Czechia'
    }

    comparison_data = comparison_data.replace(country_name_replacements)

    print(f"Creating point comparison plot...")
    plot_point_comparison(comparison_data, output_dir=None)

    print(f"Creating country comparison plot...")
    plot_country_comparison(comparison_data, year_start=2023, year_end=2098, output_dir=None)

    plot_country_from_processed_data(historical_data=historical_data,
                                     ssp_data=combined_ssp,
                                     country_name='United Kingdom', output_dir=None)

    print(f"Preparing cumulative data...")
    # Prepare cumulative data
    df_scenarios = comparison_data.loc[comparison_data['year'] > 2022, :]
    df_aggregated = df_scenarios.groupby(['year', 'exp'])[
        ['yll_prediction_mean', 'yld_prediction_mean', 'money_cost_corrected']
    ].sum().reset_index()

    df_aggregated['cumsum_yll'] = df_aggregated.groupby('exp')['yll_prediction_mean'].transform('cumsum') / 1e6
    df_aggregated['cumsum_yld'] = df_aggregated.groupby('exp')['yld_prediction_mean'].transform('cumsum') / 1e6
    df_aggregated['cumsum_money'] = df_aggregated.groupby('exp')['money_cost_corrected'].transform('cumsum') / 1e12

    print(f"Creating cumulative impact plot...")
    print(df_aggregated.loc[df_aggregated['year'].values == 2098, :])

    plot_cumulative_impact(df_aggregated, output_dir=None)

    print(f"All plots saved to {output_dir}/")

def load_processed_data(input_dir="processed_data"):
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
    historical_data = pd.read_csv(f"{input_dir}/historical_predictions.csv")
    ssp126_data = pd.read_csv(f"{input_dir}/ssp126.csv")
    ssp245_data = pd.read_csv(f"{input_dir}/ssp245.csv")
    ssp370_data = pd.read_csv(f"{input_dir}/ssp370.csv")
    ssp585_data = pd.read_csv(f"{input_dir}/ssp585.csv")

    # Also load combined SSP data if available, otherwise create it
    try:
        combined_ssp = pd.read_csv(f"{input_dir}/all_ssp_scenarios.csv")
    except FileNotFoundError:
        combined_ssp = pd.concat([ssp126_data, ssp245_data, ssp370_data, ssp585_data], ignore_index=True)

    return historical_data, ssp126_data, ssp245_data, ssp370_data, ssp585_data, combined_ssp

if __name__ == "__main__":
    historical_data, ssp126_data, ssp245_data, ssp370_data, ssp585_data, combined_ssp = load_processed_data(input_dir='metadata/figure_table_health_economics')
    # Create all plots
    create_all_plots(historical_data, combined_ssp)

