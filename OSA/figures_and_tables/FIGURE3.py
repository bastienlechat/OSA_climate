import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from pathlib import Path
import geopandas as gpd
plt.style.use('nature.mplstyle')

def get_country_city_mapping():
    """Get standard mapping between countries and representative cities"""
    return {
        'Germany': 'Berlin',
        'France': 'Paris',
        'United Kingdom': 'London',
        'United States': 'New_York',
        'Switzerland': 'Zurich',
        'Japan': 'Tokyo',
        'Italy': 'Rome',
        'Netherlands': 'Amsterdam',
        'Austria': 'Vienna',
        'Spain': 'Madrid',
        'Finland': 'Helsinki',
        'Belgium': 'Brussels',
        'Sweden': 'Stockholm',
        'Australia': 'Sydney',
        'Canada': 'Toronto',
        'Norway': 'Oslo',
        'Poland': 'Warsaw',
        'Denmark': 'Copenhagen',
        'Ireland': 'Dublin',
        'Hungary': 'Budapest',
        'Czech Republic': 'Prague',
        'Portugal': 'Lisbon',
        'Romania': 'Bucharest',
        'Greece': 'Athens',
        'Luxembourg': 'Luxembourg',
        'China': 'Shanghai',
        'New Zealand': 'Auckland',
        'Estonia': 'Tallinn',
        'Russia': 'Moscow',
        'Slovakia': 'Bratislava',
        'Croatia': 'Zagreb',
        'Singapore': 'Singapore',
        'Thailand': 'Bangkok',
        'India': 'Kolkata',
        'Turkey': 'Istanbul',
        'United Arab Emirates': 'Dubai',
        'Korea, South': 'Seoul',
        'Bulgaria': 'Sofia',
        'Mexico': 'Mexico_City',
        'Brazil': 'Sao_Paulo',
        'Israel': 'Jerusalem',
    }


def prepare_country_data_for_map(cities_data):
    """Prepare country data for mapping"""
    # Fix country names for mapping
    cities_fixed = cities_data.replace({
        'United States': 'United States of America',
        'Congo, Democratic Republic of the': 'Dem. Rep. Congo',
        'Dominican Republic': 'Dominican Rep.',
        'Bahamas, The': 'Bahamas',
        'Central African Republic': 'Central African Rep.',
        'Congo, Republic of the': 'Congo',
        'Palestinian Territory': 'Palestine',
        'Myanmar (Burma)': 'Myanmar',
        'Korea, South': 'South Korea',
        'Czech Republic': 'Czechia',
        'Bosnia and Herzegovina': 'Bosnia and Herz.',
        'North Macedonia': 'Macedonia',
    })

    # Load world map data
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Remove Antarctica and Seven seas
    drop_idxs = world["continent"].isin([
        "Antarctica",
        "Seven seas (open ocean)"
    ])
    world = world.drop(world[drop_idxs].index)

    # Fix country names for merge
    world = world.replace({"CÃ´te d'Ivoire": "Cote d'Ivoire"})

    # Merge cities data with world map
    world = world.merge(cities_fixed, how='left', left_on='name', right_on='name_country')

    # Categorize risk ratios
    world['or_count'] = pd.cut(
        world['est_wsa_binary'],
        bins=[1, 1.125, 1.25, 1.50, 1.75, np.inf],
        labels=['1-1.1', '1.1-1.3', '1.3-1.5', '1.5-1.7', '1.7+']
    )

    return world


def prepare_country_comparison_data(cities_data):
    """Prepare country data for comparison plot"""
    country = cities_data.loc[cities_data['name_country'] != 'Hong Kong (China)']

    # Simplify country names for display
    country = country.replace({
        'United Kingdom': 'UK',
        'Korea, South': 'S.Korea',
        'United Arab Emirates': 'U.A.E',
        'New Zealand': 'N.Z',
        'United States': 'USA'
    })

    # Assign colors based on confidence intervals
    country['colors'] = '#9A7065'  # Default color
    country.loc[country['lb_wsa_binary'] < 1, 'colors'] = '#70659A'
    country.loc[country['hb_wsa_binary'] < 1, 'colors'] = '#659A70'

    return country


# Plotting functions
def plot_risk_ratio_curve(ax, wsa_data, wsa_severe=None, temp_data=None, quantiles=None):
    """Plot risk ratio curve with option for severe data and histogram"""
    # Plot standard curve
    ax.plot(wsa_data['xvar'], wsa_data['est'], color='#606c38', label='WSA')
    ax.fill_between(wsa_data['xvar'], y1=wsa_data['lb'], y2=wsa_data['hb'],
                    color='#606c38', alpha=0.2)

    # Plot severe curve if provided
    if wsa_severe is not None:
        ax.plot(wsa_severe['xvar'], wsa_severe['est'], color='#6C3860',
                label='WSA Severe', linestyle=(0, (5, 10)))
        ax.fill_between(wsa_severe['xvar'], y1=wsa_severe['lb'], y2=wsa_severe['hb'],
                        color='#6C3860', alpha=0.2)

    ax.set_ylabel('Risk ratio')

    # Add histogram if temp data provided
    if temp_data is not None:
        axhist = ax.twinx()
        axhist.hist(temp_data, bins=100, color='#38606C', alpha=0.4)

        # Add vertical line at 99th percentile
        if quantiles is not None:
            axhist.axvline(x=quantiles['q99'], linestyle='--', color='k', alpha=0.8)

        # Clean up histogram axis
        axhist.set_ylim(axhist.get_ylim()[0], axhist.get_ylim()[1] * 3)
        if quantiles is not None:
            axhist.set_xlim(quantiles['q001'], quantiles['q999'])
        axhist.spines.right.set_visible(False)
        axhist.tick_params(right=False, labelright=False)

        for spine in axhist.spines.values():
            spine.set_visible(False)

    # Clean up main axis
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.grid(visible=True)
    ax.spines.bottom.set_visible(False)
    ax.spines.left.set_visible(False)

    if quantiles is not None:
        ax.set_xlim(quantiles['q001'], quantiles['q999'])

    ax.tick_params(which='both', bottom=False, top=False, left=False, labelleft=True)


def plot_gdp_comparison(ax, gdp_data, temp_quantile_99):
    """Plot GDP comparison at the 99th temperature percentile"""
    # Extract values at the 99th percentile
    gdp_values = []

    for gdp_key, data in gdp_data.items():
        # Find index closest to 99th percentile
        idx = np.abs(data['xvar'] - temp_quantile_99).argmin()

        gdp_values.append({
            'name': gdp_key,
            'est': data['est'][idx],
            'lb': data['lb'][idx],
            'hb': data['hb'][idx]
        })

    # Sort by GDP category
    gdp_values.sort(key=lambda x: x['name'])

    # Plot
    bar_colors = ['#e56b6f', '#b56576', '#355070']

    for gdp, color in zip(gdp_values, bar_colors):
        err = gdp['hb'] - gdp['est']  # Upper error
        ax.errorbar(gdp['name'], gdp['est'], err, fmt="o", color=color)

    # Format axis
    ax.grid(visible=True, axis='y')
    ax.set_ylabel('RR at T99')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(which='both', bottom=False, top=False, left=True, labelleft=True)


def plot_world_map(ax, world_data, zoom=False):
    """Plot world map with risk ratio data"""
    world_data.plot(
        column='or_count',
        ax=ax,
        cmap='RdYlGn_r',
        legend=not zoom,  # Only show legend for full map
        edgecolor="#737373",
        linewidth=0.5,
        missing_kwds={'color': '#d9d9d9'},
        legend_kwds={'loc': "lower left"}
    )

    # Format axis
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelleft=False,
        labelbottom=False
    )

    # Set limits for zoom
    if zoom:
        ax.set_xlim(-15, 35)
        ax.set_ylim(35, 75)


def plot_country_comparison(ax, country_data):
    """Plot country comparison with error bars"""
    # Extract data
    y_values = country_data['est_wsa_binary'].values
    x_values = country_data['name_country'].values
    colors = country_data['colors'].values
    lower_errors = np.abs(country_data['lb_wsa_binary'].values - country_data['est_wsa_binary'].values)
    upper_errors = np.abs(country_data['hb_wsa_binary'].values - country_data['est_wsa_binary'].values)

    # Replace underscores with spaces in country names
    x_values = [name.replace('_', ' ') for name in x_values]

    # Plot each country
    for x, y, lower, upper, c in zip(x_values, y_values, lower_errors, upper_errors, colors):
        ax.errorbar(y, x, xerr=[[lower], [upper]], color=c, fmt='o')

    # Format axis
    ax.set_xlabel('Risk ratio at T99')
    ax.axvline(x=1, color='k', linestyle='--')
    ax.set_xticks([3, 2, 1])
    ax.grid(visible=True, axis='x', linestyle='dotted', color='k', alpha=0.8)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(True)
    ax.spines.left.set_visible(False)

    ax.yaxis.tick_right()
    ax.invert_xaxis()


def create_full_figure(data, output_path=None):
    """Create complete figure with all plots"""
    # Extract data
    wsa_curve = pd.DataFrame(data['wsa_curve'])
    temp_histogram = np.array(data['temp_histogram'])
    temp_quantiles = data['temp_quantiles']

    wsa_severe = None
    if 'wsa_severe' in data:
        wsa_severe = pd.DataFrame(data['wsa_severe'])

    gdp_data = {k: pd.DataFrame(v) for k, v in data['gdp_data'].items()}

    # Load cities data
    cities = pd.DataFrame.from_dict(data['cities_data'])

    # Prepare world map data
    world_data = prepare_country_data_for_map(cities)

    # Prepare country comparison data
    country_data = prepare_country_comparison_data(cities)

    # Create figure
    fig, ax = plt.subplot_mosaic(
        [['a', 'a', 'g2', 'a3'],
         ['a2', 'a2', 'a2', 'a3'],
         ['a2', 'a2', 'a2', 'a3'],
         ['a4', 'a4', 'a4', 'a3']],
        figsize=(7.2, 5)
    )

    # Set titles
    ax['a'].set_title('a', loc='left', fontweight='bold', horizontalalignment='left')
    ax['g2'].set_title('b', loc='left', fontweight='bold', horizontalalignment='left')
    ax['a2'].set_title('c', loc='left', fontweight='bold', horizontalalignment='left')
    ax['a3'].set_title('d', loc='left', fontweight='bold', horizontalalignment='left')

    # Plot risk ratio curve
    plot_risk_ratio_curve(ax['a'], wsa_curve, wsa_severe, temp_histogram, temp_quantiles)

    # Plot GDP comparison
    plot_gdp_comparison(ax['g2'], gdp_data, temp_quantiles['q99'])

    # Plot world map
    plot_world_map(ax['a2'], world_data)

    # Plot zoomed Europe map
    plot_world_map(ax['a4'], world_data, zoom=True)

    # Plot country comparison
    plot_country_comparison(ax['a3'], country_data)

    # Adjust layout
    plt.tight_layout()

    plt.show()


    return fig

def load_minimal_data(data_path):
    """Load previously saved minimal data for plotting

    Parameters:
    -----------
    data_path : str
        Path to the directory containing the saved minimal data

    Returns:
    --------
    dict
        Dictionary containing all the data needed for plotting
    """
    import json
    import os

    # Check if the JSON file exists
    json_path = os.path.join(data_path, "plot_data.json")
    if os.path.exists(json_path):
        # Load from JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded data from {json_path}")
        return data

    # If JSON file doesn't exist, try loading from individual CSV files
    print(f"JSON file not found at {json_path}, attempting to load from CSV files...")
    data = {}

    # Load cities data
    cities_path = os.path.join(data_path, "cities_data.csv")
    if os.path.exists(cities_path):
        cities = pd.read_csv(cities_path)
        data['cities_data'] = cities.to_dict()
    else:
        raise FileNotFoundError(f"Cities data not found at {cities_path}")

    # Load WSA curve data
    wsa_path = os.path.join(data_path, "wsa_curve.csv")
    if os.path.exists(wsa_path):
        wsa_curve = pd.read_csv(wsa_path)
        data['wsa_curve'] = {
            'xvar': wsa_curve['xvar'].tolist(),
            'est': wsa_curve['est'].tolist(),
            'lb': wsa_curve['lb'].tolist(),
            'hb': wsa_curve['hb'].tolist()
        }
    else:
        raise FileNotFoundError(f"WSA curve data not found at {wsa_path}")

    # Load temperature quantiles
    quantiles_path = os.path.join(data_path, "temp_quantiles.csv")
    if os.path.exists(quantiles_path):
        quantiles = pd.read_csv(quantiles_path).iloc[0].to_dict()
        data['temp_quantiles'] = quantiles
    else:
        raise FileNotFoundError(f"Temperature quantiles not found at {quantiles_path}")

    # Load temperature histogram data
    hist_path = os.path.join(data_path, "temp_histogram.csv")
    if os.path.exists(hist_path):
        temp_hist = pd.read_csv(hist_path)
        data['temp_histogram'] = temp_hist['temperature'].tolist()
    else:
        raise FileNotFoundError(f"Temperature histogram data not found at {hist_path}")

    # Load GDP data
    data['gdp_data'] = {}
    for i in range(1, 4):
        gdp_path = os.path.join(data_path, f"GDP{i}_curve.csv")
        if os.path.exists(gdp_path):
            gdp_curve = pd.read_csv(gdp_path)
            data['gdp_data'][f'GDP{i}'] = {
                'xvar': gdp_curve['xvar'].tolist(),
                'est': gdp_curve['est'].tolist(),
                'lb': gdp_curve['lb'].tolist(),
                'hb': gdp_curve['hb'].tolist()
            }
        else:
            print(f"Warning: GDP{i} curve data not found at {gdp_path}")

    # Check for severe WSA data
    severe_path = os.path.join(data_path, "wsa_severe_curve.csv")
    if os.path.exists(severe_path):
        severe_curve = pd.read_csv(severe_path)
        data['wsa_severe'] = {
            'xvar': severe_curve['xvar'].tolist(),
            'est': severe_curve['est'].tolist(),
            'lb': severe_curve['lb'].tolist(),
            'hb': severe_curve['hb'].tolist()
        }

    print(f"Successfully loaded data from CSV files in {data_path}")
    return data

def main():
    # Output path
    output_path = 'metadata/figure3'

    # Load previously saved data
    data = load_minimal_data(output_path)
    print(f"Data loaded from {output_path}")

    # Create figure from data
    create_full_figure(data, output_path)
    print(f"Figures saved to {output_path}")


if __name__ == '__main__':
    main()