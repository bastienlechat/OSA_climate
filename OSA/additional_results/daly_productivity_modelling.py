import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('rr_curves_main.xlsx')
df = df.loc[df['city']=='Paris',:]

#### Plot RR curves for France
fig, ax = plt.subplots()
ax.plot(df['xvar'], df['est'], color='#606c38', label='WSA')
ax.fill_between(df['xvar'], y1=df['lb'], y2=df['hb'],
                color='#606c38', alpha=0.2)

ax.axvline(x=df['hbt'].values[0], linestyle='--', color='r', alpha=0.6)
ax.axhline(y=1, linestyle='-', color='k', alpha=0.8)
ax.set_title("RR curve for France")
ax.set_xlabel("Temperature, C")
ax.set_ylabel("RR of nightly OSA")
ax.grid(True)
plt.show()

rr_curve_map = df[['xvar', 'est']]

#### Plot 2023 temperature vs historical temperature
dt = pd.read_excel('france_temp_2023.xlsx')
fig, ax = plt.subplots()
ax.plot(dt['day_of_year'], dt['t2m_mean'], color='red', label='2023')
ax.plot(dt['day_of_year'], dt['historical'], color='k', label='1950-1990')
ax.set_title("France: 2023 vs historical temperature")
ax.set_ylabel("Temperature, C")
ax.set_xlabel("Day of year")
ax.grid(True)
ax.legend(loc='best')
plt.show()


### Matchin rr curves to historical and measured temperatures
# Note: RR curves were estimated based on historical values (between 0.1th percentiles and 99.9th percentiles, hence
# when we use merge_asof, values outside of boundaries are coded as maximum/minimum observed values. This is a conservative
# choice, which is likely to lead to underestimation of the real impact of climate change in scenarios where unobserved
# high temperatures are likely to occur.
#
# See below for discussion on the subject:
#
# Vicedo-Cabrera, A. M., et al. (2019). "Hands-on Tutorial on a Modeling Framework for Projections of
# Climate Change Impacts on Health." Epidemiology 30(3): 321-329.

dt = dt.sort_values(by='t2m_mean')
dt = pd.merge_asof(dt, rr_curve_map, left_on='t2m_mean', right_on='xvar')
dt = dt.rename(columns={'est': 'rr_actual'})
dt = dt.sort_values(by='historical')
dt = pd.merge_asof(dt, rr_curve_map, left_on='historical', right_on='xvar')
dt = dt.rename(columns={'est': 'rr_hist'})
dt = dt.sort_values(by=['day_of_year'])

##### DALY modelling
france_gdp = 44460.82
adult_popultion = 52362442
osa_prev = 0.246
yll_mva_counts = 163837
death_counts_mva = 4053
death_mva_rate = 5.5

##
disability_weight = 0.08
rr_mva = 2 #%100

######### YLL ############
# YLL calculation are based on the global burden of disease
print('----------------------------------------')
print('---DALY modelling FRANCE 2023')
print('----------------------------------------')
rle = yll_mva_counts / death_counts_mva
annual_rate_mva_mortality = death_mva_rate / 100000
annual_rate_mva6_mortality = annual_rate_mva_mortality * rr_mva
daily_excess_mortality = (annual_rate_mva6_mortality - annual_rate_mva_mortality) / 365
excess_mortality_hist = dt['rr_hist'].values * osa_prev * adult_popultion * daily_excess_mortality
excess_mortality_actual = dt['rr_actual'].values * osa_prev * adult_popultion * daily_excess_mortality

yll_hist = excess_mortality_hist * rle
yll_actual = excess_mortality_actual * rle
yll_diff = yll_actual - yll_hist

print('Total number of YLL for 2023: {:.1f}'.format(np.sum(yll_diff))) #The number is slightly lower than manuscript
# since estimates were smoothed across year in the main analysis (but here single year)
print('YLL rates per 100,000 for 2023: {:.1f}'.format(100000*np.sum(yll_diff)/adult_popultion))


######### YLD ############
osa_person_days = (dt['rr_actual'].values - dt['rr_hist'].values) * osa_prev * adult_popultion
yld_diff = osa_person_days * disability_weight / 365

print('Total number of temperature-induced OSA person-days for 2023: {:.1f}'.format(np.sum(osa_person_days)))
print('Total number of YLD for 2023: {:.1f}'.format(np.sum(yld_diff)))
print('YLD rates per 100,000 for 2023: {:.1f}'.format(100000*np.sum(yld_diff)/adult_popultion))


#######################################
######### Health economics ############
print('----------------------------------------')
print('--- Productivity modelling FRANCE 2023')
print('----------------------------------------')

n_working_days = 235
absenteism_rate = 5/n_working_days
presenteism_rate = 0.068
gdpe = 127281.7
labor_force = 25722812.46525 #accounts for part-time labor

osa_person_days_labor_force = (dt['rr_actual'].values - dt['rr_hist'].values) * osa_prev * labor_force
absenteism_days = osa_person_days_labor_force * absenteism_rate * 235/365
presenteism_days = osa_person_days_labor_force * presenteism_rate * 235/365
labor_loss_days = absenteism_days + presenteism_days

print('Total number of days of absenteism for 2023: {:.1f}'.format(np.sum(absenteism_days)))
print('Total number of days of presenteism for 2023: {:.1f}'.format(np.sum(presenteism_days)))
print('Total number of days of labor loss for 2023: {:.1f}'.format(np.sum(labor_loss_days))) #8 millions days
print('Equivalent USD: {:.1f}'.format(np.sum(labor_loss_days)*gdpe/365 /1000)) #2.8 billions


