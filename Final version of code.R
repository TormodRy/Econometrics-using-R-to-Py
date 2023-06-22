##### Converting R to Python 

# Python has a lot of packages meant to be used for data manipulation. Its main use case is still for 
# making model. R is designed for statistics and for this reason has a really good visuals. These
# differences result in that the capabilities of R and python are quit different. The visuals of graphs, the
# look of the data frame are strengths of R. While these are better in python than in many other
# languages, R has a significant advantage in these areas. This results in that trying to convert R code to
# an equivalent in python is quit challenging, and is difficult to produce the same level of quality that R
# is able to produce. 

# To make python work as well as possible using jupyter notebook might be the best idea. The visuals in 
# jupyter are quit good. Running python code in Rstudio can result in a bad experience as the graphs 
# look bad. To see the data frame from the python code, Rstudios work pretty well.

# For this reason i was not able to make the second half of the task work as intended. I still have
#  provided some of the code that I have used to try to make it work. Some of the lines work, but not
#  the entirety of the code.

# things to do to make it work
# 1) find the bf2017.dta file in compute by changing line 60(or around that)
# 2) Recommend using Jupyter notebook to read run the code.
#   (remember to keep the terminal running) 
# 
# #First part is to install packages. If you run jupyter, do not comment them out if in jupyter
# # In python the most important packages would be pandas, numpy and matplot.
# !pip install pandas
# !pip install matplotlib
# !pip install statsmodels
# !pip install linearmodels
# !pip install pyreadstat
# !pip install datetime
# !pip install scikit-learn
# # Here I use the normal keywords such as pd, np and plt.
# SHould never be commented out
from linearmodels.panel import PanelOLS, RandomEffects
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pyreadstat


# PART ONE
# These are the functions used to reverse vector. 
# This part is to estimate errors
# It will be used in the later parts of the paper


# function for reverse cumulative sum of vector
# summing up from end to start
def revcumsum(x):# function of x. It will be possible to use for any function later
    return x[::-1].cumsum()[::-1] #reversing order sum

# function to calculate standard errors of cumulative sum
# takes the matrix and sums things together
def secumsum(vcov): 
    L = vcov.shape[0]
    se = []
    for i in range(1, L+1):
        a = [1] * i
        V = a @ vcov.iloc[0:i, 0:i] @ a
        se.append(V**0.5)
    return se

# function to calculate standard errors of reverse cumulative sum
# summing up from end to start
def serevcumsum(vcov):
    L = vcov.shape[0]
    se = []
    for i in range(L, 0, -1):
        a = [1] * (L-i+1)
        V = a @ vcov.iloc[i-1:L, i-1:L] @ a
        se.append(V**0.5)
    return se


#Read data
# This part must be set to the right place in storage. 
# we are reading the bf2017.dta file
bf2017, _ = pyreadstat.read_dta(r"C:\Users\tormo\Jupyter for econometrics\Other files\bf2017.dta")


# create period variable from year and month in R 'yearmon' format
# THis does not work as well as in R. I made the variable, but could not make it look as in R dataframe
bf2017["yearmonth"] = pd.to_datetime(bf2017["year"].astype(str) + bf2017["month"].astype(str),format = "%Y%m")



# create covariate used in Baker and Fradkin (2017)
# simple math to create new variable corresponding to relevant observations
bf2017['frac_total_ui'] = (bf2017['cont_claims'] + bf2017['init_claims']) / bf2017['population']
print(bf2017.columns) # shows what columns that are in the data


# create treatment adoption indicator
# increase by at least 13 weeks
# This part is about cleaning the data 
bf2017['D_PBD_incr'] = bf2017['PBD'].diff().where(lambda x:x>=13, 0).astype(int)
# replace missing first treatment with zero (implicitly assumed by BF2017)
bf2017['D_PBD_incr'] = bf2017['D_PBD_incr'].fillna(0) #fillna changes the variables with NaN to 0

# create 3 leads 
# Shifting the variables -3, -2 and -1
bf2017['F3_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-3) 
bf2017['F2_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-2) 
bf2017['F1_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-1) 

# create 4 lags
# Shifting the variables -3, -2 and -1
bf2017['L1_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-1)
bf2017['L2_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-2)
bf2017['L3_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-3)
bf2017['L4_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-4)

# replace missing lags with zero (implicitly assumed by BF2017)
# same as earlier, fillna does NaN to 0 
bf2017['L1_D_PBD_incr'] = bf2017['L1_D_PBD_incr'].fillna(0)
bf2017['L2_D_PBD_incr'] = bf2017['L2_D_PBD_incr'].fillna(0)
bf2017['L3_D_PBD_incr'] = bf2017['L3_D_PBD_incr'].fillna(0)
bf2017['L4_D_PBD_incr'] = bf2017['L4_D_PBD_incr'].fillna(0)



# Making a variable for GJSI with log
# New variable of GJSI in log form
# use of numpy, it is the natural logarithm not log 
bf2017['log_GJSI'] = np.log(bf2017['GJSI'])
#############################################################################

bf2017= bf2017.dropna() # making sure of no problems from other variables
print(bf2017.dtypes)
# want to make sure all data is float. Only futher cleaning of data
# since state and yearmonth is str, we drop them and add them back
cols_to_convert = bf2017.columns.drop(['state_abb', "yearmonth"]) 
bf2017_float = bf2017[cols_to_convert].astype(float)
bf2017_float['state_abb'] = bf2017["state_abb"]
bf2017_float["yearmonth"] = bf2017["yearmonth"]
bf2017 = bf2017_float
bf2017['yearmonth'] = bf2017['yearmonth'].astype(str)

bf2017['state_abb'] = bf2017['state_abb'].astype(str)
print(bf2017.dtypes)
bf2017.dtypes

#############################################################################
# test om funksjon fungerer istedenfor

# This is the simple approch to the data
# regression within function. use a subset of data and drop dependent variable from X
# leaves dependent to equal the log_GJSI column
def estimate_fixed_effects_model(bf2017):
    subset_bf2017 = bf2017.loc[bf2017['year'] <= 2011, ['log_GJSI', 'F3_D_PBD_incr', 'F2_D_PBD_incr', 'F1_D_PBD_incr', 'D_PBD_incr', 'L1_D_PBD_incr', 'L2_D_PBD_incr', 'L3_D_PBD_incr', 'L4_D_PBD_incr', 'frac_total_ui']]
    x = subset_bf2017.drop(['log_GJSI'], axis = 1 )
    y = subset_bf2017['log_GJSI']
    print(subset_bf2017.columns)
# print(x)
# print(subset_bf2017.columns)  

    model = LinearRegression()
    model.fit(x,y)

    coefficients = model.coef_  # Coefficients of the linear regression line
    intercept = model.intercept_  # Intercept of the linear regression line
    return model, coefficients, intercept

result_model, result_coefficients, result_intercept = estimate_fixed_effects_model(bf2017)
estimate_fixed_effects_model(bf2017)

# The second approch to the data. This will be the way of solving further problems

################################## den riktige som fungerer VIKTIG
# function that takes in the data set
# uses the sm.OLS method
# show the regression result using a summary function instead

# f??rste plt
def test(bf2017):
    subset_bf2017 = bf2017.loc[bf2017['year'] <= 2011, ['log_GJSI', 'F3_D_PBD_incr', 'F2_D_PBD_incr', 'F1_D_PBD_incr', 'D_PBD_incr', 'L1_D_PBD_incr', 'L2_D_PBD_incr', 'L3_D_PBD_incr', 'L4_D_PBD_incr', 'frac_total_ui']]
    x = subset_bf2017.drop(['log_GJSI'], axis=1)
    y = subset_bf2017['log_GJSI']
    model = sm.OLS(y, x)
    result = model.fit()

    summary = result.summary()

    beta_coeffs = result.params # takes the betas out of the regression into a matrix
    beta_coeffs = beta_coeffs[:-1]  # dropping the frac_total_ui from the list
    std_errors = result.bse # takes the standard errors out of the regression into a matrix
    std_errors = std_errors[:-1]  # dropping the frac_total_ui from the list
  # print(std_errors) # to see the matrix 
  # print(beta_coeffs)

    x = range(-3, 5)  # Months relative to reform

    # plt to view the graphs. They will not look like R. Best viewed in my opion in jupyter notebook.
    # Rstudio can make it look incorrect
    plt.figure(figsize=(8, 4)) 
    
    plt.plot(x, beta_coeffs, color='darkblue', marker='o', linestyle='-')
    plt.errorbar(x, beta_coeffs, yerr=1.96*std_errors, color='darkblue', fmt='none')
    plt.axhline(0, color='black')  # Horizontal line at y=0
    plt.axvline(-0.5, color='black', linestyle='dashed')  # Vertical line at x=-0.5
    
    plt.xlabel("Months relative to reform")
    plt.ylabel("Effect on log search intensity")
    plt.title("Beta Coefficients")
    
    plt.xlim(-3.5, 4.5)
    plt.ylim(-0.02, 0.04)
    plt.xticks(range(-3, 5))
    plt.yticks([-0.02, 0, 0.02, 0.04])
    
    plt.grid(True)
    
    plt.show()
    
    return result

test(bf2017)


# DEL 1 SLUTT


# DEL 2 



# Create a new column 'D_PBD_decr' based on condition
# Here a new variable is created out of PBD based on if the contions hold.
bf2017['D_PBD_decr'] = np.where(bf2017['PBD'].diff(1) <= -7, 1, 0)


# replace missing first treatment with zero (implicitly assumed by BF2017)
bf2017['D_PBD_decr'] = bf2017['D_PBD_decr'].fillna(0) #fillna changes the variables with NaN to 0


# from the variable just created we produce lags of the variable
# Create 3 lead columns
bf2017['F3_D_PBD_decr'] = bf2017['D_PBD_decr'].shift(-3)
bf2017['F2_D_PBD_decr'] = bf2017['D_PBD_decr'].shift(-2)
bf2017['F1_D_PBD_decr'] = bf2017['D_PBD_decr'].shift(-1)

# Create 4 lag columns
bf2017['L1_D_PBD_decr'] = bf2017['D_PBD_decr'].shift(1)
bf2017['L2_D_PBD_decr'] = bf2017['D_PBD_decr'].shift(2)
bf2017['L3_D_PBD_decr'] = bf2017['D_PBD_decr'].shift(3)
bf2017['L4_D_PBD_decr'] = bf2017['D_PBD_decr'].shift(4)



# replace missing lags with zero (implicitly assumed by BF2017)
bf2017['L1_D_PBD_decr'] = bf2017['L1_D_PBD_decr'].fillna(0)
bf2017['L2_D_PBD_decr'] = bf2017['L2_D_PBD_decr'].fillna(0)
bf2017['L3_D_PBD_decr'] = bf2017['L3_D_PBD_decr'].fillna(0)
bf2017['L4_D_PBD_decr'] = bf2017['L4_D_PBD_decr'].fillna(0)

# replace missing lags with zero (implicitly assumed by BF2017)
bf2017['F3_D_PBD_decr'] = bf2017['F3_D_PBD_decr'].fillna(0)
bf2017['F2_D_PBD_decr'] = bf2017['F2_D_PBD_decr'].fillna(0)
bf2017['F1_D_PBD_decr'] = bf2017['F1_D_PBD_decr'].fillna(0)

# andre plt
def test2(bf2017):
    subset_bf2017 = bf2017.loc[bf2017['year'] >= 2012, ['log_GJSI', 'F3_D_PBD_decr', 'F2_D_PBD_decr', 'F1_D_PBD_decr', 'D_PBD_decr', 'L1_D_PBD_decr', 'L2_D_PBD_decr', 'L3_D_PBD_decr', 'L4_D_PBD_decr', 'frac_total_ui']]
    x = subset_bf2017.drop(['log_GJSI'], axis=1)
    y = subset_bf2017['log_GJSI']
    model = sm.OLS(y, x)
    result = model.fit()

    summary = result.summary()

    beta_coeffs = result.params
    beta_coeffs = beta_coeffs[:-1]  # dropping the frac_total_ui from the list
    std_errors = result.bse
    std_errors = std_errors[:-1]  # dropping the frac_total_ui from the list
    
    x = range(-3, 5)  # Months relative to reform
    
    plt.figure(figsize=(8, 4))
    
    plt.plot(x, beta_coeffs, color='darkblue', marker='o', linestyle='-')
    plt.errorbar(x, beta_coeffs, yerr=1.96*std_errors, color='darkblue', fmt='none')
    plt.axhline(0, color='black')  # Horizontal line at y=0
    plt.axvline(-0.5, color='black', linestyle='dashed')  # Vertical line at x=-0.5
    
    plt.xlabel("Months relative to reform")
    plt.ylabel("Effect on log search intensity")
    plt.title("Beta Coefficients")
    
    plt.xlim(-3.5, 4.5)
    plt.ylim(-0.4, 1)
    plt.xticks(range(-3, 5))
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    
    plt.grid(True)
    
    plt.show()

    return result

test2(bf2017)


# ------------------------------------------------------------------------------
# Figure B.1, Panel B, right
# multiple treatments of identical intensities (Schmidheiny/Siegloch case 2)
# effect window from -3 to +4
# estimation with distributed-lags in levels
# ------------------------------------------------------------------------------

def test3(bf2017):
  
    subset_bf2017 = bf2017.loc[bf2017['year'] >= 2012, ['log_GJSI', 'F2_D_PBD_decr', 'F1_D_PBD_decr', 'D_PBD_decr', 'L1_D_PBD_decr', 'L2_D_PBD_decr', 'L3_D_PBD_decr', 'L4_D_PBD_decr', 'frac_total_ui']]
    x = subset_bf2017.drop(['log_GJSI'], axis=1)
    y = subset_bf2017['log_GJSI']
    model = sm.OLS(y, x)
    result = model.fit()

    summary = result.summary()

    beta_coeffs = result.params
    beta_coeffs = beta_coeffs[:-1]  # dropping the frac_total_ui from the list
    std_errors = result.bse
    std_errors = std_errors[:-1]  # dropping the frac_total_ui from the list
    
    x = range(-2, 5)  # Months relative to reform
    
    plt.figure(figsize=(8, 4))
    
    plt.plot(x, beta_coeffs, color='darkblue', marker='o', linestyle='-')
    plt.errorbar(x, beta_coeffs, yerr=1.96*std_errors, color='darkblue', fmt='none')
    plt.axhline(0, color='black')  # Horizontal line at y=0
    plt.axvline(-0.5, color='black', linestyle='dashed')  # Vertical line at x=-0.5
    
    plt.xlabel("Months relative to reform")
    plt.ylabel("Effect on log search intensity")
    plt.title("Beta Coefficients")
    
    plt.xlim(-2.5, 4.5)
    plt.ylim(-0.4, 1)
    plt.xticks(range(-3, 5))
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    
    plt.grid(True)
    
    plt.show()

    return result

test3(bf2017)

# ------------------------------------------------------------------------------
# Figure B.1, Panel B, left 
# alternative estimation with event study specification
# ------------------------------------------------------------------------------

# overwrite treatment adoption indicators from above because of assumed zeros for NAs

# # Create treatment adoption indic ator
# bf2017['D_PBD_incr'] = np.where(np.diff(bf2017['PBD'], 1) >= 13, 1, 0)
# 
# # Create 3 leads
# bf2017['F3_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-3)
# bf2017['F2_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-2)
# bf2017['F1_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(-1)
# 
# # Create 4 lags
# bf2017['L1_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(1)
# bf2017['L2_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(2)
# bf2017['L3_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(3)
# bf2017['L4_D_PBD_incr'] = bf2017['D_PBD_incr'].shift(4)
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
import matplotlib.pyplot as plt


def test4(bf2017):
    bf2017['yearmonth'] = pd.to_datetime(bf2017['yearmonth'])
    # print(bf2017.isnull().sum())
    # Set arbitrary value for the 4th lag of the treatment adoption indicator for 2006-05
    # subset_bf2017[GJSI] = bf2017[GJSI]
    # test4 was not possible to get working. I dont know what to do
    bf2017.loc[bf2017['yearmonth'] == '2006-05', 'L4_D_PBD_incr'] = 0
    print(bf2017.columns)
    # Define crisis sample: 2006-05 to 2011-12
    bf2017['crisis'] = (pd.to_datetime(bf2017['yearmonth']).dt.to_period('M') >= pd.Period('2006-05')) & \
    (pd.to_datetime(bf2017['yearmonth']).dt.to_period('M') <= pd.Period('2011-12'))

    # Generate binned endpoints according to eq. (5)
    bf2017['L4bin_D_PBD_incr'] = bf2017.groupby(['state', 'crisis'])['L4_D_PBD_incr'].cumsum()

    # Generate binned endpoint at lead 3 according to eq. (5)
    bf2017['F3bin_D_PBD_incr'] = bf2017.groupby(['state', 'crisis'])['F3_D_PBD_incr'].cumsum()[::-1]

    # Estimate event-study model in levels with fixed effects
    exog_vars = ['F3bin_D_PBD_incr', 'F2_D_PBD_incr', 'D_PBD_incr', 'L1_D_PBD_incr',
                 'L2_D_PBD_incr', 'L3_D_PBD_incr', 'L4bin_D_PBD_incr', 'frac_total_ui', 'yearmonth']
    y_var = bf2017['log_GJSI']

    subset_bf2017 = bf2017.loc[bf2017['year'] >= 2012, ['log_GJSI', 'F2_D_PBD_decr', 'F1_D_PBD_decr', 'D_PBD_decr', 'L1_D_PBD_decr', 'L2_D_PBD_decr', 'L3_D_PBD_decr', 'L4_D_PBD_decr', 'frac_total_ui']]
    x = subset_bf2017.drop(['log_GJSI'], axis=1)
    y = subset_bf2017['log_GJSI']
    model = sm.OLS(y_var, exog_vars)
    result = model.fit()

    # model_data = bf2017[bf2017['year'] <= 2011]
    estim_incr_es_fe = PanelOLS.from_formula(model, data=subset_bf2017)
    summary_estim_incr_es_fe = estim_incr_es_fe.fit()
    print(summary_estim_incr_es_fe)

    # Beta coefficients and standard errors
    results_incr_es_fe = summary_estim_incr_es_fe.coefs
    beta_incr_es_fe = pd.DataFrame({
      'month_to_reform': [-3, -2, -1, 0, 1, 2, 3, 4],
      'coef': [results_incr_es_fe[1:3]['coefficient'].values, 0, results_incr_es_fe[3:]['coefficient'].values],
      'se': [results_incr_es_fe[1:3]['std err'].values, 0, results_incr_es_fe[3:]['std err'].values]
    })

    # Plot beta coefficients
    plt.figure()
    plt.errorbar(x=beta_incr_es_fe['month_to_reform'], y=beta_incr_es_fe['coef'],
                 yerr=1.96*beta_incr_es_fe['se'], color='darkblue', fmt='o')
    plt.axhline(y=0, color='black')
    plt.axvline(x=-0.5, linestyle='dashed')
    plt.xticks(np.arange(-3, 5))
    plt.xlabel("Months relative to reform \n Observations: {}, states: {}, periods: {} ({} - {})."
               .format(estim_incr_es_fe.nobs,
                       estim_incr_es_fe.model.dataframe['state'].nunique(),
                       estim_incr_es_fe.model.dataframe['yearmonth'].nunique(),
                       estim_incr_es_fe.model.dataframe['yearmonth'].min(),
                       estim_incr_es_fe.model.dataframe['yearmonth'].max()))
    plt.ylabel("Effect on log search intensity")
    plt.ylim(-0.06, 0.025)
    plt.yticks(np.arange(-0.06, 0.021, 0.02))
    plt.grid(True)
    plt.show()

test4(bf2017)

# ------------------------------------------------------------------------------
# Figure B.2, left
# multiple treatments of varying intensities (Schmidheiny/Siegloch case 4)
# effect window from -3 to +4
# estimation with distributed-lags in levels
# full sample
# ------------------------------------------------------------------------------


# Estimate dynamic panel model with fixed effects
exog_vars = ['lead(PBD, 2)', 'lead(PBD, 1)', 'PBD', 'lag(PBD, 1)', 'lag(PBD, 2)', 'lag(PBD, 3)', 'lag(PBD, 4)', 'frac_total_ui', 'yearmonth']
model_formula = 'np.log(GJSI) ~ ' + ' + '.join(exog_vars)
model_data = bf2017.dropna(subset=exog_vars + ['GJSI'])
estim_dl_fe_full = PanelOLS.from_formula(model_formula, data=model_data, effect='individual',
                                         entity_effects=True)
summary_estim_dl_fe_full = estim_dl_fe_full.fit()
print(summary_estim_dl_fe_full)

# Gamma coefficients
gamma_dl_fe_full = summary_estim_dl_fe_full.params[:7]

# Store variance-covariance matrix
vcov_dl_fe_full = summary_estim_dl_fe_full.cov[:7, :7]

# Beta coefficients and standard errors
beta_dl_fe_full = pd.DataFrame({
  'month_to_reform': list(range(-3, 5)),
  'coef': [-np.cumsum(gamma_dl_fe_full[:2])[::-1], 0, np.cumsum(gamma_dl_fe_full[2:])],
  'se': [np.sqrt(np.cumsum(np.diag(vcov_dl_fe_full)[:2]))[::-1], 0,
         np.sqrt(np.cumsum(np.diag(vcov_dl_fe_full)[2:]))]
})

# Plot beta coefficients
plt.figure()
plt.plot(beta_dl_fe_full['month_to_reform'], beta_dl_fe_full['coef'], color='darkblue')
plt.scatter(beta_dl_fe_full['month_to_reform'], beta_dl_fe_full['coef'], color='darkblue')
plt.errorbar(x=beta_dl_fe_full['month_to_reform'], y=beta_dl_fe_full['coef'],
             yerr=1.96*beta_dl_fe_full['se'], color='darkblue', fmt='o')
plt.axhline(y=0, color='black')
plt.axvline(x=-0.5, linestyle='dashed')
plt.xticks(np.arange(-3, 5))
plt.xlabel("Months relative to reform \n Observations: {}, states: {}, periods: {} ({} - {})."
           .format(estim_dl_fe_full.nobs,
                   estim_dl_fe_full.model.dataframe['state'].nunique(),
                   estim_dl_fe_full.model.dataframe['yearmonth'].nunique(),
                   estim_dl_fe_full.model.dataframe['yearmonth'].min(),
                   estim_dl_fe_full.model.dataframe['yearmonth'].max()))
plt.ylabel("Effect on log search intensity")
plt.ylim(-0.006, 0.002)
plt.yticks(np.arange(-0.006, 0.003, 0.002))
plt.grid(True)
plt.show()

# Save plot to PDF
plt.figure(figsize=(8, 4))
plt.plot(beta_dl_fe_full['month_to_reform'], beta_dl_fe_full['coef'], color='darkblue')
plt.scatter(beta_dl_fe_full['month_to_reform'], beta_dl_fe_full['coef'], color='darkblue')
plt.errorbar(x=beta_dl_fe_full['month_to_reform'], y=beta_dl_fe_full['coef'],
             yerr=1.96*beta_dl_fe_full['se'], color='darkblue', fmt='o')
plt.axhline(y=0, color='black')
plt.axvline(x=-0.5, linestyle='dashed')
plt.xticks(np.arange(-3, 5))
plt.xlabel("Months relative to reform \n Observations: {}, states: {}, periods: {} ({} - {})."
           .format(estim_dl_fe_full.nobs,
                   estim_dl_fe_full.model.dataframe['state'].nunique(),
                   estim_dl_fe_full.model.dataframe['yearmonth'].nunique(),
                   estim_dl_fe_full.model.dataframe['yearmonth'].min(),
                   estim_dl_fe_full.model.dataframe['yearmonth'].max()))
plt.ylabel("Effect on log search intensity")
plt.ylim(-0.006, 0.002)
plt.yticks(np.arange(-0.006, 0.003, 0.002))
plt.grid(True)
plt.savefig("./Fig_B2_left.pdf")
plt.close()

# ------------------------------------------------------------------------------
# Figure B.2, right
# also reported in Figure B.4, left (FE)
# multiple treatments of varying intensities (Schmidheiny/Siegloch case 4)
# effect window from -3 to +4
# estimation with distributed-lags in levels
# crisis sample
# ------------------------------------------------------------------------------

# Describe treatment status
print(bf2017['PBD'].describe())

# Estimate distributed-lag model in levels with fixed effects with 2 leads and 4 lags
estim_dl_fe_crisis = PanelOLS.from_formula('np.log(GJSI) ~ lead(PBD, 2) + lead(PBD, 1) + PBD + lag(PBD, 1) + lag(PBD, 2) + lag(PBD, 3) + lag(PBD, 4) + frac_total_ui + C(yearmonth)',
                                           data=bf2017[bf2017['year'] <= 2011].dropna(subset=['GJSI', 'PBD', 'frac_total_ui', 'yearmonth']),
                                           effect='individual', entity_effects=True)
summary_estim_dl_fe_crisis = estim_dl_fe_crisis.fit()
print(summary_estim_dl_fe_crisis)

# Gamma coefficients
gamma_dl_fe_crisis = summary_estim_dl_fe_crisis.params[:7]

# Store variance-covariance matrix
vcov_dl_fe_crisis = summary_estim_dl_fe_crisis.cov[:7, :7]

# Beta coefficients and standard errors
beta_dl_fe_crisis = pd.DataFrame({
  'month_to_reform': list(range(-3, 5)),
  'coef': [-np.cumsum(gamma_dl_fe_crisis[:2])[::-1], 0, np.cumsum(gamma_dl_fe_crisis[2:])],
  'se': [np.sqrt(np.cumsum(np.diag(vcov_dl_fe_crisis)[:2]))[::-1], 0,
         np.sqrt(np.cumsum(np.diag(vcov_dl_fe_crisis)[2:]))]
})

# Plot beta coefficients
plt.figure()
plt.plot(beta_dl_fe_crisis['month_to_reform'], beta_dl_fe_crisis['coef'], color='darkblue')
plt.scatter(beta_dl_fe_crisis['month_to_reform'], beta_dl_fe_crisis['coef'], color='darkblue')
plt.errorbar(x=beta_dl_fe_crisis['month_to_reform'], y=beta_dl_fe_crisis['coef'],
             yerr=1.96*beta_dl_fe_crisis['se'], color='darkblue', fmt='o')
plt.axhline(y=0, color='black')
plt.axvline(x=-0.5, linestyle='dashed')
plt.xticks(np.arange(-3, 5))
plt.xlabel("Months relative to reform \n Observations: {}, states: {}, periods: {} ({} - {})."
           .format(estim_dl_fe_crisis.nobs,
                   estim_dl_fe_crisis.model.dataframe['state'].nunique(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].nunique(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].min(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].max()))
plt.ylabel("Effect on log search intensity")
plt.ylim(-0.006, 0.002)
plt.yticks(np.arange(-0.006, 0.003, 0.002))
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# Figure B.4, left
# multiple treatments of varying intensities (Schmidheiny/Siegloch case 4)
# effect window from -3 to +4
# estimation with distributed-lags in first differences and in levels
# crisis sample
# ------------------------------------------------------------------------------



# Describe treatment status in levels
print(bf2017['PBD'].describe())

# Describe treatment status in first differences
print(np.diff(bf2017['PBD']).describe())

# Estimate distributed-lag model in levels (stored in beta_dl_fe_crisis and estim_dl_fe_crisis)

# Estimate distributed-lag model in first differences with 2 leads and 4 lags
# No constant in first differences, i.e. no linear trend in levels
estim_dl_fd_crisis = PanelOLS.from_formula('np.diff(np.log(GJSI)) ~ lead(np.diff(PBD), 2) + np.diff(PBD) + lag(np.diff(PBD), 1) + lag(np.diff(PBD), 2) + lag(np.diff(PBD), 3) + lag(np.diff(PBD), 4) + np.diff(frac_total_ui) + C(yearmonth) - 1',
                                           data=bf2017[bf2017['year'] <= 2011].diff().dropna(subset=['GJSI', 'PBD', 'frac_total_ui', 'yearmonth']),
                                           entity_effects=False, time_effects=False)
summary_estim_dl_fd_crisis = estim_dl_fd_crisis.fit()
print(summary_estim_dl_fd_crisis)

# Gamma coefficients
gamma_dl_fd_crisis = summary_estim_dl_fd_crisis.params[:7]

# Store variance-covariance matrix
vcov_dl_fd_crisis = summary_estim_dl_fd_crisis.cov[:8, :8]

# Beta coefficients and standard errors
beta_dl_fd_crisis = pd.DataFrame({
  'month_to_reform': list(range(-3, 5)),
  'coef': [-np.cumsum(gamma_dl_fd_crisis[:2])[::-1], 0, np.cumsum(gamma_dl_fd_crisis[2:])],
  'se': [np.sqrt(np.cumsum(np.diag(vcov_dl_fd_crisis)[:2]))[::-1], 0,
         np.sqrt(np.cumsum(np.diag(vcov_dl_fd_crisis)[2:]))]
})

# Combine gamma coefficients from fixed effects and first differences
beta_dl_fe_crisis['estimator'] = 'Fixed effects'
beta_dl_fd_crisis['estimator'] = 'First difference'
beta_dl_fefd_crisis = pd.concat([beta_dl_fe_crisis, beta_dl_fd_crisis])

# Plot beta coefficients (combined fixed effects and first differences)
plt.figure()
colors = ['lightblue', 'darkblue']
shapes = ['o', '^']
for estimator, color, shape in zip(['Fixed effects', 'First difference'], colors, shapes):
  subset = beta_dl_fefd_crisis[beta_dl_fefd_crisis['estimator'] == estimator]
plt.plot(subset['month_to_reform'], subset['coef'], color=color)
plt.scatter(subset['month_to_reform'], subset['coef'], color=color, marker=shape)
plt.errorbar(x=subset['month_to_reform'], y=subset['coef'],
             yerr=1.96*subset['se'], color=color, fmt=shape)
plt.axhline(y=0, color='black')
plt.axvline(x=-0.5, linestyle='dashed')
plt.xticks(np.arange(-3, 5))
plt.xlabel("Months relative to reform \n Observations: {}, states: {}, periods: {} ({} - {})."
           .format(estim_dl_fe_crisis.nobs,
                   estim_dl_fe_crisis.model.dataframe['state'].nunique(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].nunique(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].min(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].max()))
plt.ylabel("Effect on log search intensity")
plt.ylim(-0.01, 0.002)
plt.yticks(np.arange(-0.01, 0.003, 0.002))
plt.grid(True)
plt.show()

# Save the plot as a PDF
plt.figure()
colors = ['lightblue', 'darkblue']
shapes = ['o', '^']
for estimator, color, shape in zip(['Fixed effects', 'First difference'], colors, shapes):
  subset = beta_dl_fefd_crisis[beta_dl_fefd_crisis['estimator'] == estimator]
plt.plot(subset['month_to_reform'], subset['coef'], color=color)
plt.scatter(subset['month_to_reform'], subset['coef'], color=color, marker=shape)
plt.errorbar(x=subset['month_to_reform'], y=subset['coef'],
             yerr=1.96*subset['se'], color=color, fmt=shape)
plt.axhline(y=0, color='black')
plt.axvline(x=-0.5, linestyle='dashed')
plt.xticks(np.arange(-3, 5))
plt.xlabel("Months relative to reform \n Observations: {}, states: {}, periods: {} ({} - {})."
           .format(estim_dl_fe_crisis.nobs,
                   estim_dl_fe_crisis.model.dataframe['state'].nunique(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].nunique(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].min(),
                   estim_dl_fe_crisis.model.dataframe['yearmonth'].max()))
plt.ylabel("Effect on log search intensity")
plt.ylim(-0.01, 0.002)
plt.yticks(np.arange(-0.01, 0.003, 0.002))
plt.grid(True)
plt.savefig("./Fig_B4_left.pdf", format='pdf')
plt.close()




