# Fama & French 3 factors residual variance
# Note: Please use the latest version of pandas, this version should support returning to pd.Series after rolling
# To get a faster speed, we split the big dataframe into small ones
# Then using different process to calculate the variance
# We use 20 process to calculate variance, you can change the number of process according to your CPU situation
# You can use the following code to check your CPU situation
# import multiprocessing
# multiprocessing.cpu_count()

import pandas as pd
import numpy as np
import datetime as dt
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import datetime
import pickle as pkl
import multiprocessing as mp

###################
# Connect to WRDS #
###################
conn = wrds.Connection()

# CRSP Block
crsp = conn.raw_sql("""
                      select a.permno, a.date, a.ret, a.prc, a.vol
                      from crsp.dsf as a
                      where a.date > '01/01/1959'
                      """)

# sort variables by permno and date
crsp = crsp.sort_values(by=['permno', 'date'])

# change variable format to int
crsp['permno'] = crsp['permno'].astype(int)

# Line up date to be end of month
crsp['date'] = pd.to_datetime(crsp['date'])

# find the closest trading day to the end of the month
crsp['monthend'] = crsp['date'] + MonthEnd(0)
crsp['date_diff'] = crsp['monthend'] - crsp['date']
date_temp = crsp.groupby(['permno', 'monthend'])['date_diff'].min()
date_temp = pd.DataFrame(date_temp)  # convert Series to DataFrame
date_temp.reset_index(inplace=True)
date_temp.rename(columns={'date_diff': 'min_diff'}, inplace=True)
crsp = pd.merge(crsp, date_temp, how='left', on=['permno', 'monthend'])
crsp['sig'] = np.where(crsp['date_diff'] == crsp['min_diff'], 1, np.nan)

# label every date of month end
crsp['month_count'] = crsp[crsp['sig'] == 1].groupby(['permno']).cumcount()

# mark the number of each month to each day of this month
crsp['month_count'] = crsp.groupby(['permno'])['month_count'].fillna(method='bfill')


######################
#    Calculate dtv   #
######################

def main(start, end, step):
    crsp['dtv'] = crsp['prc'] * crsp['vol']
    return crsp


# calculate variance of residual through rolling window
# Note: please split dataframe according to your CPU situation. For example, we split dataframe to (1-0)/0.05 = 20 sub
# dataframes here, so the function will use 20 cores to calculate variance of residual.
if __name__ == '__main__':
    crsp = main(0, 1, 0.05)

with open('rvar_ff3.pkl', 'wb') as f:
    pkl.dump(crsp, f)