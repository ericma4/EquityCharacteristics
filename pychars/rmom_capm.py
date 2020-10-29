# CAPM residual momentum
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

# CRSP Block. We use crsp.msf and ff.factors_monthly
cr = conn.raw_sql("""
                      select a.permno, a.date, a.ret from crsp.msf as a
                      where a.date >= '01/01/1959'
                      """)

ff = conn.raw_sql("""
                      select b.rf, b.mktrf, b.date from ff.factors_monthly as b
                      where b.date >= '01/01/1959'
                      """)

ff['date'] = pd.to_datetime(ff['date']) + MonthEnd(0)
cr['date'] = pd.to_datetime(cr['date']) + MonthEnd(0)

crsp = pd.merge(cr,ff,how = 'left', on = ['date'])
crsp['exret'] = crsp['ret'] - crsp['rf']
crsp = crsp.sort_values(by=['permno', 'date'])
crsp['permno'] = crsp['permno'].astype(int)
crsp = crsp[['permno','date','ret','exret','mktrf']]

# create a firm list. Use reset_index to create a count to number companies without duplicates
df_firm = crsp.drop_duplicates(['permno'])
df_firm = df_firm[['permno']]
df_firm['permno'] = df_firm['permno'].astype(int)
df_firm = df_firm.reset_index(drop=True)
df_firm = df_firm.reset_index()
df_firm = df_firm.rename(columns={'index': 'count'})

#Extract number of data points for each permno
crsp['month_count'] = crsp.groupby('permno').cumcount()
month_num = crsp.groupby('permno')['month_count'].tail(1)
month_num = month_num.astype(int)

######################
# Calculate residual #
######################


def get_res_var(df, firm_list):
    #for every permno, we have count as its number of obervations, prog as its number in firmlist
    for firm, count, prog in zip(firm_list['permno'], month_num, range(firm_list['permno'].count()+1)):
        prog = prog + 1
        #this is to demonstrate how many companies have already been calculated, and convert it into percentage
        print('processing permno %s' % firm, '/', 'finished', '%.2f%%' % ((prog/firm_list['permno'].count())*100))
        #Actually there are count+1 months
        for i in range(count + 1):
            # if you want to change the rolling window, please change here: i - 2 means 3 months is a window.
            # Now we have temp as sixty month data
            temp = df[(df['permno'] == firm) & (i -59 <= df['month_count']) & (df['month_count'] <= i)]
            # if observations in last 3 months are less 21, we drop the rvar of this month
            if temp['permno'].count() < 20:
                pass
            else:
                rolling_window = temp['permno'].count()
                index = temp.tail(1).index
              
                X = pd.DataFrame()
                X[['mktrf']] = temp[['mktrf']]
                X['intercept'] = 1
                X = X[['intercept', 'mktrf']]
                X = np.mat(X)
                Y = np.mat(temp[['exret']])
                res = (np.identity(rolling_window) - X.dot(X.T.dot(X).I).dot(X.T)).dot(Y)
                #print(df)
                df.loc[index, 'res'] = res[-1]
    return df


def sub_df(start, end, step):
    # we use dict to store different sub dataframe
    temp = {}
    for i, h in zip(np.arange(start, end, step), range(int((end-start)/step))):
        print('processing splitting dataframe:', round(i, 2), 'to', round(i + step, 2))
        if i == 0:  # to get the left point
            temp['firm' + str(h)] = df_firm[df_firm['count'] <= df_firm['count'].quantile(i + step)]
            temp['crsp' + str(h)] = pd.merge(crsp, temp['firm' + str(h)], how='left',
                                             on='permno').dropna(subset=['count'])
        else:
            temp['firm' + str(h)] = df_firm[(df_firm['count'].quantile(i) < df_firm['count']) & (
                    df_firm['count'] <= df_firm['count'].quantile(i + step))]
            temp['crsp' + str(h)] = pd.merge(crsp, temp['firm' + str(h)], how='left',
                                             on='permno').dropna(subset=['count'])
    return temp

def main(start, end, step):
    df = sub_df(start, end, step)
    pool = mp.Pool()
    p_dict = {}
    for i in range(int((end-start)/step)):
        p_dict['p' + str(i)] = pool.apply_async(get_res_var, (df['crsp%s' % i], df['firm%s' % i],))
    pool.close()
    pool.join()
    result = pd.DataFrame()
    print('processing pd.concat')
    for h in range(int((end-start)/step)):
        result = pd.concat([result, p_dict['p%s' % h].get()])
    return result


# calculate variance of residual through rolling window
# Note: please split dataframe according to your CPU situation. For example, we split dataframe to (1-0)/0.05 = 20 sub
# dataframes here, so the function will use 20 cores to calculate variance of residual.
if __name__ == '__main__':
    crsp = main(0, 1, 0.05)

# process dataframe
crsp = crsp.dropna(subset=['res'])  # drop NA due to rolling
crsp = crsp.rename(columns={'res': 'rmom_capm_1m'})
crsp = crsp.reset_index(drop=True)
crsp = crsp[['permno', 'date', 'rmom_capm_1m']]

def mom(start, end, df):
    lag = pd.DataFrame()
    result = 1
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['permno'])['rmom_capm_1m'].shift(i)
        result = result * (1+lag['mom%s' % i])
    result = result - 1
    return result


crsp['rmom_capm_12m'] = mom(1,12,crsp)
crsp['rmom_capm_60m'] = mom(12,60,crsp)

with open('rmom_capm.pkl', 'wb') as f:
    pkl.dump(crsp, f)





