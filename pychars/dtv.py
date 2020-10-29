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

#######################################################################################################################
#                                                  CRSP Block                                                    #
#######################################################################################################################

# Create a CRSP Subsample with Daily Stock and Event Variables
# Restrictions will be applied later
# Select variables from the CRSP daily stock and event datasets
crsp = conn.raw_sql("""
                      select a.prc, a.ret, a.retx, a.shrout, a.vol, a.cfacpr, a.cfacshr, a.date, a.permno, a.permco,
                      b.ticker, b.ncusip, b.shrcd, b.exchcd
                      from crsp.dsf as a
                      left join crsp.dsenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date >= '01/01/1959'
                      and b.exchcd between 1 and 3
                      """)

# change variable format to int
crsp[['permco', 'permno', 'shrcd', 'exchcd']] = crsp[['permco', 'permno', 'shrcd', 'exchcd']].astype(int)

# Line up date to be end of month
crsp['date'] = pd.to_datetime(crsp['date'])
crsp['monthend'] = crsp['date'] + MonthEnd(0)  # set all the date to the standard end date of month

crsp['me'] = crsp['prc'].abs() * crsp['shrout']  # calculate market equity

# if Market Equity is Nan then let return equals to 0
crsp['ret'] = np.where(crsp['me'].isnull(), 0, crsp['ret'])
crsp['retx'] = np.where(crsp['me'].isnull(), 0, crsp['retx'])

# impute me
crsp = crsp.sort_values(by=['permno', 'date']).drop_duplicates()
crsp['me'] = np.where(crsp['permno'] == crsp['permno'].shift(1), crsp['me'].fillna(method='ffill'), crsp['me'])

# Aggregate Market Cap
'''
There are cases when the same firm (permco) has two or more securities (permno) at same date.
For the purpose of ME for the firm, we aggregated all ME for a given permco, date.
This aggregated ME will be assigned to the permno with the largest ME.
'''
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['monthend', 'permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_maxme = crsp.groupby(['monthend', 'permco'])['me'].max().reset_index()
# join by monthend/maxme to find the permno
crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['monthend', 'permco', 'me'])
# drop me column and replace with the sum me
crsp1 = crsp1.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['monthend', 'permco'])
# sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values(by=['permno', 'monthend']).drop_duplicates()


#######################################################################################################################
#                                                  Calculate                                                    #
#######################################################################################################################


def mom_1(start, end, df):
    """
    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result = 0
    for i in range(start, end):
        lag['mom%s' % i] = df['dtvm'].shift(i)
        result = result + (lag['mom%s' % i])
    result = result/(end-start)
    return result


def mom_2(start, end, df):
    """
    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result = 0
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['permno'])['day_count'].shift(i)
        result = result + (lag['mom%s' % i])
    result = result
    return result


#Calculating daily trading volume
#at least there is only one datapoint for a permno at one day
crsp2['dtv'] = crsp2['prc'] * crsp2['vol']

#Average trading value for one month
#This will be a dataframe with every permno's average dtv for every month
#It has 3 columns, and we reset its index
dtv_m = crsp2.groupby(['permno','monthend'])[['dtv']].mean()
dtv_m.rename(columns = {'dtv':'dtvm'})

#record how many datapoints we have for a typical month
dtv_m['day_count'] = crsp2.groupby(['permno','monthend'])[['dtv']].count()['dtv']
dtv_m.reset_index()

#Merge it back with crsp2 to only store monthly average data
crsp3 = pd.merge(crsp2,dtv_m,how = 'inner', on = ['monthend','permno'])
crsp3.drop(['dtv'],axis = 1)

#Generate firm list
df_firm = crsp3.drop_duplicates(['permno'])
df_firm = df_firm[['permno']]
df_firm['permno'] = df_firm['permno'].astype(int)
df_firm = df_firm.reset_index(drop=True)
df_firm = df_firm.reset_index()
df_firm = df_firm.rename(columns={'index': 'count'})

#Extract number of data points for each permno
crsp3['month_count'] = crsp3.groupby('permno').cumcount()
month_num = crsp3.groupby('permno')['month_count'].tail(1)
month_num = month_num.astype(int)

#dtv
#crsp3['half_year_count'] =  mom_2(crsp3,0,6)
#crsp3['dtv'] = mom_1(crsp3,0,6)
#change the ones with less than 50 records to nan
#crsp3['dtv'] = np.where(crsp3['half_year_count']<50, np.nan, crsp3['dtv'])


def get_dtv(df, firm_list):
    """
    :param df: stock dataframe
    :param firm_list: list of firms matching stock dataframe
    :return: dataframe with variance of residual
    """
    for firm, count, prog in zip(firm_list['permno'], month_num, range(firm_list['permno'].count()+1)):
        prog = prog + 1
        print('processing permno %s' % firm, '/', 'finished', '%.2f%%' % ((prog/firm_list['permno'].count())*100))
        for i in range(count + 1):
            # if you want to change the rolling window, please change here: i - 2 means 3 months is a window.
            temp = df[(df['permno'] == firm) & (i - 5 <= df['month_count']) & (df['month_count'] <= i)]
            # if observations in last 3 months are less than 2 months, we drop the rvar of this month
            if temp['permno'].count() < 2:
                pass
            else:
                index = temp.tail(1).index
                df.loc[index, 'dtv'] = mom_1(0,6,temp)
    return df


def sub_df(start, end, step):
    """
    :param start: the quantile to start cutting, usually it should be 0
    :param end: the quantile to end cutting, usually it should be 1
    :param step: quantile step
    :return: a dictionary including all the 'firm_list' dataframe and 'stock data' dataframe
    """
    # we use dict to store different sub dataframe
    temp = {}
    for i, h in zip(np.arange(start, end, step), range(int((end-start)/step))):
        print('processing splitting dataframe:', round(i, 2), 'to', round(i + step, 2))
        if i == 0:  # to get the left point
            temp['firm' + str(h)] = df_firm[df_firm['count'] <= df_firm['count'].quantile(i + step)]
            temp['cr' + str(h)] = pd.merge(crsp3, temp['firm' + str(h)], how='left',
                                             on='permno').dropna(subset=['count'])
        else:
            temp['firm' + str(h)] = df_firm[(df_firm['count'].quantile(i) < df_firm['count']) & (
                    df_firm['count'] <= df_firm['count'].quantile(i + step))]
            temp['cr' + str(h)] = pd.merge(crsp3, temp['firm' + str(h)], how='left',
                                             on='permno').dropna(subset=['count'])
    return temp

def main(start, end, step):
    """

    :param start: the quantile to start cutting, usually it should be 0
    :param end: the quantile to end cutting, usually it should be 1
    :param step: quantile step
    :return: a dataframe with calculated variance of residual
    """
    df = sub_df(start, end, step)
    pool = mp.Pool()
    p_dict = {}
    for i in range(int((end-start)/step)):
        p_dict['p' + str(i)] = pool.apply_async(get_dtv, (df['cr%s' % i], df['firm%s' % i],))
    pool.close()
    pool.join()
    result = pd.DataFrame()
    print('processing pd.concat')
    for h in range(int((end-start)/step)):
        result = pd.concat([result, p_dict['p%s' % h].get()])
    return result

if __name__ == '__main__':
    crsp3 = main(0, 1, 0.05)

crsp3 = crsp3.dropna(subset=['dtv'])  # drop NA due to rolling
crsp3 = crsp3.reset_index(drop=True)
crsp3 = crsp3[['permno', 'date', 'dtv']]


with open('dtv.pkl', 'wb') as f:
    pkl.dump(crsp3, f)