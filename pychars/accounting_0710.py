import pandas as pd
import numpy as np
import datetime as dt
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import pickle as pkl

###################
# Connect to WRDS #
###################
conn = wrds.Connection()





#######################################################################################################################
#                                                  Compustat Block                                                    #
#######################################################################################################################
comp = conn.raw_sql("""
                    /*header info*/
                    select c.gvkey,  f.datadate, c.sic, 

                    /*Firm variables*/
                    f.dp, f.ib,

                    /*assets*/
                    f.at,

                    /*liabilities*/
                    f.lt, f.pstk,

                    /*equity and other*/
                    f.ceq, f.seq, f.txditc, f.pstkrv, f.pstkl, f.tstkp, 

                    /*others*/
                    f.csho, f.oibdp,

                    /*dividends*/
                    f.dvpa,

                    /*debt*/
                    f.dltt, f.dlc,

                    /*cash*/
                    f.che,

                    /*market*/
                    abs(f.prcc_f) as prcc_f

                    from comp.funda as f
                    left join comp.company as c
                    on f.gvkey = c.gvkey
                    
                    /*get consolidated, standardized, industrial format statements*/
                    where f.indfmt = 'INDL' 
                    and f.datafmt = 'STD'
                    and f.popsrc = 'D'
                    and f.consol = 'C'
                    and f.datadate >= '01/01/2000'
                    """)

# convert datadate to date fmt
comp['datadate'] = pd.to_datetime(comp['datadate'])

# sort and clean up
comp = comp.sort_values(by=['gvkey', 'datadate']).drop_duplicates()

# # prep for clean-up and using time series of variables
comp['count'] = comp.groupby(['gvkey']).cumcount()  # number of years in Compustat

# calculate Compustat market equity
comp['mve_f'] = comp['csho'] * comp['prcc_f']





#######################################################################################################################
#                                                       CRSP Block                                                    #
#######################################################################################################################
# Create a CRSP Subsample with Monthly Stock and Event Variables
# Restrictions will be applied later
# Select variables from the CRSP monthly stock and event datasets
crsp_m = conn.raw_sql("""
                      select a.prc, a.ret, a.retx, a.shrout, a.date, a.permno, a.permco,
                      b.shrcd, b.exchcd
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date >= '01/01/2000'
                      and b.exchcd between 1 and 3
                      """)

# change variable format to int
crsp_m[['permco', 'permno', 'shrcd', 'exchcd']] = crsp_m[['permco', 'permno', 'shrcd', 'exchcd']].astype(int)

# Line up date to be end of month
crsp_m['date'] = pd.to_datetime(crsp_m['date'])
crsp_m['monthend'] = crsp_m['date'] + MonthEnd(0)  # set all the date to the standard end date of month

# calculate market equity
crsp_m['me'] = crsp_m['prc'].abs() * crsp_m['shrout']  

# if Market Equity is Nan then let return equals to 0
crsp_m['ret'] = np.where(crsp_m['me'].isnull(), 0, crsp_m['ret'])
crsp_m['retx'] = np.where(crsp_m['me'].isnull(), 0, crsp_m['retx'])

# impute me
crsp_m = crsp_m.sort_values(by=['permno', 'date']).drop_duplicates()
crsp_m['me'] = np.where(crsp_m['permno'] == crsp_m['permno'].shift(1), crsp_m['me'].fillna(method='ffill'), crsp_m['me'])

# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp_m.groupby(['monthend', 'permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_maxme = crsp_m.groupby(['monthend', 'permco'])['me'].max().reset_index()
# join by monthend/maxme to find the permno
crsp1 = pd.merge(crsp_m, crsp_maxme, how='inner', on=['monthend', 'permco', 'me'])
# drop me column and replace with the sum me
crsp1 = crsp1.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['monthend', 'permco'])
# sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values(by=['permno', 'monthend']).drop_duplicates()





#######################################################################################################################
#                                                        CCM Block                                                    #
#######################################################################################################################
# merge CRSP and Compustat
# reference: https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/
ccm = conn.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """)

ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])

# if linkenddt is missing then set to today date
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

# merge ccm and comp
ccm1 = pd.merge(comp, ccm, how='left', on=['gvkey'])

# we can only get the accounting data after the firm public their report
# for annual data, we ues 6 months lagged data
ccm1['yearend'] = ccm1['datadate'] + YearEnd(0)
ccm1['jdate'] = ccm1['yearend'] + MonthEnd(6)

# set link date bounds
ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]

# link comp and crsp
crsp2 = crsp2.rename(columns={'monthend': 'jdate'})
data_rawa = pd.merge(crsp2, ccm2, how='inner', on=['permno', 'jdate'])

# filter exchcd & shrcd
data_rawa = data_rawa[((data_rawa['exchcd'] == 1) | (data_rawa['exchcd'] == 2) | (data_rawa['exchcd'] == 3)) &
                   ((data_rawa['shrcd'] == 10) | (data_rawa['shrcd'] == 11))]



# process Market Equity
'''
Note: me is CRSP market equity, mve_f is Compustat market equity. Please choose the me below.
'''
data_rawa['me'] = data_rawa['me']/1000  # CRSP ME
data_rawa['me_comp'] = data_rawa['mve_f']  # Compustat ME

# count single stock years
data_rawa['count'] = data_rawa.groupby(['gvkey']).cumcount()



# deal with the duplicates
data_rawa.loc[data_rawa.groupby(['datadate', 'permno', 'linkprim'], as_index=False).nth([0]).index, 'temp'] = 1
data_rawa = data_rawa[data_rawa['temp'].notna()]
data_rawa.loc[data_rawa.groupby(['permno', 'yearend', 'datadate'], as_index=False).nth([-1]).index, 'temp'] = 1
data_rawa = data_rawa[data_rawa['temp'].notna()]




# stockholders' equity 
data_rawa['se'] = np.where(data_rawa['seq'].isnull(), data_rawa['ceq']+data_rawa['pstk'], data_rawa['seq'])
data_rawa['se'] = np.where(data_rawa['se'].isnull(), data_rawa['at']-data_rawa['lt'], data_rawa['se'])

data_rawa['txditc'] = data_rawa['txditc'].fillna(0)

# preferrerd stock
data_rawa['ps'] = np.where(data_rawa['pstkrv'].isnull(), data_rawa['pstkl'], data_rawa['pstkrv'])
data_rawa['ps'] = np.where(data_rawa['ps'].isnull(), data_rawa['pstk'], data_rawa['ps'])
data_rawa['ps'] = np.where(data_rawa['ps'].isnull(), 0, data_rawa['ps'])


# book equity
data_rawa['be'] = data_rawa['se'] + data_rawa['txditc'] - data_rawa['ps']
data_rawa['be'] = np.where(data_rawa['be'] > 0, data_rawa['be'], np.nan)


# bm
data_rawa['bm'] = data_rawa['be'] / data_rawa['me']
data_rawa['bm_n'] = data_rawa['be']

# Bmj
data_rawa['be_per'] = data_rawa['be'] / data_rawa['csho']
data_rawa['bmj'] = data_rawa['be_per'] / data_rawa['prc'] 
############### *Q*: used prc as  share price  from crsp ##########

# Cp
data_rawa['cf'] = data_rawa['ib'] + data_rawa['dp']
data_rawa['cp'] = data_rawa['cf'] / data_rawa['me']

# Dp
###### *Q* difference return with without divident

# Dur
# me = data_rawa['me_comp']


# Ebp
data_rawa['dvpa'] = np.where(data_rawa['dvpa'].isnull(), 0, data_rawa['dvpa'])
data_rawa['tstkp'] = np.where(data_rawa['tstkp'].isnull(), 0, data_rawa['tstkp'])
data_rawa['f_liab'] = data_rawa['dltt'] + data_rawa['dlc'] + data_rawa['pstk'] + data_rawa['dvpa'] - data_rawa['tstkp']
data_rawa['f_asse'] = data_rawa['che']
# net debt : = ﬁnancial liabilities - ﬁnancial assets.
data_rawa['n_debt'] = data_rawa['f_liab'] - data_rawa['f_asse']
data_rawa['be'] = data_rawa['ceq'] + data_rawa['tstkp'] - data_rawa['dvpa']
data_rawa['ebp'] = (data_rawa['n_debt']+data_rawa['be']) / (data_rawa['n_debt']+data_rawa['me'])


# Em
data_rawa['enteprs_v'] = data_rawa['me'] + data_rawa['dlc'] + data_rawa['dltt'] + data_rawa['pstkrv'] - data_rawa['che']
data_rawa['em'] = data_rawa['enteprs_v'] / data_rawa['oibdp']

# Annual Accounting Variables
chars_a = data_rawa[['permno','gvkey', 'jdate', 'datadate', 'count', 'exchcd', 'shrcd', 'sic', 'be', 'bm', 'bmj','cp', 'ebp', 'em', 'ib', 'dp', 'dvpa', 'tstkp', 'dltt', 'dlc', 'pstk', 'che', 'ceq', 'pstkrv', 'oibdp']]
chars_a.reset_index(drop=True, inplace=True)













#######################################################################################################################
#                                                       Momentum                                                      #
#######################################################################################################################
crsp_mom = conn.raw_sql("""
                        select permno, date, ret, retx, prc, shrout
                        from crsp.msf
                        where date >= '01/01/2000'
                        """)


crsp_mom['permno'] = crsp_mom['permno'].astype(int)
crsp_mom['date'] = pd.to_datetime(crsp_mom['date'])
crsp_mom = crsp_mom.dropna()
# populate the chars to monthly
crsp_mom['jdate'] = crsp_mom['date'] + MonthEnd(0)



# add delisting return
dlret = conn.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     """)

dlret.permno = dlret.permno.astype(int)
dlret['dlstdt'] = pd.to_datetime(dlret['dlstdt'])
dlret['jdate'] = dlret['dlstdt'] + MonthEnd(0)

# merge delisting return to crsp return
crsp_mom = pd.merge(crsp_mom, dlret, how='left', on=['permno', 'jdate'])
crsp_mom['dlret'] = crsp_mom['dlret'].fillna(0)
crsp_mom['ret'] = crsp_mom['ret'].fillna(0)
crsp_mom['retadj'] = (1 + crsp_mom['ret']) * (1 + crsp_mom['dlret']) - 1
crsp_mom['me'] = crsp_mom['prc'].abs() * crsp_mom['shrout']  # calculate market equity
crsp_mom['retx'] = np.where(crsp_mom['me'].isnull(), 0, crsp_mom['retx'])
crsp_mom = crsp_mom.drop(['dlret', 'dlstdt', 'prc', 'shrout'], axis=1)



# chars_a
chars_a = pd.merge(crsp_mom, chars_a, how='left', on=['permno', 'jdate'])
chars_a['datadate'] = chars_a.groupby(['permno'])['datadate'].fillna(method='ffill')
chars_a = chars_a.groupby(['permno', 'datadate'], as_index=False).fillna(method='ffill')
chars_a = chars_a[((chars_a['exchcd'] == 1) | (chars_a['exchcd'] == 2) | (chars_a['exchcd'] == 3)) &
                      ((chars_a['shrcd'] == 10) | (chars_a['shrcd'] == 11))]



with open('chars_a0710.pkl', 'wb') as f:
    pkl.dump(chars_a, f)

print(chars_a)
