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
                    select c.gvkey,  f.datadate, c.sic, substr(c.sic,1,2) as sic2,

                    /*Firm variables*/
                    f.dp, f.ib, f.capx, f.sale, f.mib, 

                    /*assets*/
                    f.at, f.act, f.intan, f.ao, 

                    /*liabilities*/
                    f.lt, f.lct, f.pstk, f.lo, 

                    /*equity and other*/
                    f.ceq, f.seq, f.txditc, f.pstkrv, f.pstkl, f.tstkp, 

                    /*Investments*/
                    f.invt, f.ivao, f.ivst,

                    /*others*/
                    f.csho, f.oibdp, f.ppegt, f.ppent, f.xsga, f.ap, f.capxv, 

                    /*dividends*/
                    f.dvpa,

                    /*debt*/
                    f.dltt, f.dlc, f.drlt, 

                    /*cash*/
                    f.che,

                    /*Revenue and cost*/
                    f.revt, f.rect, f.cogs, f.xrd, f.xpp, f.drc, f.xacc, 

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




############### Investment ###############
# Aci
data_rawa['ce'] = data_rawa['capx'] / data_rawa['sale']
data_rawa['ce1'] = data_rawa['ce'].shift(1)
data_rawa['ce2'] = data_rawa['ce'].shift(2)
data_rawa['ce3'] = data_rawa['ce'].shift(3)
data_rawa['aci'] = data_rawa['ce']/ (data_rawa['ce1']+data_rawa['ce2']+data_rawa['ce3'])-1

# Cei
data_rawa['lg_me'] = np.log(data_rawa['me']/data_rawa['me'].shift(6))
data_rawa['lg_ret'] =  np.log(data_rawa['ret']*data_rawa['ret'].shift(1)*data_rawa['ret'].shift(2)*data_rawa['ret'].shift(3)*data_rawa['ret'].shift(5)*data_rawa['ret'].shift(6)) 
data_rawa['cei'] = data_rawa['lg_me'] - data_rawa['lg_ret']

# Dac



# dCoa
data_rawa['coa'] = data_rawa['act'] - data_rawa['che']
data_rawa['dcoa'] = (data_rawa['coa']-data_rawa['coa'].shift(1)) / data_rawa['at'].shift(1)


# dBe
data_rawa['dBe'] = (data_rawa['ceq'] - data_rawa['ceq'].shift(1)) / data_rawa['at'].shift(1)


# dFnl & dFin
data_rawa['fna'] = data_rawa['ivst'] + data_rawa['ivao']
data_rawa['fnl'] = data_rawa['dltt'] + data_rawa['dlc'] + data_rawa['pstk']

data_rawa['d_dlc'] = data_rawa['dlc'] - data_rawa['dlc'].shift(1)
data_rawa['d_dlc'] = np.where(data_rawa['d_dlc'].isnull(), 0, data_rawa['d_dlc'])
data_rawa['d_pstk'] = data_rawa['pstk'] - data_rawa['pstk'].shift(1)
data_rawa['d_pstk'] = np.where(data_rawa['d_pstk'].isnull(), 0, data_rawa['d_pstk'])

data_rawa['dfnl'] = (data_rawa['dltt']-data_rawa['dltt'].shift(1)) + data_rawa['d_dlc'] + data_rawa['d_pstk']

data_rawa['d_ivst'] = data_rawa['ivst'] - data_rawa['ivst'].shift(1)
data_rawa['d_ivst'] = np.where(data_rawa['d_ivst'].isnull(), 0, data_rawa['d_ivst'])
data_rawa['d_ivao'] = data_rawa['ivao'] - data_rawa['ivao'].shift(1)
data_rawa['d_ivao'] = np.where(data_rawa['d_ivao'].isnull(), 0, data_rawa['d_ivao'])

data_rawa['dfna'] = data_rawa['d_ivst'] + data_rawa['d_ivao']
data_rawa['dfin'] = data_rawa['dfna'] - data_rawa['dfnl']

data_rawa['dfin'] = data_rawa['dfin'] / data_rawa['at'].shift(1)
data_rawa['dfnl'] = data_rawa['dfnl'] / data_rawa['at'].shift(1)




# dIi
data_rawa['e_invt'] = (data_rawa['capxv'] + data_rawa['capxv'].shift(1))/2
data_rawa['dinvt'] = (data_rawa['capxv'] - data_rawa['e_invt']) / data_rawa['e_invt']

data_rawa['ind'] = data_rawa['capxv']
s = data_rawa.groupby(['jdate', 'sic2'])['ind'].sum()
data_rawa = pd.merge(data_rawa, s, on=['jdate', 'sic2'])
# new industry investment will be named as ind_y, cause it's been grouped by ind
data_rawa['e_ind'] = (data_rawa['ind_y'] + data_rawa['ind_y'].shift(1))/2
data_rawa['dind'] = (data_rawa['ind_y']-data_rawa['e_ind']) / data_rawa['e_ind']
data_rawa['dIi'] = data_rawa['dinvt'] - data_rawa['dind']

# dLno
data_rawa['dlno'] = (data_rawa['ppent']-data_rawa['ppent'].shift(1)) + (data_rawa['intan']-data_rawa['intan'].shift(1)) + (data_rawa['ao']-data_rawa['ao'].shift(1)) - (data_rawa['lo']-data_rawa['lo'].shift(1)) + data_rawa['dp'] 
avg_at = []
for i in range(data_rawa.shape[0]):
  avg_at.append(data_rawa.loc[0:i, 'at'].mean())
data_rawa['avg_at'] = pd.DataFrame(avg_at)
data_rawa['dlno'] = data_rawa['dlno'] / data_rawa['avg_at']


# dNco
data_rawa['nca'] = data_rawa['at'] - data_rawa['act'] - data_rawa['ivao']
data_rawa['ncl'] = data_rawa['lt'] - data_rawa['lct'] - data_rawa['dltt'] 
data_rawa['nco'] = data_rawa['nca'] - data_rawa['ncl']
data_rawa['dnoc'] = data_rawa['nco'] - data_rawa['nco'].shift(1)


# dNca
data_rawa['ivao_0'] = np.where(data_rawa['ivao'].isnull(), 0, data_rawa['ivao'])
data_rawa['dltt_0'] = np.where(data_rawa['dltt'].isnull(), 0, data_rawa['dltt'])

data_rawa['nca'] = data_rawa['at'] - data_rawa['act'] - data_rawa['ivao_0']
data_rawa['ncl'] = data_rawa['lt'] - data_rawa['lct'] - data_rawa['dltt_0'] 
data_rawa['nco'] = data_rawa['nca'] - data_rawa['ncl']
data_rawa['dnca'] = data_rawa['nco'] - data_rawa['nco'].shift(1)



# dNoa
data_rawa['dlc_0'] = np.where(data_rawa['dlc'].isnull(), 0, data_rawa['dlc'])
data_rawa['mib_0'] = np.where(data_rawa['mib'].isnull(), 0, data_rawa['mib'])
data_rawa['pstk_0'] = np.where(data_rawa['pstk'].isnull(), 0, data_rawa['pstk'])

data_rawa['op_at'] = data_rawa['at'] - data_rawa['che']
data_rawa['op_lia'] = data_rawa['at'] - data_rawa['dlc_0'] - data_rawa['dltt_0'] - data_rawa['mib_0'] - data_rawa['pstk_0'] - data_rawa['ceq']
data_rawa['net_op'] = data_rawa['op_at'] - data_rawa['op_lia']
data_rawa['dnoa'] = (data_rawa['net_op']-data_rawa['net_op'].shift(1))/ data_rawa['at'].shift(1)


# dPia
data_rawa['c_propty'] = data_rawa['ppegt'] - data_rawa['ppegt'].shift(1)
data_rawa['c_invt'] = data_rawa['invt'] - data_rawa['invt'].shift(1)
data_rawa['dpia'] = (data_rawa['c_propty'] + data_rawa['c_invt']) / data_rawa['at'].shift(1)





######### Profitability ##########
# Ato
data_rawa['op_at'] = data_rawa['at'] - data_rawa['che'] - data_rawa['ivao_0']
data_rawa['op_lia'] = data_rawa['dlc_0'] - data_rawa['dltt_0'] - data_rawa['mib_0'] - data_rawa['pstk_0'] - data_rawa['ceq']
data_rawa['noa'] = data_rawa['op_at'] - data_rawa['op_lia']
data_rawa['ato'] = data_rawa['sale'] / data_rawa['noa'].shift(1)


# Cla
data_rawa['d_rect'] = data_rawa['rect'] - data_rawa['rect'].shift(1)
data_rawa['d_invt'] = data_rawa['invt'] - data_rawa['invt'].shift(1)
data_rawa['d_xpp'] = data_rawa['xpp'] - data_rawa['xpp'].shift(1)
data_rawa['d_dr'] = (data_rawa['drc']-data_rawa['drc'].shift(1)) + (data_rawa['drlt']-data_rawa['drlt'].shift(1))
data_rawa['d_ap'] = data_rawa['ap'] - data_rawa['ap'].shift(1)
data_rawa['d_xacc'] = data_rawa['xacc'] - data_rawa['xacc'].shift(1)

data_rawa['xrd_0'] = np.where(data_rawa['xrd'].isnull(), 0, data_rawa['xrd'])
data_rawa['d_rect_0'] = np.where(data_rawa['d_rect'].isnull(), 0, data_rawa['d_rect'])
data_rawa['d_invt_0'] = np.where(data_rawa['d_invt'].isnull(), 0, data_rawa['d_invt'])
data_rawa['d_xpp_0'] = np.where(data_rawa['d_xpp'].isnull(), 0, data_rawa['d_xpp'])
data_rawa['d_dr_0'] = np.where(data_rawa['d_dr'].isnull(), 0, data_rawa['d_dr'])
data_rawa['d_ap_0'] = np.where(data_rawa['d_ap'].isnull(), 0, data_rawa['d_ap'])
data_rawa['d_xacc_0'] = np.where(data_rawa['d_xacc'].isnull(), 0, data_rawa['d_xacc'])

data_rawa['cla'] = data_rawa['revt'] - data_rawa['cogs'] - data_rawa['xsga'] + data_rawa['xrd_0']\
                 - data_rawa['d_rect_0'] - data_rawa['d_invt_0'] - data_rawa['d_xpp_0']\
                 + data_rawa['d_dr_0'] + data_rawa['d_ap_0'] + data_rawa['d_xacc_0']
data_rawa['cla'] = data_rawa['cla'] / data_rawa['at'].shift(1)


# Cop
data_rawa['cop'] = data_rawa['revt'] - data_rawa['cogs'] - data_rawa['xsga'] + data_rawa['xrd_0']\
                 - data_rawa['d_rect_0'] - data_rawa['d_invt_0'] - data_rawa['d_xpp_0']\
                 + data_rawa['d_dr_0'] + data_rawa['d_ap_0'] + data_rawa['d_xacc_0']
data_rawa['cop'] = data_rawa['cop'] / data_rawa['at'] 


# Cto
data_rawa['cto'] = data_rawa['sale'] / data_rawa['at'].shift(1)








# Annual Accounting Variables
#chars_a = data_rawa[['permno','gvkey', 'jdate', 'datadate', 'count', 'exchcd', 'shrcd', 'sic', 'be', 'bm',  
#                     'bmj', 'csho', 'prc',  
#                     'cp','ib', 'dp', 
#                     'ebp', 'dvpa', 'tstkp', 'dltt', 'dlc', 'pstk', 'che', 'ceq',
#                     'em', 'pstkrv', 'oibdp',
#                     'aci', 'capx', 'sale',
#                    'ia', 'at',
#                     'dpia', 'ppegt', 'invt']]
chars_a = data_rawa[['permno','gvkey', 'jdate', 'datadate', 'count', 'exchcd', 'shrcd', 'sic', 'be', 
                     'aci', 'capx', 'sale',
                     'at', 'dpia', 'ppegt', 'invt',
                     'cei', 'me', 
                     'dBe', 'ceq', 'dfnl', 'dfin', 'ivst', 'ivao', 
                     'dcoa', 'act', 'che', 
                     'dlno', 'ppent', 'intan', 'ao', 'lo', 'dp',
                     'dnoc', 'lt', 'lct', 
                     'dnoa', 'dlc', 'dltt', 'mib', 'pstk',
                     'ato', 'cla', 'revt', 'cogs', 'xsga', 'xrd', 'rect', 'xpp', 'drc', 'drlt', 'ap', 'xacc',
                     'cop', 'cto',
                     'dIi', 'sic2', 'capxv']]
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















with open('chars_a0724.pkl', 'wb') as f:
    pkl.dump(chars_a, f)
print(chars_a)


