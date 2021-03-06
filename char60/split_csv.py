import pickle as pkl
import pandas as pd

# with open('/Users/eric/WeDrive/Feng-CityUHK/Data/pychars/chars60/raw/chars_impute_60.pkl', 'rb') as f:
#     chars = pkl.load(f)

with open('/Users/eric/WeDrive/Feng-CityUHK/Data/pychars/chars60/rank/chars_rank_60.pkl', 'rb') as f:
    chars = pkl.load(f)

# chars = chars[['permno', 'gvkey', 'datadate', 'jdate', 'ffi49', 'sic', 'exchcd', 'shrcd', 'ret', 'retx', 'retadj',
#                'rank_abr', 'rank_acc', 'rank_adm', 'rank_agr', 'rank_alm', 'rank_ato', 'rank_baspread', 'rank_beta',
#                'rank_bm', 'rank_bm_ia', 'rank_cash', 'rank_cashdebt', 'rank_cfp',
#                'rank_chcsho', 'rank_chpm', 'rank_chtx', 'rank_cinvest', 'rank_depr', 'rank_dolvol', 'rank_dy',
#                'rank_ep', 'rank_gma', 'rank_grltnoa', 'rank_herf', 'rank_hire',
#                'rank_ill', 'rank_lev', 'rank_lgr', 'rank_maxret', 'rank_me_ia', 'rank_mom12m', 'rank_mom1m',
#                'rank_mom36m', 'rank_mom60m', 'rank_mom6m', 'rank_ni', 'rank_nincr',
#                'rank_noa', 'rank_op', 'rank_pctacc', 'rank_pm', 'rank_pscore', 'rank_rd_sale', 'rank_rdm', 'rank_re',
#                'rank_rna', 'rank_roa', 'rank_roe', 'rank_rsup', 'rank_rvar_capm',
#                'rank_rvar_ff3', 'rank_rvar_mean', 'rank_seas1a', 'rank_sgr', 'rank_sp', 'rank_std_dolvol',
#                'rank_std_turn', 'rank_sue', 'rank_turn', 'rank_zerotrade']]

print(chars.columns.values)

chars['jdate'] = pd.to_datetime(chars['jdate'])
chars['year'] = chars['jdate'].dt.year
chars_1970s = chars[chars['year'] < 1980]
chars_1980s = chars[(chars['year'] >= 1980) & (chars['year'] < 1990)]
chars_1990s = chars[(chars['year'] >= 1990) & (chars['year'] < 2000)]
chars_2000s = chars[(chars['year'] >= 1990) & (chars['year'] < 2010)]
chars_2010s = chars[(chars['year'] >= 2000) & (chars['year'] < 2020)]

chars_1970s.to_csv('chars60_rank_1970s.csv', index=0)
chars_1980s.to_csv('chars60_rank_1980s.csv', index=0)
chars_1990s.to_csv('chars60_rank_1990s.csv', index=0)
chars_2000s.to_csv('chars60_rank_2000s.csv', index=0)
chars_2010s.to_csv('chars60_rank_2010s.csv', index=0)
#
# print(chars_2010s[chars_2010s['permno']==14593][['permno', 'jdate']])
