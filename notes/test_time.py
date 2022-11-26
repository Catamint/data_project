import pandas as pd
# min_list=[[1,3],[2,2],[3,1]]
# print(sorted(min_list,key=lambda x: x[-1])[-1][0])

from datetime import timedelta
from pandas.tseries.offsets import Day

df = pd.DataFrame({'a':[1,1,1,4],'b':[5,5,7,8],'c':[1,0,0,0],'d':[1,1,1,1]})
data=pd.DataFrame({'u':[1,1,1,2], 'c':[3,2,3,2], 't':[20210202,20210201,20210203,20240201]})
data["t"]=pd.to_datetime(data["t"],format="%Y%m%d")

data = data[['u', 'c', 't']]
df_t=data.set_index('t').sort_index()
print(df_t)

def inmap(date_r, uid, cid, df_t=df_t):
    a=(date_r-7*Day()).strftime('%Y-%m-%d')
    b=(date_r+7*Day()).strftime('%Y-%m-%d')
    df_t=df_t[a:b]
    return len(df_t[(df_t['c']==cid) & (df_t['u']==uid)])

data['sum']=list(map(inmap, data['t'], data['u'], data['c']))

print(data)



# df_2["t"]=pd.to_datetime(df_2["t"],format="%Y%m%d")

# five_days=pd.to_timedelta(5,'day')

# df_2['t']=df_2['t'].map(lambda x: x+five_days)

# d1=df_2[df_2['a']==1]
# d1=d1[(d1['t']>'2021/12/31') & (d1['t']<'2023/01/01')]
# print(d1)

# print(df_2[x for x in df_2['t'] if x in pd.date_range(start='2021/02/01',end='2021/02/03')]

