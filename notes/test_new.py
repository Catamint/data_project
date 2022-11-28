import random
import pandas as pd
import numpy as np
# min_list=[[1,3],[2,2],[3,1]]
# print(sorted(min_list,key=lambda x: x[-1])[-1][0])



df = pd.DataFrame({'User_id':[1,1,1,1,4,1,1,1,1,1,1,1,1,1,4],'b':[2,2,2,5,34,2,64,3,2,7,2,9,2,2,7]})
df_2=pd.DataFrame({'a':[1,4],'e':[5,8]})

# print(pd.DataFrame(list(set(df['c'].tolist()))))
print(df)
# print(pd.merge(df,df_2,on='a',how='left'))

def next_date_series(data=pd.DataFrame(), column='column_not_found'):
    users=data[['User_id']].groupby('User_id').agg(sum)
    users['next']=np.nan
    sorted_data=data[['User_id', column]]
    sorted_data['temp_index']=list(range(len(sorted_data)))
    sorted_data=sorted_data.sort_values(by=[column,'temp_index'],ascending=False)
    sorted_data['next_'+column]=np.nan
    print(sorted_data)
    for index,item in sorted_data.iterrows():
        user_id=int(item['User_id'])
        # print(index)
        sorted_data['next_'+column][index]=users['next'][user_id]
        users['next'][user_id]=item[column]
    sorted_data=sorted_data.sort_index()
    next_series=sorted_data['next_'+column]
    return next_series

df['next']=next_date_series(df,'b')
print(df)
# df_u=df[df['c']>0].copy()
# print(df_u)


'''
pvt=pd.pivot_table(df,index=['a','b'],values='d', aggfunc=len).reset_index().rename(columns={'b':'ab_1'})
print(pvt)
pvt_2=pd.pivot_table(pvt,index='a',values='ab_1',aggfunc=len).reset_index()
df=pd.merge(df,pvt_2,on='a',how='left')
print(df)

pvt=pd.pivot_table(df_u,index=['a','b'],values='d', aggfunc=len).reset_index().rename(columns={'b':'ab_2'})
pvt_2=pd.pivot_table(pvt,index='a',values='ab_2',aggfunc=len).reset_index()
df=pd.merge(df,pvt_2,on='a',how='left')
print(df)
# print(pvt_2[''])
'''

'''
gb=df.groupby('a').agg({'b':lambda x:len(set(x))}).reset_index().rename(columns={'b':'ab'})
print(gb)
df=pd.merge(df,gb,on='a',how='left')
print(df)

gb=df_u.groupby('a').agg({'b':lambda x:len(set(x))}).reset_index().rename(columns={'b':'ab_u'})
print(gb)
df=pd.merge(df,gb,on='a',how='left')
df['ab_u']=df['ab_u'].map(lambda x : 0 if x!=x else int(x))
print(df)


# print(random.sample(list1,1))
'''