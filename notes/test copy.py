import numpy as np
import pandas as pd
data = pd.read_csv("e:\OneDrive - stu.cqupt.edu.cn\#codes\data_project\#test\data\ccf_offline_stage1_train.csv")

data["date_received"]=pd.to_datetime(
    data["Date_received"],format="%Y%m%d")
if 'Date' in data.columns.tolist():
    data["date"]=pd.to_datetime(
        data["Date"],format="%Y%m%d")

data = data[
        data['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)

data.insert(loc=0,column='cnt',value=[int(x) for x in data["Coupon_id"].notnull()])

# data.insert(loc=0,column='cnt',value=[int(x) for x in data["Coupon_id"].notnull()])
# name_prifix="feature_basic_" #前缀

total_seconds = 15*24*3600.0
data.insert(0,'label', list(map(lambda x,y: 1 if (x - y).total_seconds()<=total_seconds else 0, data['date'],data['date_received'])))

data_used=data[data['label']==1] # 数据中有消费的部分

### cnt (排除Coupon_id==0的项)(无需如此)


name_prifix="feature_basic_" #前缀

# 6、用户领取"并消费"优惠券的平均距离、
pivoted=pd.pivot_table(
    data_used,index='User_id', values='Distance', aggfunc=np.average
        ).reset_index().rename(columns={'Distance': name_prifix+"used_average_distance"})
print(pivoted)
data=pd.merge(data,pivoted,on='User_id',how='left')
data[name_prifix+"used_average_distance"]=data["Distance"].map(lambda x: -1 if x!=x else x)


# 7、用户在多少不同商家领取"并消费"优惠券、
pivoted=pd.pivot_table(
    data_used, index=['User_id','Merchant_id'], values='cnt', aggfunc=len
        ).reset_index()
pivoted=pd.pivot_table(
    pivoted,index=['User_id','Merchant_id'],values='cnt',aggfunc=len
        ).reset_index().rename(columns={'Merchant_id': name_prifix+"used_howmuch_merchant"})
print(pivoted)
data=pd.merge(data,pivoted,on='User_id',how='left')
data[name_prifix+"used_howmuch_merchant"]=data["Distance"].map(lambda x: 0 if x!=x else x)

# 8、用户在多少不同商家领取优惠券、
pivoted=pd.pivot_table(
    data, index=['User_id','Merchant_id'], values='cnt', aggfunc=len
        ).reset_index()
pivoted=pd.pivot_table(
    pivoted,index=['User_id','Merchant_id'],values='cnt',aggfunc=len
        ).reset_index().rename(columns={'Merchant_id': name_prifix+"howmuch_merchant"})
print(pivoted)
data=pd.merge(data,pivoted,on='User_id',how='left')


# 9、用户在多少不同商家领取"并消费"优惠券 / 用户在多少不同商家领取优惠券
data.insert(0,name_prifix+'rate_used_howmuch_merchant',data[name_prifix+"used_howmuch_merchant"]/data[name_prifix+'howmuch_merchant'])
print(data[name_prifix+'rate_used_howmuch_merchant'])
print(data[data[name_prifix+'rate_used_howmuch_merchant']>1][name_prifix+'rate_used_howmuch_merchant'])

print(data)

