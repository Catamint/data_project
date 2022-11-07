import numpy as np
import pandas as pd
data = pd.read_csv("e:\OneDrive - stu.cqupt.edu.cn\#codes\data_project\#test\data\ccf_offline_stage1_train.csv")

data.insert(loc=0,column='cnt',value=[int(x) for x in data["Coupon_id"].notnull()])
name_prifix="feature_basic_" #前缀

data["date_received"]=pd.to_datetime(
    data["Date_received"],format="%Y%m%d")
if 'Date' in data.columns.tolist():
    data["date"]=pd.to_datetime(
        data["Date"],format="%Y%m%d")

total_seconds = 15*24*3600.0
data.insert(0,'label', list(map(lambda x,y: 1 if (x - y).total_seconds()<=total_seconds else 0, data['date'],data['date_received'])))

# 1、领券数
pivoted=pd.pivot_table(
data,index='User_id', values='cnt', aggfunc=np.sum
    ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
print(pivoted)
data=pd.merge(data,pivoted,on='User_id',how='left')

# 2、领券"并消费"数
pivoted=pd.pivot_table(
data,index='User_id', values='label', aggfunc=np.sum
    ).reset_index().rename(columns={'label': name_prifix+"used"})
print(pivoted)
data=pd.merge(data,pivoted,on='User_id',how='left')

# 3、领券"未消费"数
data.insert(0,name_prifix+'not_use',data[name_prifix+'total_coupon']-data[name_prifix+"used"])
print(data[name_prifix+'not_use'])

# 4、 领券"并消费"数 / 领券数、
data.insert(0,name_prifix+'rate_used',data[name_prifix+"used"]/data[name_prifix+'total_coupon'])
print(data[name_prifix+'rate_used'])

print(data)

