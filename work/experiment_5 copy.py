import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pandas import MultiIndex, Int16Dtype
# import shap

# 数据预处理
def preprocess(received):
    data=pd.DataFrame(received)
    print("preprocessing")

    ## 时间格式转换
    data["date_received"]=pd.to_datetime(
        data["Date_received"],format="%Y%m%d")

    if 'Date' in data.columns.tolist():
        data["date"]=pd.to_datetime(
            data["Date"],format="%Y%m%d")

    ## 满减格式规范
    ### 满减转换成折扣率 <注意仅当rate中有':'>
    def rate_to_count(rate):
        rate = str(rate)
        a, b = rate.split(sep=":")
        return (int(a)-int(b))/int(a)

    ### have_discount = 是否有优惠券
    data.insert(loc=4, column="have_discount",
                value=data["Discount_rate"].notnull())
    # print(off_train["have_discount"])

    ### discount = 折扣率
    data.insert(loc=5, column="discount",
                value=data["Discount_rate"].map(
                    lambda x: float(rate_to_count(x)) \
                        if ":" in str(x) else float(x)))
    # print(off_train["discount"])

    ### discount = 是否为满减
    data.insert(loc=5, column="is_manjian",
                value=data["Discount_rate"].map(
                    lambda x: True \
                        if ":" in str(x) else False))

    ### 满减门槛价格
    data.insert(loc=6,column="price_before",
                value=data["Discount_rate"].map(
                    lambda x: float(str(x).split(sep=":")[0]) \
                        if ":" in str(x) else -1))
    # print(off_train["price_before"])

    ## 异常值剔除
    ###

    ## 缺失值处理
    data["Distance"]=data["Distance"].map(
        lambda x: -1 if x!=x else x)

    ## 统一数据格式
    data["Distance"]=data["Distance"].map(int)

def next_date_series(data=pd.DataFrame(), column='column_not_found'):
    users=data[['User_id']].groupby('User_id').agg(sum)
    users['next']=np.nan
    sorted_data=data[['User_id', column]]
    sorted_data['temp_index']=list(range(len(sorted_data)))
    sorted_data=sorted_data.sort_values(by=[column,'temp_index'],ascending=False)
    sorted_data['next_'+column]=np.nan
    # print(sorted_data)
    for index,item in sorted_data.iterrows():
        user_id=int(item['User_id'])
        # print(index)
        sorted_data['next_'+column][index]=users['next'][user_id]
        users['next'][user_id]=item[column]
    sorted_data=sorted_data.sort_index()
    next_series=sorted_data['next_'+column].copy()
    print(next_series)
    return next_series

def get_user_feature_label(received):
    '''
    提取标签区间的用户特征
    - 保证传入的所有数据都有领券
    '''
    data = received.copy()
    print("getting user_feature_label")
    data['cnt']=1
    name_prifix="user_feature_label_" #前缀

    # 1、用户当月领券数
    pivoted=pd.pivot_table(
        data,index='User_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    # 用户当月领取特定种类券数
    pivoted=pd.pivot_table(
        data,index=['User_id', 'Coupon_id'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"this_coupon"})
    data=pd.merge(data,pivoted,on=['User_id', 'Coupon_id'],how='left')
    # 用户当天领券数
    pivoted=pd.pivot_table(
        data,index=['User_id', 'date_received'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"today_coupon"})
    data=pd.merge(data,pivoted,on=['User_id', 'date_received'],how='left')
    # 用户是否当天领取同一券
    pivoted=pd.pivot_table(
        data,index=['User_id', 'date_received', 'Coupon_id'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"today_this_coupon"})
    data=pd.merge(data,pivoted,on=['User_id', 'date_received', 'Coupon_id'],how='left')
    data.drop(['cnt'], axis=1, inplace=True)
    #     return data

    # def get_inter_feature_label(received):
    #     '''
    #     提取标签区间的行为特征
    #     - 保证传入的所有数据都有领券
    #     '''
    #     data = received.copy()
    #     print("getting user_feature_label")
    #     data['cnt']=1
    #     name_prifix="inter_feature_label_" #前缀
    # 用户距上次领券时间间隔

    # 用户距下次领券时间间隔

    # pivoted = data[["User_id","Date_received"]]
    # pivoted["date_received_next"]=next_date_series(pivoted,"Date_received")
    # print(pivoted)
    # print(pivoted[pivoted['date_received_next'].notnull()])

    # input()

    # pivoted[name_prifix+'timedelta_rec_use'] = received_used["Date"]-received_used["Date_received"]
    # pivoted=pivoted.groupby("User_id").agg(np.mean).reset_index()
    # print(pivoted)
    # data=pd.merge(data,pivoted,on='User_id',how='left')
    # data[name_prifix+'timedelta_rec_use']=data[name_prifix+'timedelta_rec_use'].map(lambda x: int(x) if x==x else -1)
    # print(data[name_prifix+'timedelta_rec_use'])

    # n天内领券数

    # 是否为n天内第一次/最后一次
    return data

def get_user_feature_history(received):
    '''
    提取历史区间(需要标签数据)的用户特征
    - 保证传入的所有数据都有领券
    - 保证传入的数据应有label列
    - 稍后处理缺失值中-1的影响
    '''
    print("getting user_feature_history")

    received=received.copy()
    data=pd.DataFrame(list(set(received['User_id'].tolist())),columns=['User_id'])
    received['cnt']=1
    name_prifix = "user_feature_history_" #前缀
    
    # 1、用户领券数
    pivoted=pd.pivot_table(
        received,index='User_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    # print(data)

    # 2、用户领券"并消费"数
    pivoted=pd.pivot_table(
        received,index='User_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"used"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    # print(data)

    # 3、用户领券"未消费"数
    data.insert(0,name_prifix+'not_use',data[name_prifix+'total_coupon']-data[name_prifix+"used"])
    # print(data)

    # 4、 用户领券"并消费"数 / 领券数、
    data.insert(0,name_prifix+'rate_used',data[name_prifix+"used"]/data[name_prifix+'total_coupon'])
    # print(data)
    # print(data[data[name_prifix+'rate_used']>0])

    received_used = received[received['label']==1].copy() # 数据中有消费的部分
    used_have_distance=received_used[received_used['Distance']!=-1]

    # 5、用户领取"并消费"优惠券的平均折扣率、
    pivoted=pd.pivot_table(
        received_used,index='User_id', values='discount', aggfunc=np.average
            ).reset_index().rename(columns={'discount': name_prifix+"used_average_discount"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    data[name_prifix+"used_average_discount"] = \
        data[name_prifix+"used_average_discount"].map(lambda x: -1 if x!=x else x)
    # print(data[data[name_prifix+"used_average_discount"]>0])

    # 6、用户领取"并消费"优惠券的平均距离、
    pivoted=pd.pivot_table(
        used_have_distance,index='User_id', values='Distance', aggfunc=np.average
            ).reset_index().rename(columns={'Distance': name_prifix+"used_average_distance"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    data[name_prifix+"used_average_distance"] = \
        data[name_prifix+"used_average_distance"].map(lambda x: -1 if x!=x else x)
    # print(data[data[name_prifix+"used_average_distance"]>0])

    # 7、用户在多少不同商家领取"并消费"优惠券、
    gpb=received_used.groupby('User_id').agg({'Merchant_id':lambda x:len(set(x))}) \
        .reset_index().rename(columns={'Merchant_id': name_prifix+"used_howmuch_merchant"})
    # print(gpb)
    data=pd.merge(data,gpb,on='User_id',how='left')
    data[name_prifix+"used_howmuch_merchant"] = \
        data[name_prifix+"used_howmuch_merchant"].map(lambda x : 0 if x!=x else int(x))
    # print(data[name_prifix+"used_howmuch_merchant"])

    # 8、用户在多少不同商家领取优惠券、
    gpb=received.groupby('User_id').agg({'Merchant_id':lambda x:len(set(x))}) \
        .reset_index().rename(columns={'Merchant_id': name_prifix+"howmuch_merchant"})
    data=pd.merge(data,gpb,on='User_id',how='left')
    data[name_prifix+"howmuch_merchant"] = \
        data[name_prifix+"howmuch_merchant"].map(lambda x : 0 if x!=x else int(x))
    # print(data[name_prifix+"howmuch_merchant"])

    # 9、用户在多少不同商家领取"并消费"优惠券 / 用户在多少不同商家领取优惠券
    data.insert(0,name_prifix+'rate_used_howmuch_merchant',
        data[name_prifix+"used_howmuch_merchant"]/data[name_prifix+'howmuch_merchant'])
    # print(data[name_prifix+'rate_used_howmuch_merchant'])
    # print(data[data[name_prifix+'rate_used_howmuch_merchant']>0])

    # 用户未领券消费数
    # 用户消费数
    # 用户用券购买平均距离
    pivoted=used_have_distance[['User_id','Distance']].groupby('User_id').agg(np.mean)
    pivoted=pivoted.reset_index().rename(columns={'Distance': name_prifix+"used_mean_distance"})
    print(pivoted)
    data=pd.merge(data,pivoted,on='User_id',how='left')
    data[name_prifix+'used_mean_distance']=data[name_prifix+'used_mean_distance'].map(lambda x: int(x) if x==x else -1)
    # 用户用券购买最小距离 最大距离

    # 用户领券到消费平均时间间隔
    pivoted = received_used[["User_id"]]
    pivoted[name_prifix+'timedelta_rec_use'] = received_used["Date"]-received_used["Date_received"]
    pivoted=pivoted.groupby("User_id").agg(np.mean).reset_index()
    print(pivoted)
    data=pd.merge(data,pivoted,on='User_id',how='left')
    data[name_prifix+'timedelta_rec_use']=data[name_prifix+'timedelta_rec_use'].map(lambda x: int(x) if x==x else -1)
    print(data[name_prifix+'timedelta_rec_use'])


    # 用户领券到消费最小间隔 最大间隔
    # 消费时间间隔

    # 领券消费时间间隔

    received.drop(['cnt'], axis=1, inplace=True)
    return data

# wrong...
'''
def get_user_feature_predict(dataset):
    data=pd.DataFrame(dataset).copy()
    from pandas.tseries.offsets import Day

    data = data[['User_id', 'Coupon_id', 'date_received']]
    df_t=data.set_index('date_received').sort_index()
    print(df_t)

    def inmap(date_r, uid, cid, df_t=df_t):
        df_t=df_t[(df_t['Coupon_id']==cid) & (df_t['User_id']==uid)]
        a=(date_r-7*Day()).strftime('%Y-%m-%d')
        b=(date_r+7*Day()).strftime('%Y-%m-%d')
        df_t=df_t[a:b]
        return len(df_t)

    data['sum']=list(map(inmap, data['date_received'], data['User_id'], data['Coupon_id']))

    print(data)
    return data
'''

def get_coupon_feature_history(received):
    '''
    提取历史区间(需要标签数据)的优惠券特征
    - 保证传入的所有数据都有领券
    - 保证传入的数据应有label列
    - 分数下降: 2/3/4
    '''
    print("getting coupon_feature_history")

    received=received.copy()
    data=pd.DataFrame(list(set(received['Coupon_id'].tolist())),columns=['Coupon_id'])
    received['cnt']=1
    name_prifix = "coupon_feature_history_" #前缀

    # 1、当前种类优惠券被领取数
    pivoted=pd.pivot_table(
        received,index='Coupon_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    data=pd.merge(data,pivoted,on='Coupon_id',how='left')

    # 2、当前种类优惠券被领取"并消费"数 分数下降
    pivoted=pd.pivot_table(
        received,index='Coupon_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"used"})
    data=pd.merge(data,pivoted,on='Coupon_id',how='left')
    # print(data)

    # 3、当前种类优惠券被领取"未消费"数 分数下降
    data.insert(0,name_prifix+'not_use',data[name_prifix+'total_coupon']-data[name_prifix+"used"])
    # print(data)

    # 4、 当前种类优惠券被领取"并消费"数 / 领取数 分数下降
    data.insert(0,name_prifix+'rate_used',data[name_prifix+"used"]/data[name_prifix+'total_coupon'])
    # print(data)
    # print(data[data[name_prifix+'rate_used']>0])

    received.drop(['cnt'], axis=1, inplace=True)
    return data


def get_merchant_feature_label():

    # 发放优惠券数
    # 发放优惠券种类数
    # (n天内)领券人数

    return

def get_merchant_feature_history():
    # 店家领券购买量
    # 店家不领券购买量
    # 发放优惠券数
    # 核销率
    # 用券购买此店家的平均距离/最小最大距离
    return

def get_merchant_coupon_feature_label():

    # 发放特定优惠券的领券数

    return

def get_merchant_user_feature_label():

    # 领取特定店家优惠券的领券数

    return

def get_user_merchant_feature_history(arg_received, arg_all):
    '''
    提取历史区间(需要标签数据)商家和用户特征
    - arg_received:有领券的集合, arg_all: 全体集合
    - 保证传入的数据应有label列
    - features将合并到data_received
    - 稍后处理缺失值中-1的影响
    '''
    print("getting user_merchant_feature_history")
    data_rec=arg_received.copy()[['User_id','Merchant_id','label','Distance']]
    data_all=arg_all.copy()[['User_id','Merchant_id','label','Distance']]
    data_rec['cnt']=1
    data_all['cnt']=1

    features=data_rec[['User_id','Merchant_id']].copy()
    # feature_list=[]

    # print(data)
    name_prifix = "merchant_feature_history_" #前缀

    # 用户在店家领券并消费数 分数下降
    pivoted=pd.pivot_table(
        data_rec,index=['User_id', 'Merchant_id'], values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_merchant_rec_used"})
    features=pd.merge(features,pivoted,on=['User_id','Merchant_id'],how='left')

    # 用户在店家消费数
    pivoted=pd.pivot_table(
        data_all,index=['User_id', 'Merchant_id'], values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_merchant_used"})
    features=pd.merge(features,pivoted,on=['User_id','Merchant_id'],how='left')

    # 率
    user_rec_used = pd.pivot_table(
        data_rec,index='User_id', values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_rec_used"})
    data_rec=pd.merge(data_rec, user_rec_used, on='User_id', how='left')
    
    user_used = pd.pivot_table(
        data_all,index='User_id', values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_used"})
    data_rec=pd.merge(data_rec, user_used, on='User_id', how='left')
    
    # 用户在此商家领券购买数/领券购买总数 分数下降
    features[name_prifix+"user_merchant_rec_used_rate"] = \
        features[name_prifix+"user_merchant_rec_used"] / data_rec[name_prifix+"user_rec_used"]

    # 用户在此商家购买数/购买总数 分数下降
    features[name_prifix+"user_merchant_used_rate"] = \
            features[name_prifix+"user_merchant_used"] / data_rec[name_prifix+"user_used"]

    # # 用户与店家平均距离 分数下降
    used_have_distance=data_rec[data_rec['Distance']!=-1]
    pivoted=pd.pivot_table(
        used_have_distance,index=['User_id', 'Merchant_id'], values='Distance',aggfunc=np.mean
            ).reset_index().rename(columns={'Distance': name_prifix+"user_merchant_mean_distance"})
    features=pd.merge(features,pivoted,on=['User_id','Merchant_id'],how='left')
    print(features[name_prifix+'user_merchant_mean_distance'])

    # 一个客户在一个商家一共收到的优惠券

    return features


def get_label(data):
    # 数据打标
    total_seconds = 15*24*3600.0
    data['label']=list(map(lambda x,y: 
        1 if (x - y).total_seconds()<=total_seconds else 0, data['date'], data['date_received']))

def get_dataset(field):
    '''
    构造数据集
    '''
    dataset=pd.DataFrame(field).copy()

    if 'Date' in dataset.columns.tolist():
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        # inplace?
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)

    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))

    return dataset

def model_xgb(train, validate):
    '''
    训练模型
    - return (model: xgb_booster)
    '''
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0,
            #   'lambda': 1,
              'lambda': 10,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}
    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dval = xgb.DMatrix(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=validate['label'])
    
    # 训练
    watchlist = [(dtrain, 'train'),(dval, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)

    return model

def test_online(model, test):
    '''
    返回线上测试集的预测结果
    - args (model: xgb_booster, test: DataFrame)
    - returns (result, feat_importance)
    '''
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))

    # 预测
    predict = model.predict(dtest)

    # 处理结果
    predict = pd.DataFrame(predict, columns=['pred'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    return result

def interval(data,contain,beginning,lenth):
    '''
    划分区间
    '''
    field = data[
        data[contain].isin(pd.date_range(beginning, periods=lenth))].copy()
    return field

def get_feat_importance(model):
    # 特征重要性
    feature_importance = pd.DataFrame(columns=['feature', 'importance'])
    feature_importance['feature'] = model.get_score().keys()
    feature_importance['importance'] = model.get_score().values()
    feature_importance.sort_values(['importance'], ascending=False, inplace=True)
    feature_importance.to_csv('result/feat_importance.csv', index=False, header=None)
    print("feature_importance is saved.")

def get_result(model,test):
    # 线上验证
    result = test_online(model, test=test)
    result.to_csv('result/result.csv', index=False, header=None)
    print("results are saved.")


off_train = pd.read_csv("resourse/data/ccf_offline_stage1_train.csv")
off_test = pd.read_csv("resourse/data/ccf_offline_stage1_test_revised.csv")

preprocess(off_train)
get_label(off_train)
preprocess(off_test)

# 数据划分
train_history_field=interval(off_train,'date_received','2016/3/2',60).copy()
validate_history_field=interval(off_train,'date_received','2016/1/16',60).copy()
test_history_field=interval(off_train,'date_received','2016/4/17',60).copy()

all_history_field_t=interval(off_train,'date','2016/3/2',60)
all_history_field_v=interval(off_train,'date','2016/1/16',60)
all_history_field_test=interval(off_train,'date','2016/4/17',60)

train = interval(off_train,'date_received','2016/5/16',31).copy() # 训练集 0330
validate = interval(off_train,'date_received','2016/3/1',31).copy() # 验证集
test = off_test # 测试集

# 数据划分
train_history_field=interval(off_train,'date_received','2016/3/2',60).copy()
validate_history_field=interval(off_train,'date_received','2016/1/16',60).copy()
test_history_field=interval(off_train,'date_received','2016/4/17',60).copy()

all_history_field_t=interval(off_train,'date','2016/3/2',60)
all_history_field_v=interval(off_train,'date','2016/1/16',60)
all_history_field_test=interval(off_train,'date','2016/4/17',60)

train = interval(off_train,'date_received','2016/5/16',31).copy() # 训练集 0330
validate = interval(off_train,'date_received','2016/3/1',31).copy() # 验证集
test = off_test # 测试集

# 提取特征
train=get_user_feature_label(train)
user_feature_history_train=get_user_feature_history(train_history_field)
coupon_feature_history_train=get_coupon_feature_history(train_history_field)
merchant_feature_history_train=get_user_merchant_feature_history(train_history_field,all_history_field_t)
# 合并特征
train=pd.merge(train, user_feature_history_train, on='User_id', how='left')
train=pd.merge(train, coupon_feature_history_train, on='Coupon_id', how='left')
train=pd.merge(train, merchant_feature_history_train, on=['User_id','Merchant_id'], how='left')
# 构造数据集
train=get_dataset(train)


validate=get_user_feature_label(validate)
user_feature_history_val=get_user_feature_history(validate_history_field)
coupon_feature_history_val=get_coupon_feature_history(validate_history_field)
merchant_feature_history_val=get_user_merchant_feature_history(validate_history_field,all_history_field_v)
validate=pd.merge(validate, user_feature_history_val, on='User_id', how='left')
validate=pd.merge(validate, coupon_feature_history_val, on='Coupon_id', how='left')
validate=pd.merge(validate, merchant_feature_history_val, on=['User_id','Merchant_id'], how='left')
validate=get_dataset(validate)


test=get_user_feature_label(test)
user_feature_history_test=get_user_feature_history(test_history_field)
coupon_feature_history_test=get_coupon_feature_history(test_history_field)
merchant_feature_history_test=get_user_merchant_feature_history(test_history_field,all_history_field_test)
test=pd.merge(test, user_feature_history_test, on='User_id', how='left')
test=pd.merge(test, coupon_feature_history_test, on='Coupon_id', how='left')
test=pd.merge(test, merchant_feature_history_test, on=['User_id','Merchant_id'], how='left')
test=get_dataset(test)

# 训练
model = model_xgb(train, validate)
get_feat_importance(model)
# model.save_model("model/")
# get_result(model,test)


### end ###



def get_intervaled_1(off_train, off_test):

    # 训练集历史区间、中间区间、标签区间
    train_history_field = interval(off_train, 'date_received', '2016/3/2', 60)  # [20160302,20160501)
    # train_middle_field = interval(off_train, 'date','2016/5/1', 15) # [20160501,20160516)
    train_label_field = interval(off_train, 'date_received', '2016/5/16', 31)  # [20160516,20160616)

    # 验证集历史区间、中间区间、标签区间
    validate_history_field = interval(off_train, 'date_received', '2016/1/16', 60)  # [20160116,20160316)
    # validate_middle_field = interval(off_train, 'date', '2016/3/16', 15)  # [20160316,20160331)
    validate_label_field = interval(off_train, 'date_received', '2016/3/31', 31)  # [20160331,20160501)

    # 测试集历史区间、中间区间、标签区间
    test_history_field = interval(off_train, 'date_received', '2016/4/17', 60)  # [20160417,20160616)
    # test_middle_field = interval(off_train, 'date', '2016/6/16', 15)  # [20160616,20160701)
    test_label_field = off_test.copy()  # [20160701,20160801)

    return train_history_field, train_label_field, \
        validate_history_field, validate_label_field, \
        test_history_field, test_label_field
