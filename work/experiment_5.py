import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
# from pandas import MultiIndex, Int16Dtype
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

def get_user_feature_history(received, concat_on):
    '''
    提取历史区间(需要标签数据)的用户特征
    - 保证传入的所有数据都有领券
    - 保证传入的数据应有label列
    - 稍后处理缺失值中-1的影响
    '''
    print("getting user_feature_history")
    received=received.copy()
    features=concat_on[['User_id']].copy()
    received['cnt']=1
    name_prifix = "user_feature_history_" #前缀
    
    # 1、用户领券数
    pivoted=pd.pivot_table(
        received,index='User_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    features=pd.merge(features,pivoted,on='User_id',how='left')
    # 2、用户领券"并消费"数
    pivoted=pd.pivot_table(
        received,index='User_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"used"})
    features=pd.merge(features,pivoted,on='User_id',how='left')
    # 3、用户领券"未消费"数
    features[name_prifix+'not_use']=features[name_prifix+'total_coupon']-features[name_prifix+"used"]
    # 4、 用户领券"并消费"数 / 领券数、
    features[name_prifix+'rate_used']=features[name_prifix+"used"]/features[name_prifix+'total_coupon']
    # 对前四个特征进行缺失值处理
    features[name_prifix+"total_coupon"] = features[name_prifix+"total_coupon"].map(lambda x: 0 if x!=x else int(x))
    features[name_prifix+"used"] = features[name_prifix+"used"].map(lambda x: 0 if x!=x else int(x))
    features[name_prifix+"not_use"] = features[name_prifix+"not_use"].map(lambda x: 0 if x!=x else int(x))
    features[name_prifix+"rate_used"] = features[name_prifix+"rate_used"].map(lambda x: -1 if x!=x else x)

    received_used = received[received['label']==1].copy() # 数据中有消费的部分
    used_have_distance=received_used[received_used['Distance']!=-1]

    # 5、用户领取"并消费"优惠券的平均折扣率、
    pivoted=pd.pivot_table(
        received_used,index='User_id', values='discount', aggfunc=np.average
            ).reset_index().rename(columns={'discount': name_prifix+"used_average_discount"})
    features=pd.merge(features,pivoted,on='User_id',how='left')
    features[name_prifix+"used_average_discount"] = \
        features[name_prifix+"used_average_discount"].map(lambda x: -1 if x!=x else x)

    # 6、用户领取"并消费"优惠券的平均距离、
    pivoted=pd.pivot_table(
        used_have_distance,index='User_id', values='Distance', aggfunc=np.average
            ).reset_index().rename(columns={'Distance': name_prifix+"used_average_distance"})
    features=pd.merge(features,pivoted,on='User_id',how='left')
    features[name_prifix+"used_average_distance"] = \
        features[name_prifix+"used_average_distance"].map(lambda x: -1 if x!=x else int(x))

    # 7、用户在多少不同商家领取"并消费"优惠券
    gpb=received_used.groupby('User_id').agg({'Merchant_id':lambda x:len(set(x))}) \
        .reset_index().rename(columns={'Merchant_id': name_prifix+"used_howmuch_merchant"})
    features=pd.merge(features,gpb,on='User_id',how='left')
    # 8、用户在多少不同商家领取优惠券
    gpb=received.groupby('User_id').agg({'Merchant_id':lambda x:len(set(x))}) \
        .reset_index().rename(columns={'Merchant_id': name_prifix+"howmuch_merchant"})
    features=pd.merge(features,gpb,on='User_id',how='left')
    # 9、用户在多少不同商家领取"并消费"优惠券 / 用户在多少不同商家领取优惠券
    features.insert(0,name_prifix+'rate_used_howmuch_merchant',
        features[name_prifix+"used_howmuch_merchant"]/features[name_prifix+'howmuch_merchant'])
    # 对特征7-9进行缺失值处理
    features[name_prifix+"used_howmuch_merchant"] = \
        features[name_prifix+"used_howmuch_merchant"].map(lambda x: 0 if x!=x else int(x))
    features[name_prifix+"howmuch_merchant"] = \
        features[name_prifix+"howmuch_merchant"].map(lambda x: 0 if x!=x else int(x))
    features[name_prifix+"rate_used_howmuch_merchant"] = \
        features[name_prifix+"rate_used_howmuch_merchant"].map(lambda x: -1 if x!=x else x)

    # 用户未领券消费数
    # 用户消费数
    # 用户用券购买平均距离
    pivoted=used_have_distance[['User_id','Distance']].groupby('User_id').agg(np.mean)
    pivoted=pivoted.reset_index().rename(columns={'Distance': name_prifix+"used_mean_distance"})
    # print(pivoted)
    features=pd.merge(features,pivoted,on='User_id',how='left')
    features[name_prifix+'used_mean_distance']=features[name_prifix+'used_mean_distance'].map(lambda x: int(x) if x==x else -1)
    # 用户用券购买最小距离 最大距离

    # 用户领券到消费平均时间间隔
    pivoted = received_used[["User_id"]]
    pivoted[name_prifix+'timedelta_rec_use'] = received_used["Date"]-received_used["Date_received"]
    pivoted=pivoted.groupby("User_id").agg(np.mean).reset_index()
    # print(pivoted)
    features=pd.merge(features,pivoted,on='User_id',how='left')
    features[name_prifix+'timedelta_rec_use']=features[name_prifix+'timedelta_rec_use'].map(lambda x: int(x) if x==x else -1)

    # 用户领券到消费最小间隔 最大间隔
    # 消费时间间隔

    # 领券消费时间间隔
    concat_on=pd.concat([concat_on, features.drop(['User_id'],axis=1)], axis=1)
    # concat_on=pd.merge(concat_on, features, on='User_id', how='left')
    received.drop(['cnt'], axis=1, inplace=True)
    return concat_on

def get_coupon_feature_history(received, concat_on):
    '''
    提取历史区间(需要标签数据)的优惠券特征
    - 保证传入的所有数据都有领券
    - 保证传入的数据应有label列
    - 分数下降: 2/3/4
    '''
    print("getting coupon_feature_history")
    received=received.copy()
    features=concat_on[['Coupon_id']].copy()
    received['cnt']=1
    name_prifix = "coupon_feature_history_" #前缀

    # 1、当前种类优惠券被领取数
    pivoted=pd.pivot_table(
        received,index='Coupon_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    features=pd.merge(features,pivoted,on='Coupon_id',how='left')
    # 2、当前种类优惠券被领取"并消费"数 分数下降
    pivoted=pd.pivot_table(
        received,index='Coupon_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"used"})
    features=pd.merge(features,pivoted,on='Coupon_id',how='left')
    # 3、当前种类优惠券被领取"未消费"数 分数下降
    features.insert(0,name_prifix+'not_use',features[name_prifix+'total_coupon']-features[name_prifix+"used"])
    # 4、 当前种类优惠券被领取"并消费"数 / 领取数 分数下降
    features.insert(0,name_prifix+'rate_used',features[name_prifix+"used"]/features[name_prifix+'total_coupon'])
    # 1-4缺失值处理
    features[name_prifix+'total_coupon'] = features[name_prifix+'total_coupon'].map(lambda x: int(x) if x==x else -1)
    features[name_prifix+'used'] = features[name_prifix+'used'].map(lambda x: int(x) if x==x else -1)
    features[name_prifix+'not_use'] = features[name_prifix+'not_use'].map(lambda x: int(x) if x==x else -1)
    features[name_prifix+'rate_used'] = features[name_prifix+'rate_used'].map(lambda x: x if x==x else -1)

    # 根据优惠券折扣率等提取特征
    concat_on=pd.concat([concat_on, features.drop(['Coupon_id'],axis=1)], axis=1)
    # concat_on=pd.merge(concat_on, features, on='Coupon_id', how='left')
    received.drop(['cnt'], axis=1, inplace=True)

    return concat_on

def get_merchant_feature_label(received):
    '''
    提取标签区间的商家特征
    - 保证传入的所有数据都有领券
    '''
    data = received[['Merchant_id','User_id','Coupon_id']].copy()
    features=received[['Merchant_id']]
    print("getting merchant_feature_label")
    data['cnt']=1
    name_prifix="merchant_feature_label_" #前缀

    # 发放优惠券数
    pivoted=data.groupby('Merchant_id').agg({'cnt': sum}).reset_index() \
        .rename(columns={'Merchant_id': name_prifix+"howmuch_coupon"})
    features=pd.merge(features,pivoted,on='Merchant_id',how='left')

    # 发放优惠券种类数
    pivoted=data.groupby('Merchant_id').agg({'Coupon_id': len}).reset_index() \
        .rename(columns={'Merchant_id': name_prifix+"howmuch_kind_coupon"})
    features=pd.merge(features,pivoted,on='Merchant_id',how='left')

    # 当天领券人数
    pivoted=data.groupby('Merchant_id').agg({'date_received': len}).reset_index() \
        .rename(columns={'Merchant_id': name_prifix+"today_total_received"})
    features=pd.merge(features,pivoted,on='Merchant_id',how='left')

    # (n天内)领券人数
    
    return features

def get_merchant_feature_history(arg_received, arg_all, concat_on):
    '''
    提取历史区间的商家特征
    - 保证传入的所有数据都有领券
    - 稍后试试把history的领券数删除
    '''
    print("getting coupon_feature_history")
    data_received=arg_received.copy()
    # data_all=arg_all.copy()
    data_received['cnt']=1
    name_prifix = "merchant_feature_history_" #前缀

    # 发放优惠券数
    pivoted=pd.pivot_table(
        data_received,index='Merchant_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    concat_on=pd.merge(concat_on,pivoted,on='Merchant_id',how='left')
    # 15天内被领券购买量
    pivoted=pd.pivot_table(
        data_received,index='Merchant_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"rec_used"})
    concat_on=pd.merge(concat_on,pivoted,on='Merchant_id',how='left')
    # 店家总销量

    # 15天内核销率
    concat_on[name_prifix+'rate_used']=concat_on[name_prifix+"rec_used"]/concat_on[name_prifix+'total_coupon']

    # 用券率
    # 缺失值处理
    concat_on[name_prifix+"total_coupon"] = \
        concat_on[name_prifix+"total_coupon"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"rec_used"] = \
        concat_on[name_prifix+"rec_used"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"rate_used"] = \
        concat_on[name_prifix+"rate_used"].map(lambda x: -1 if x!=x else x)

    data_received.drop(['cnt'], axis=1, inplace=True)

    return concat_on

def get_merchant_coupon_feature_label():

    # 发放特定优惠券的领券数

    return

def get_merchant_user_feature_label():

    # 领取特定店家优惠券的领券数

    return

def get_user_merchant_feature_history(arg_received, arg_all, concat_on):
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
    name_prifix = "user_merchant_feature_history_" #前缀

    # 用户在店家领券并消费数 分数下降
    pivoted=pd.pivot_table(
        data_rec,index=['User_id', 'Merchant_id'], values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_merchant_rec_used"})
    concat_on=pd.merge(concat_on,pivoted,on=['User_id','Merchant_id'],how='left')

    # 用户在店家消费数
    pivoted=pd.pivot_table(
        data_all,index=['User_id', 'Merchant_id'], values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_merchant_used"})
    concat_on=pd.merge(concat_on,pivoted,on=['User_id','Merchant_id'],how='left')

    # 率
    user_rec_used = pd.pivot_table(
        data_rec,index='User_id', values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_rec_used"})
    concat_on=pd.merge(concat_on, user_rec_used, on='User_id', how='left')
    
    user_used = pd.pivot_table(
        data_all,index='User_id', values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_used"})
    concat_on=pd.merge(concat_on, user_used, on='User_id', how='left')
    
    # 用户在此商家领券购买数/领券购买总数 分数下降
    concat_on[name_prifix+"user_merchant_rec_used_rate"] = \
        concat_on[name_prifix+"user_merchant_rec_used"] / concat_on[name_prifix+"user_rec_used"]
    # 用户在此商家购买数/购买总数 分数下降
    concat_on[name_prifix+"user_merchant_used_rate"] = \
            concat_on[name_prifix+"user_merchant_used"] / concat_on[name_prifix+"user_used"]

    concat_on=concat_on.drop([name_prifix+"user_rec_used", name_prifix+"user_used"],axis=1)

    concat_on[name_prifix+"user_merchant_rec_used"] = \
        concat_on[name_prifix+"user_merchant_rec_used"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"user_merchant_used"] = \
        concat_on[name_prifix+"user_merchant_used"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"user_merchant_rec_used_rate"] = \
        concat_on[name_prifix+"user_merchant_rec_used_rate"].map(lambda x: -1 if x!=x else x)
    concat_on[name_prifix+"user_merchant_used_rate"] = \
        concat_on[name_prifix+"user_merchant_used_rate"].map(lambda x: -1 if x!=x else x)

    # # 用户与店家平均距离 分数下降
    used_have_distance=data_rec[data_rec['Distance']!=-1]
    pivoted=pd.pivot_table(
        used_have_distance,index=['User_id', 'Merchant_id'], values='Distance',aggfunc=np.mean
            ).reset_index().rename(columns={'Distance': name_prifix+"user_merchant_mean_distance"})
    concat_on=pd.merge(concat_on,pivoted,on=['User_id','Merchant_id'],how='left')

    concat_on[name_prifix+"user_merchant_mean_distance"] = \
        concat_on[name_prifix+"user_merchant_mean_distance"].map(lambda x: -1 if x!=x else int(x))
    # print(concat_on[name_prifix+'user_merchant_mean_distance'])

    # 一个客户在一个商家一共收到的优惠券
    return concat_on


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
            #   'tree_method': ,
              'scale_pos_weight': 1}
    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dval = xgb.DMatrix(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=validate['label'])
    
    # 训练
    watchlist = [(dtrain, 'train'),(dval, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=2400, evals=watchlist)

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


def get_feature_for(history_field, all_his_field, label_field, filename=None):
    # 提取特征
    label_field=get_user_feature_label(label_field)
    label_field=get_user_feature_history(history_field, label_field)
    label_field=get_coupon_feature_history(history_field, label_field)
    label_field=get_merchant_feature_history(history_field,all_his_field,label_field) # 不对劲
    label_field=get_user_merchant_feature_history(history_field,all_his_field,label_field)

    if filename != None:
        label_field.to_csv('result/features/'+filename+'.csv')
    # 构造数据集
    label_field=get_dataset(label_field)
    return label_field


off_train = pd.read_csv("resourse/data/ccf_offline_stage1_train.csv")
off_test = pd.read_csv("resourse/data/ccf_offline_stage1_test_revised.csv")

preprocess(off_train)
get_label(off_train)
preprocess(off_test)

# 数据划分
train_history_field=interval(off_train,'date_received','2016/1/1',60).copy()
validate_history_field=interval(off_train,'date_received','2016/3/1',60).copy()
test_history_field=interval(off_train,'date_received','2016/4/17',60).copy()

all_history_field_t=interval(off_train,'date','2016/1/1',60)
all_history_field_v=interval(off_train,'date','2016/3/1',60)
all_history_field_test=interval(off_train,'date','2016/4/17',60)

train = interval(off_train,'date_received','2016/3/1',62).copy() # 训练集 0330
validate = interval(off_train,'date_received','2016/5/17',31).copy() # 验证集
test = off_test # 测试集

train=get_feature_for(train_history_field, all_history_field_t, train, 'train')
# input('ok')
validate=get_feature_for(validate_history_field, all_history_field_v, validate)
test=get_feature_for(test_history_field, all_history_field_test, test)

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
