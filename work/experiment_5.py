import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

# 数据预处理
def preprocess(data):
    off_train=pd.DataFrame(data)
    print("preprocessing")

    ## 时间格式转换
    off_train["date_received"]=pd.to_datetime(
        off_train["Date_received"],format="%Y%m%d")

    if 'Date' in off_train.columns.tolist():
        off_train["date"]=pd.to_datetime(
            off_train["Date"],format="%Y%m%d")

    ## 满减格式规范
    ### 满减转换成折扣率 <注意仅当rate中有':'>
    def rate_to_count(rate):
        rate = str(rate)
        a, b = rate.split(sep=":")
        return (int(a)-int(b))/int(a)

    ### have_discount = 是否有优惠券
    off_train.insert(loc=4, column="have_discount",
                value=off_train["Discount_rate"].notnull())
    # print(off_train["have_discount"])

    ### discount = 折扣率
    off_train.insert(loc=5, column="discount",
                value=off_train["Discount_rate"].map(
                    lambda x: float(rate_to_count(x)) \
                        if ":" in str(x) else float(x)))
    # print(off_train["discount"])

    ### 满减门槛价格
    off_train.insert(loc=6,column="price_before",
                value=off_train["Discount_rate"].map(
                    lambda x: float(str(x).split(sep=":")[0]) \
                        if ":" in str(x) else float(x)))
    # print(off_train["price_before"])

    ## 异常值剔除
    ###

    ## 缺失值处理
    off_train["Distance"]=off_train["Distance"].map(
        lambda x: -1 if x!=x else x)

    ## 统一数据格式
    off_train["Distance"]=off_train["Distance"].map(int)



def get_user_feature(received):
    '''
    提取用户特征
    - 保证传入的所有数据都有领券
    '''
    data = received.copy()
    print("getting features")
    data['cnt']=1
    name_prifix="feature_" #前缀

    # 用户领取特定种类券数
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
    return data

def get_user_feature_l(received):
    '''
    提取需要标签数据的用户特征
    - 保证传入的所有数据都有领券
    - 保证传入的数据应有label列
    '''
    received=received.copy()
    data=pd.DataFrame(list(set(received['User_id'].tolist())),columns=['User_id'])
    received['cnt']=1
    name_prifix = "feature_basic_" #前缀
    
    # 1、用户领券数
    pivoted=pd.pivot_table(
        received,index='User_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    print(data)
    # 2、用户领券"并消费"数
    pivoted=pd.pivot_table(
        received,index='User_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"used"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    print(data)
    # 3、用户领券"未消费"数
    data.insert(0,name_prifix+'not_use',data[name_prifix+'total_coupon']-data[name_prifix+"used"])
    print(data)
    # 4、 用户领券"并消费"数 / 领券数、
    data.insert(0,name_prifix+'rate_used',data[name_prifix+"used"]/data[name_prifix+'total_coupon'])
    print(data)
    print(data[data[name_prifix+'rate_used']>0])

    received_used = received[received['label']==1].copy() # 数据中有消费的部分
    
    # 5、用户领取"并消费"优惠券的平均折扣率、
    pivoted=pd.pivot_table(
        received_used,index='User_id', values='discount', aggfunc=np.average
            ).reset_index().rename(columns={'discount': name_prifix+"used_average_discount"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    data[name_prifix+"used_average_discount"] = \
        data[name_prifix+"used_average_discount"].map(lambda x: -1 if x!=x else x)
    print(data[data[name_prifix+"used_average_discount"]>0])

    # 6、用户领取"并消费"优惠券的平均距离、
    pivoted=pd.pivot_table(
        received_used,index='User_id', values='Distance', aggfunc=np.average
            ).reset_index().rename(columns={'Distance': name_prifix+"used_average_distance"})
    data=pd.merge(data,pivoted,on='User_id',how='left')
    data[name_prifix+"used_average_distance"] = \
        data[name_prifix+"used_average_distance"].map(lambda x: -1 if x!=x else x)
    print(data[data[name_prifix+"used_average_distance"]>0])

    # 7、用户在多少不同商家领取"并消费"优惠券、
    gpb=received_used.groupby('User_id').agg({'Merchant_id':lambda x:len(set(x))}) \
        .reset_index().rename(columns={'Merchant_id': name_prifix+"used_howmuch_merchant"})
    print(gpb)
    data=pd.merge(data,gpb,on='User_id',how='left')
    data[name_prifix+"used_howmuch_merchant"] = \
        data[name_prifix+"used_howmuch_merchant"].map(lambda x : 0 if x!=x else int(x))
    print(data[name_prifix+"used_howmuch_merchant"])
    # 8、用户在多少不同商家领取优惠券、
    gpb=received.groupby('User_id').agg({'Merchant_id':lambda x:len(set(x))}) \
        .reset_index().rename(columns={'Merchant_id': name_prifix+"howmuch_merchant"})
    data=pd.merge(data,gpb,on='User_id',how='left')
    data[name_prifix+"howmuch_merchant"] = \
        data[name_prifix+"howmuch_merchant"].map(lambda x : 0 if x!=x else int(x))
    print(data[name_prifix+"howmuch_merchant"])
    # 9、用户在多少不同商家领取"并消费"优惠券 / 用户在多少不同商家领取优惠券
    data.insert(0,name_prifix+'rate_used_howmuch_merchant',
        data[name_prifix+"used_howmuch_merchant"]/data[name_prifix+'howmuch_merchant'])
    print(data[name_prifix+'rate_used_howmuch_merchant'])
    print(data[data[name_prifix+'rate_used_howmuch_merchant']>0])

    received.drop(['cnt'], axis=1, inplace=True)
    return data

def get_label(data):
    '''
    数据打标
    '''
    total_seconds = 15*24*3600.0
    data['label']=list(map(lambda x,y: 
        1 if (x - y).total_seconds()<=total_seconds else 0, data['date'], data['date_received']))
    # print(pd.pivot_table(data,index='label',values='cnt',aggfunc=len))


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


def model_xgb(train, validate, test):
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}
    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    # if validate != None:
    dval = xgb.DMatrix(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=validate['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    
    # 训练
    # if validate != None:
    watchlist = [(dtrain, 'train'),(dval, 'val')]
    # else:
        # watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
    
    # 预测
    predict = model.predict(dtest)

    # 处理结果
    predict = pd.DataFrame(predict, columns=['pred'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    return result, feat_importance

def interval(data,contain,beginning,lenth):
    '''
    划分区间
    '''
    field = data[
        data[contain].isin(pd.date_range(beginning, periods=lenth))]
    return field


off_train = pd.read_csv("#test\data\ccf_offline_stage1_train.csv")
off_test = pd.read_csv("#test\data\ccf_offline_stage1_test_revised.csv")

preprocess(off_train)
preprocess(off_test)
get_label(off_train)

# 数据划分

'''
# 训练集历史区间、中间区间、标签区间
train_history_field = interval(off_train, 'date_received', '2016/3/2', 60)  # [20160302,20160501)
train_middle_field = interval(off_train, 'date','2016/5/1', 15) # [20160501,20160516)
train_label_field = interval(off_train, 'date_received', '2016/5/16', 31)  # [20160516,20160616)

# 验证集历史区间、中间区间、标签区间
validate_history_field = interval(off_train, 'date_received', '2016/1/16', 60)  # [20160116,20160316)
validate_middle_field = interval(off_train, 'date', '2016/3/16', 15)  # [20160316,20160331)
validate_label_field = interval(off_train, 'date_received', '2016/3/31', 31)  # [20160331,20160501)

# 测试集历史区间、中间区间、标签区间
test_history_field = interval(off_train, 'date_received', '2016/4/17', 60)  # [20160417,20160616)
test_middle_field = interval(off_train, 'date', '2016/6/16', 15)  # [20160616,20160701)
test_label_field = off_test.copy()  # [20160701,20160801)

def train_easy():
    train = get_dataset(interval(off_train,'date_received','2016/3/16',60)) # 训练集
    validate = get_dataset(interval(off_train,'date_received','2016/5/17',31)) # 模型训练过程中用来检验的区间
    test = get_dataset(test_label_field) # 测试集

def train_validate_basic():
    pre_test=interval(off_train,'date_received','2016/5/31', 31) #准备进行预测的区间
    pre_test['Date']=np.nan
    pre_test['label']=0
    train = get_dataset(interval(off_train,'date_received','2016/4/1',61)) # 训练集
    # validate = get_dataset(interval(off_train,'date_received','2016/5/17',31)) # 训练过程中用来检验的区间
    test = get_dataset(pd.concat([interval(off_train,'date_received','2016/4/30', 31), pre_test], axis=0)) 
    #验证集(pre_test和前31天的数据)

def train_small():
    train = get_dataset(interval(off_train,'date_received','2016/4/1',60)) # 训练集
    pre_val=interval(off_train,'date_received','2016/5/31',31)
    pre_val['label']=0
    validate = get_dataset(pre_val) # 训练过程中用来检验的区间
    test = get_dataset(off_test) 
'''

# def train_final():
train = interval(off_train,'date_received','2016/3/1',62).copy() # 训练集 0330
validate = interval(off_train,'date_received','2016/5/17',31).copy() # 训练过程中用来检验的区间
test = off_test # 测试集

# 提取特征
train=get_user_feature(train)
validate=get_user_feature(validate)
test=get_user_feature(test)
user_feature_l=get_user_feature_l(train)

# 合并特征
train=pd.merge(train, user_feature_l, on='User_id', how='left')
validate=pd.merge(validate, user_feature_l, on='User_id', how='left')
test=pd.merge(test, user_feature_l, on='User_id', how='left')

# 构造数据集
train=get_dataset(train)
validate=get_dataset(validate)
test=get_dataset(test)


#测试集(off_test和前31天的数据)

# 线下验证
# result_off = model_xgb(train, validate.drop(['label'], axis=1))

# 线上训练
result, feat_importance = model_xgb(train, validate, test=test)
feat_importance.to_csv('feat_importance.csv', index=False, header=None)
result.to_csv('result.csv', index=False, header=None)

