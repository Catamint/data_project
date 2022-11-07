import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

# 数据预处理
def preprocess(data):
    off_train=pd.DataFrame(data).copy()
    print("preprocessing")

    ## 时间格式转换
    # print(off_train["Date_received"])
    # print(off_train["Date"])

    off_train["date_received"]=pd.to_datetime(
        off_train["Date_received"],format="%Y%m%d")
    # print(off_train["date_received"][100:110])
    if 'Date' in off_train.columns.tolist():
        off_train["date"]=pd.to_datetime(
            off_train["Date"],format="%Y%m%d")
    # print(off_train["date"][100:110])

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
    ### distance
    # print(off_train.isnull().any())
    # print(off_train.isnull().sum()/len(off_train))
    off_train["Distance"]=off_train["Distance"].map(lambda x: -1 if x!=x else x)
    # print(off_train["Distance"][10:30])
    # print(off_train["Distance"].isnull().sum())

    ## 其他预处理
    ### 新增column:是否节假日

    ## 统一数据格式
    off_train["Distance"]=off_train["Distance"].map(int)
    # print(off_train["Distance"])
    return off_train


# 提特征
def get_feature(received):
    data = received.copy()
    print("getting features")

    ### cnt (排除Coupon_id==0的项)
    data.insert(loc=0,column='cnt',value=[int(x) for x in data["Coupon_id"].notnull()])

    name_prifix="feature_" #前缀

    # 领取特定种类券数
    pivoted=pd.pivot_table(
        data,index=['User_id', 'Coupon_id'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"_this_coupon"})
    # print(pivoted)
    data=pd.merge(data,pivoted,on=['User_id', 'Coupon_id'],how='left')
    
    # 当天领券数
    pivoted=pd.pivot_table(
        data,index=['User_id', 'date_received'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"_today_coupon"})
    # print(pivoted)
    data=pd.merge(data,pivoted,on=['User_id', 'date_received'],how='left')

    # 是否当天领取同一券
    pivoted=pd.pivot_table(
        data,index=['User_id', 'date_received', 'Coupon_id'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"_today_this_coupon"})
    # print(pivoted)
    data=pd.merge(data,pivoted,on=['User_id', 'date_received', 'Coupon_id'],how='left')
    # 周几领券
    # 距离是否为空

    # print(data)
    data.drop(['cnt'], axis=1, inplace=True)
    return data

def get_basic_feature(received):
    data=received.copy()

    ### cnt (排除Coupon_id==0的项)
    data.insert(loc=0,column='cnt',value=[int(x) for x in data["Coupon_id"].notnull()])

    name_prifix="feature_basic_" #前缀

    # 1、领券数
    pivoted=pd.pivot_table(
        data,index='User_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    print(pivoted)
    data=pd.merge(data,pivoted,on='User_id',how='left')

    # 2、领券"并消费"数 
    # 3、领券"未消费"数、
    # 4、 领券"并消费"数 / 领券数、
    # 5、领取"并消费"优惠券的平均折扣率、
    pivoted=pd.pivot_table(
        data,index='User_id', values='discount', aggfunc=np.average
            ).reset_index().rename(columns={'cnt': name_prifix+"average_discount"})
    print(pivoted)
    data=pd.merge(data,pivoted,on='User_id',how='left')

    # 6、领取"并消费"优惠券的平均距离、
    pivoted=pd.pivot_table(
        data,index='User_id', values='Distance', aggfunc=np.average
            ).reset_index().rename(columns={'cnt': name_prifix+"average_distance"})
    print(pivoted)
    data=pd.merge(data,pivoted,on='User_id',how='left')

    # 7、在多少不同商家领取"并消费"优惠券、
    # 8、在多少不同商家领取优惠券、

    # 9、在多少不同商家领取"并消费"优惠券 / 在多少不同商家领取优惠券
    data.drop(['cnt'], axis=1, inplace=True)
    return data



# 数据打标
def get_label(dataset):
    total_seconds = 15*24*3600.0
    data=pd.DataFrame(dataset).copy()
    data.insert(0,'label', list(map(lambda x,y: 1 if (x - y).total_seconds()<=total_seconds else 0, data['date'],data['date_received'])))
    # data['cnt']=1
    # print(pd.pivot_table(data,index='label',values='cnt',aggfunc=len))
    return data


def get_dataset(field):
    '''
    构造数据集
    '''
    dataset=pd.DataFrame(field).copy()
    # shared_columns=[]
    dataset=get_feature(dataset)
    # dataset=pd.concat()

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
    dval = xgb.DMatrix(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=validate['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    # 训练
    watchlist = [(dtrain, 'train'),(dval, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 返回
    return result


def main():
    off_train = pd.read_csv("#test\data\ccf_offline_stage1_train.csv")
    off_test = pd.read_csv("#test\data\ccf_offline_stage1_test_revised.csv")

    off_train=preprocess(off_train)
    off_test=preprocess(off_test)
    off_train=get_label(off_train)

    # 数据划分
    # 训练集历史区间、中间区间、标签区间
    train_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
    train_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)

    # 验证集历史区间、中间区间、标签区间
    validate_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    validate_middle_field = off_train[
        off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
    validate_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)

    # 测试集历史区间、中间区间、标签区间
    test_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
    test_label_field = off_test.copy()  # [20160701,20160801)

    get_basic_feature(validate_history_field)
    input()

    train = get_dataset(validate_history_field)
    validate = get_dataset(train_label_field)
    test = get_dataset(test_label_field)

    # 线下验证
    # result_off = model_xgb(train, validate.drop(['label'], axis=1))

    # 线上训练
    big_train = train
    # big_train = pd.concat([train, validate], axis=0)
    result = model_xgb(big_train, validate, test)
    # result.to_csv('E:\\OneDrive - stu.cqupt.edu.cn\\#codes\\data_project\\work\\result')

main()
