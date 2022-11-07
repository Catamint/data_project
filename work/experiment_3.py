import pandas as pd
import os
import pyecharts
from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts.charts import Pie
from pyecharts import options as opts

off_train = pd.read_csv("#test\data\ccf_offline_stage1_train.csv")
# 描述性统计
# 共有多少条记录
record_count = off_train.shape[0]
print('record_count', record_count)
# 共有多少条优惠券的领取记录
received_count = off_train['Date_received'].count()
print('received_count', received_count)
# 共有多少种不同的优惠券
coupon_count = len(off_train['Coupon_id'].value_counts())
print('coupon_count', coupon_count)
# 共有多少个用户
user_count = len(off_train['User_id'].value_counts())
print('user_count', user_count)
# 共有多少个商家
merchant_count = len(off_train['Merchant_id'].value_counts())
print('merchant_count', merchant_count)
# 最早领券时间
min_received = str(int(off_train['Date_received'].min()))
print('min_received', min_received)
# 最晚领券时间
max_received = str(int(off_train['Date_received'].max()))
print('max_received', max_received)
# 最早消费时间
min_date = str(int(off_train['Date'].min()))
print('min_date', min_date)
# 最晚消费时间
max_date = str(int(off_train['Date'].max()))
print('max_date',max_date)


# pyecharts

# 缺失值填充
off_train['Distance'].fillna(-1, downcast='infer', inplace=True)

## 时间格式转换
off_train["date_received"]=pd.to_datetime(
    off_train["Date_received"],format="%Y%m%d")
off_train["date"]=pd.to_datetime(
    off_train["Date"],format="%Y%m%d")

# 满减转换成折扣率
def rate_to_count(rate):
    rate = str(rate)
    a, b = rate.split(sep=":")
    return (int(a)-int(b))/int(a)

### discount = 折扣率
off_train.insert(loc=5, column="discount_rate",
        value=off_train["Discount_rate"].map(lambda x: 
        float(rate_to_count(x)) if ":" in str(x) else float(x)))

# 打标
off_train['label'] = list(map(lambda x, y: 
        1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
        off_train['date'],
        off_train['date_received']))


##########################
# 选取领券日期不为空的数据
df_1 = off_train[off_train['Date_received'].notna()]
# 以Date_received为分组目标并统计优惠券的数量
tmp = df_1.groupby('Date_received', as_index=False)['Coupon_id'].count()


# 消费距离柱状图

# 统计各类距离的消费次数
import collections
dis = off_train[off_train['Distance']!=-1]['Distance'].values
dis = dict(collections.Counter(dis))

x = list(dis.keys())
y = list(dis.values())

# 建立柱状图
bar_distance_used = (
    Bar()
    .add_xaxis()
    .add_yaxis('', y)
    .set_global_opts(
        title_opts=opts.TitleOpts(title='各类距离的消费次数柱状图'), # title
    )
)

# 消费距离与核销率柱状图
rate = [off_train[off_train['Distance']==i]['label'].value_counts()[1]/
off_train[off_train['Distance']==i]['label'].value_counts().sum() for i in range(11)]

bar_distance_used_rate = (
    Bar()
    .add_xaxis(list(range(11)))
    .add_yaxis('核销率', list(rate))
    .set_global_opts(title_opts=opts.TitleOpts(title='消费距离与核销率柱状图'))
    .set_series_opts(
        opts.LabelOpts(is_show=False) # 显示值大小
    )
)


# 各种折扣率的领取和核销数量
# 统计领券日期不为空的数据中各种折扣率的优惠券领取数量
received = off_train[['discount_rate']]
received['cnt'] = 1
received = received.groupby('discount_rate').agg('sum').reset_index()

# 为offline数据添加一个received_month即领券月份
off_train['received_month'] = off_train['date_received'].apply(lambda x: x.month)
# 为offline数据添加一个date_month即消费月份
off_train['date_month'] = off_train['date'].apply(lambda x: x.month)

# 每月各类消费折线图
# 每月核销的数量
consume_coupon = off_train[off_train['label']==1]['received_month'].value_counts(sort=False)
# 每月收到的数量
received = off_train['received_month'].value_counts(sort=False)
# 每月消费数量
consume = off_train['date_month'].value_counts(sort=False)

consume_coupon.sort_index(inplace=True)
received.sort_index(inplace=True)
consume.sort_index(inplace=True)

line_1 = (
    Line()
    .add_xaxis([str(x) for x in range(1, 7)])
    .add_yaxis('核销', list(consume_coupon))
    .add_yaxis('领取', list(received))
    .add_yaxis('消费', list(consume))
    .set_global_opts(title_opts={'text': '每月各类消费折线图'})
    .set_series_opts(
        opts.LabelOpts(is_show=False) # 显示值大小
    )
)

# 添加优惠券是否为满减类型
off_train['is_manjian'] = off_train['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)

# 正负例饼图
v4 = ['正例', '负例']
v5 = list(off_train['label'].value_counts(True))

pie_positive_negative = (
    Pie()
    .add('', [list(v) for v in zip(v4, v5)])
    .set_global_opts(title_opts={'text': '正负例饼图'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}')) # 格式化标签输出内容
)

# 核销优惠券数量占比饼图
v1 = ['折扣', '满减']
v3 = list(off_train[off_train['label']==1].is_manjian.value_counts(True))
pie_receive_used = (
    Pie()
    .add('', [list(v) for v in zip(v1, v3)])
    .set_global_opts(title_opts={'text': '核销优惠券数量占比饼图'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}')) # 格式化标签输出内容
)
path = './output'

if not os.path.exists(path):
        os.makedirs(path)

# 生成HTML文件
bar_distance_used.render(path+'/bar_2.html')
bar_distance_used_rate.render(path+'/bar_3.html')

line_1.render(path+'/line_1.html')

pie_positive_negative.render(path+'/pie_pn.html')
pie_receive_used.render(path+'/pie_ru.html')