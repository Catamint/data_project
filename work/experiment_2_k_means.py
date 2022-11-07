import numpy as np
from matplotlib import pyplot as plt


def get_clusters(dots, centroids):
    '''
    (取遍dots) 找新的距离最近的c.
    '''
    k = len(centroids)
    clusters = dict()
    for dot in dots:
        min_distance = 10000000
        for i in range(k):
            vector = np.subtract(dot, centroids[i])
            distance = abs(np.dot(vector, vector))  # 无需开方
            if distance < min_distance:
                min_distance = distance
                min_distance_index = i
        if min_distance_index not in clusters:
            clusters[min_distance_index] = []
        clusters[min_distance_index].append(dot)
    return clusters


def reset_centroids(clusters):
    '''
    根据新的c{x}更新c的位置
        centroid[i]=average(clusters[i]), 由SSE决定.
    '''
    centroids = []
    for i in clusters.values():
        centroids.append(np.average(i, axis=0))
    return centroids


def get_sse(centroids, clusters):
    '''
    目标函数 (判断条件)
        问题: 如果有的key中没有点,则dict中无此key,会越界.
        如:
        dots=[[1,10],[20,2],[20,3],[2,11]]
        cent=[[1,2],[2,3]]
        一种解决办法是把centroids初始值放在k个dot处,保证k个簇都有点.
        问题是哪k个dot.
        已解决.
    '''
    centroids = np.array(centroids)
    sse = 0.0
    for i in range(len(centroids)):
        for dot in clusters[i]:
            vector = np.subtract(dot, centroids[i])
            sse += abs(np.dot(vector, vector))  # 无需开方
    return sse


def get_sse_means(centroids, clusters,dots):
    '''
    SSE/len(dots)
    '''
    sse_means = get_sse(centroids,clusters)/len(centroids)
    return sse_means

def draw(clusters, centroids):
    # only 2D
    board = plt.figure()
    ax = board.add_axes([0.1, 0.1, 0.8, 0.8])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in clusters.keys():
        xy_list = np.array(clusters[i]).transpose()
        ax.scatter(xy_list[0], xy_list[1], s=10, c=colors[i])
        
    centroids_xy = np.array(centroids).transpose()
    ax.scatter(centroids_xy[0], centroids_xy[1], s=200, c='black', marker='+')
    board.show()


def k_means(dots, k):
    '''
    K-means
        暂时从dots中随机取k个作为初始centroids
    '''
    sse = 10000
    sse_last = 0
    # centroids = dots[:k]  # 暂时取前k个dots作为初始的centroids
    centroids = list(map(lambda x: 
        dots[np.random.randint(len(dots))], range(k))) #改为随机取k个dots
    while(abs(sse-sse_last) > 0.01):
        # 注意改结束取值
        sse_last = sse
        clusters = get_clusters(dots, centroids)
        centroids = reset_centroids(clusters)
        sse = get_sse(centroids, clusters)
        print(sse)

    # sse_means=get_sse_means(centroids,clusters,dots)
    # print(sse_means)
    draw(clusters, centroids)
    # print(clusters)
    input()


def import_dots():
    dots = np.random.randn(40000, 2)
    return dots

def import_file():
    file_1=open("work\iris.data")
    a=[list(map(float,x.split(',')[:2])) for x in file_1.readlines()[:-1]]
    # print(a)
    return a


k_means(import_file(), k=6)
# k_means(import_dots(), k=3)   


def test():
    dots = [1.5, 2.1, 2.2, 2.5, 5]
    cent = [1, 2]
    for i in range(10):
        print(i)
        a = get_clusters(dots, cent)
        print('clusters', a)
        cent = reset_centroids(a)
        print('centtroids', cent)
        print('sse ', get_sse(cent, a))


def test_2d():
    dots = np.random.randn(400, 4)
    cent = dots[:6]
    for i in range(20):
        print(i)
        a = get_clusters(dots, cent)
        # print('clusters', a)
        cent = reset_centroids(a)
        print('centtroids', cent)
        print('sse ', get_sse(cent, a))
