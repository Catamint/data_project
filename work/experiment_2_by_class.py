from operator import truediv
import numpy as np
from matplotlib import pyplot as plt
import random
class k_class:
    def __init__(self, dots=np.random.randn(100, 2), k=2):
        self.dots=list(dots)
        self.k=int(k)
        self.centroids = []
        # self.set_random_centroids()
            # 暂时从dots中随机取k个作为初始centroids
        self.k_means_pp()
        self.clusters=self.reget_clusters()
        self.sse = 0

    def set_random_centroids(self):
        self.centroids = random.sample(self.dots,self.k)
            # 暂时从dots中随机取k个作为初始centroids

    def k_means_pp(self):
        dots=self.dots.copy()
        # print(dots)
        self.centroids=random.sample(self.dots,1)
        dots.remove(self.centroids[-1]) # 把已经选择的质心从dots中剔除
        # self.draw(dots=True,clusters=False)
        for i in range(self.k - 1):
            min_list=[]
            for dot in dots:
                min_distance = 10000000
                for centroid in self.centroids:
                    distance = pow(centroid[0]-dot[0],2) + \
                        pow(centroid[1]-dot[1],2)
                    if distance < min_distance:
                        min_distance = distance
                min_list.append([dot,min_distance])
            # print(sorted(min_list,key=lambda x: x[-1])[-1])
            self.centroids.append \
                (sorted(min_list,key=lambda x: x[-1])[-1][0])
            dots.remove(self.centroids[-1])
            # self.draw(dots=True,clusters=False)


    def reget_clusters(self):
        '''
        (取遍dots) 找新的距离最近的c.
        '''
        centroids=self.centroids
        k = self.k
        clusters = dict()
        
        for dot in self.dots:
            min_distance = 10000000
            for i in range(k):
                distance = pow(centroids[i][0]-dot[0],2) + \
                    pow(centroids[i][1]-dot[1],2)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_index = i
            if min_distance_index not in clusters:
                clusters[min_distance_index] = []
            clusters[min_distance_index].append(dot)
        return clusters

    def reget_centroids(self):
        '''
        根据新的c{x}更新c的位置
            centroid[i]=average(clusters[i]), 由SSE决定.
        '''
        centroids = []
        for i in self.clusters.values():
            centroids.append(np.average(i, axis=0))
        return centroids

    def reget_sse(self):
        '''
        目标函数 (判断条件)
            问题: 如果有的key中没有点,则dict中无此key,会越界.
            如:
            dots=[[1,10],[20,2],[20,3],[2,11]]
            cent=[[1,2],[2,3]]
            一种解决办法是把centroids初始值放在k个dot处,保证k个簇都有点.
            已解决,暂时从dots中随机取k个作为初始centroids.
        '''
        centroids = list(self.centroids)
        sse = 0.0
        for i in range(self.k):
            for dot in self.clusters[i]:
                sse += pow(centroids[i][0]-dot[0],2)+ \
                    pow(centroids[i][1]-dot[1],2) # 无需开方
        return sse

    def reget_sse_means(self):
        '''
        SSE/len(dots)
        '''
        sse_means = self.reget_sse()/len(self.dots)
        return sse_means

    def draw(self,dots=False,clusters=True,centroids=True):
        # only 2D
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        board = plt.figure()
        ax = board.add_axes([0.1, 0.1, 0.8, 0.8])

        # 是否绘制dots
        if dots == True:
            dots = self.dots
            xy_list = np.array(dots).transpose()
            ax.scatter(xy_list[0], xy_list[1], s=10, c='b')
        # 是否绘制clusters
        if clusters == True:
            clusters = self.clusters
            for i in clusters.keys():
                xy_list = np.array(clusters[i]).transpose()
                ax.scatter \
                    (xy_list[0], xy_list[1], s=10, c=colors[i])
        # 是否绘制centroids
        if centroids == True:
            centroids=np.array(self.centroids)
            centroids_xy = centroids.transpose()
            ax.scatter(centroids_xy[0], centroids_xy[1], 
                s=200, c='black', marker='+')
        board.show()

    def reset(self, dots=None, k=None):
        '''
        重新设置dots和k值
        '''
        if dots != None:
            self.dots = list(dots)
        if k != None:
            self.k = int(k)
        self.set_random_centroids()
            #暂时从dots中随机取k个作为初始centroids
            #可能会取到重复的点
        self.clusters=self.reget_clusters()
        self.sse = 0

    def k_means(self, silent=0):
        '''
        K-means
        '''
        self.sse = 0
        sse_last = 10000
        while(abs(self.sse - sse_last) > 0.01):
            # 注意改结束取值
            sse_last = self.sse
            self.clusters = self. \
                reget_clusters()
            self.centroids = self. \
                reget_centroids()
            if len(self.centroids)!=self.k:
                # 避免出现有的质心没有获得点的情况
                self.reset()
                sse_last = -10000
                print('restart')
                continue
            self.sse = self.reget_sse()
            if silent==1:
                print(self.sse)

def import_dots():
    return np.random.randn(400, 2).tolist()

def import_file():
    file_1=open("work\iris.data")
    return [list(map(float,[x.split(',')[2],x.split(',')[3]])) 
        for x in file_1.readlines()[:-1]]

def arm():
    sse_by_k=[np.nan,np.nan]
    k_obj=k_class(dots=import_file(), k=2)
    for i in range(2,8):
        k_obj.reset(k=i)
        k_obj.k_means(silent=1)
        sse_by_k.append(k_obj.sse)
    plt.plot(sse_by_k)
    plt.show()
    # k_obj.draw()

k_obj=k_class(dots=import_dots(), k=3)
# k_obj.reset(k=6)
k_obj.k_means(silent=1)

k_obj.draw()

input()
# k_means(import_dots(), k=3)


'''
def test(self):
    dots = [1.5, 2.1, 2.2, 2.5, 5]
    cent = [1, 2]
    for i in range(10):
        print(i)
        a = get_clusters(dots, cent)
        print('clusters', a)
        cent = reset_centroids(a)
        print('centtroids', cent)
        print('sse ', get_sse(cent, a))

def test_2d(self):
    dots = np.random.randn(400, 4)
    cent = dots[:6]
    for i in range(20):
        print(i)
        a = get_clusters(dots, cent)
        # print('clusters', a)
        cent = reset_centroids(a)
        print('centtroids', cent)
        print('sse ', get_sse(cent, a))
'''
