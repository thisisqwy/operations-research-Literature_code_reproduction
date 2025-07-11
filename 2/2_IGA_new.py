import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class IGA():
    def __init__(self,num_pop,num_city,iteration,p_c,p_m,data):
        self.num_pop = num_pop
        self.num_city = num_city
        self.iteration = iteration
        self.p_c = p_c
        self.p_m = p_m
        self.location=data
        self.dis_mat=self.compute_dismat(num_city)
        self.pop=self.init_pop(num_pop,num_city)
        self.num_parents=int(p_c*num_pop)#这里必须加int，不然得到的是浮点数，后面sort_index = np.argsort(-fitness_select).copy()[0:self.num_parents]会报错
    #根据二维坐标计算距离矩阵
    def compute_dismat(self,num_city):
        matrix=np.zeros((num_city,num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i==j:
                    matrix[i,j]=np.inf
                else:
                    matrix[i,j]=math.sqrt((self.location[i][0]-self.location[j][0])**2+(self.location[i][1]-self.location[j][1])**2)
        return matrix.copy()
    #种群初始化，这里采用随机生成，而不是最近邻算法
    def init_pop(self,num_pop,num_city):
        pop=[]
        for i in range(num_pop):
            pop.append(random.sample(range(num_city),num_city))
        return pop.copy()
    #计算单条路径的长度
    def path_length(self,path):
        sum=self.dis_mat[path[-1]][path[0]]
        for i in range(len(path)-1):
            sum=sum+self.dis_mat[path[i]][path[i+1]]
        return sum
    #计算种群适应度函数
    def compute_fitness(self,fruits,dis_mat):
        score=[]
        for fruit in fruits:
            length = self.path_length(fruit)
            score.append(1.0 / length)
        return np.array(score)
    #轮盘赌选择两个父代
    def select_wheel(self,score,pop):
        sum_score=sum(score)
        p=[i/sum_score for i in score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(p):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(pop[index1]), list(pop[index2])
    #选择算子
    def select(self,pop,fitness):
        genes=pop.copy()#由于后面要删除最优的那个，所以先进行一个拷贝。
        score=fitness.copy()
        best_index=np.argsort(-score).copy()[0]
        best_one=genes[best_index]
        del genes[best_index]
        score=np.delete(score, best_index)
        new_pop=[best_one]
        sum_score=sum(score)
        p=[i/sum_score for i in score]
        while len(new_pop)<self.num_pop:#通过轮盘赌的方式对删除best后的genes进行选择算子。
            r=np.random.rand()
            for i,sub in enumerate(p):
                if r>=0:
                    r-=sub
                    if r<0:
                        index=i
                if r<0:
                    break
            new_pop.append(genes[index])
        return new_pop.copy()
    #父代交叉
    def cv(self,path1,path2):
        #先随机选择一个起点城市为c
        c=random.randint(0, len(path1) - 1)
        childx=[c]
        childy=[c]
        #先生成childx
        current = c
        X=path1.copy()
        Y=path2.copy()
        while len(childx)<self.num_city:
            idx_x = X.index(current)#index_x是指current在X中的索引，index_y同理
            Cx = X[(idx_x + 1) % len(X)]
            idx_y = Y.index(current)
            Cy = Y[(idx_y + 1) %len(Y)]
            dx=self.dis_mat[current,Cx]#当前点到下一个点的距离
            dy=self.dis_mat[current,Cy]
            # 选择较近的那个城市作为下一个
            if dx <= dy:
                next_city = Cx
            else:
                next_city = Cy
            # 加入next_city到childx中去，并从X和Y删除current，但是next_city还要保留
            childx.append(next_city)
            X.remove(current)
            Y.remove(current)
            # 更新当前城市
            current=next_city
        #后生成childy
        current=c
        X=path1.copy()
        Y=path2.copy()
        while len(childy)<self.num_city:
            idx_x = X.index(current)#index_x是指current在X中的索引，index_y同理
            Cx = X[idx_x-1]
            idx_y = Y.index(current)
            Cy = Y[idx_y-1]
            dx=self.dis_mat[Cx,current]#当前点到下一个点的距离
            dy=self.dis_mat[Cy,current]
            # 选择较近的那个城市作为下一个
            if dx <= dy:
                next_city = Cx
            else:
                next_city = Cy
            # 加入next_city到childy中去，并从X和Y删除current，但是next_city还要保留
            childy.append(next_city)
            X.remove(current)
            Y.remove(current)
            # 更新当前城市
            current=next_city
        return childx,childy
    #变异
    def mutate(self,path):
        if np.random.random()<self.p_m:
            start = random.randint(0, len(path) - 2)
            end = random.randint(start, len(path) - 1)
            new_gene = path.copy()
            new_gene[start:end + 1] = new_gene[start:end + 1][::-1]
            child=new_gene
        else:
            child=path.copy()
        return child
    #主函数运行
    def main(self):
        fitness=self.compute_fitness(self.pop,self.dis_mat)
        sort_index = np.argsort(-fitness).copy()#如果fitness不是数组，那么np.argsort就无效。
        best_path = self.pop[sort_index[0]].copy()
        best_fitness =fitness[sort_index[0]]
        print('当前为第0代，最短路径长度为：%.2f' % (1/best_fitness))
        for i in range(1,self.iteration+1):
            new_pop_select=self.select(self.pop,fitness)
            fitness_select=self.compute_fitness(new_pop_select,self.dis_mat)#计算交叉过后的种群适应度
            sort_index = np.argsort(-fitness_select).copy()[0:self.num_parents]
            parents=[]
            for index in sort_index:
                parents.append(new_pop_select[index])#取适应度最高的前27个个体作为父本集以供等下配对
            #开始交叉操作形成新种群
            new_pop_cross=[]
            while len(new_pop_cross)<self.num_pop:
                x, y =parents[np.random.randint(len(parents))],parents[np.random.randint(len(parents))]#随机从parents中选取一对父本x和y
                childx, childy = self.cv(x, y)
                childx_adp = 1. / self.path_length(childx)
                childy_adp = 1. / self.path_length(childy)
                if childx_adp > childy_adp and (not childx in parents):
                    child = childx.copy()
                elif childx_adp > childy_adp and (not childy in parents):
                    child = childy.copy()
                else:
                    continue
                new_pop_cross.append(child)
            #交叉完后进行变异
            new_pop=[]
            for path in new_pop_cross:
                if np.random.random()<self.p_m:
                    new_pop.append(self.mutate(path))
                else:
                    new_pop.append(path)
            self.pop = new_pop.copy()
            fitness=self.compute_fitness(self.pop,self.dis_mat)
            sort_index = np.argsort(-fitness).copy()
            tmp_best_path = self.pop[sort_index[0]].copy()
            tmp_best_fitness =fitness[sort_index[0]]
            if tmp_best_fitness > best_fitness:
                best_fitness=tmp_best_fitness
                best_path=tmp_best_path
            print(f"当前为第{i}代，最短路径长度为：{1/best_fitness:.2f}")
        print(f"最终的最优路径为:{best_path}")
        print(f"最短路径长度为:{1/best_fitness:.2f}")
coordinates = [
    [10, 75], [36, 9], [91, 78], [54, 53], [8, 51],
    [78, 51], [73, 14], [23, 56], [15, 40], [26, 60],
    [74, 98], [91, 16], [92, 75], [63, 73], [95, 16],
    [21, 84], [4, 23], [41, 29], [36, 74], [56, 15],
    [14, 18], [26, 80], [49, 13], [24, 94], [5, 92],
    [66, 43], [48, 26], [13, 33], [11, 53], [47, 65],
    [15, 70], [37, 91], [63, 54], [46, 42], [66, 8],
    [52, 28], [61, 34], [70, 13], [91, 55], [36, 60],
    [20, 20], [39, 2], [29, 90], [87, 46], [62, 87],
    [70, 92], [19, 74], [32, 10], [81, 84], [73, 90]
]
data=np.array(coordinates)
model = IGA(num_pop=30,num_city=data.shape[0], iteration=500, p_c=0.9,p_m=0.01,data=data.copy())
model.main()