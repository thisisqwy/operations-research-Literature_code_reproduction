import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class GA(object):
    #初始化函数，只要创建了这个类的实例就一定会运行。
    def __init__(self, num_city, num_total, iteration, data,k1,k2,k3,k4):#各参数分别是点的数量，种群个体数量，最大迭代次数，每点的二维坐标，自适应交叉与变异概率的4个参数。
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        self.iteration = iteration
        self.location = data#将data赋给self.location是为了后续绘图。
        self.ga_elitism_ratio = 0.5
        self.dis_mat = self.compute_dis_mat(num_city, data)#根据已有的横轴坐标数据和城市数量去计算各点间的欧式距离。
        self.fruits = self.greedy_init(self.dis_mat,int(num_total/2),num_city)#初始化种群由贪婪算法和随机生成各一半。本代码中的种群fruits用一个大列表表示，其中每个个体也是一个列表。
        self.fruits.extend(self.random_init(int(num_total/2), num_city))
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.k4=k4
        self.gen=0
    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue#即跳出当前循环，如果当前是j，那么就进入j+1
                a = location[i] #a和b此时都是一个维度为2的一维数组。
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    def greedy_init(self, dis_mat, num_total, num_city):
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            start_index=np.random.randint(0, num_city)
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # 计算路径长度
    def compute_pathlen(self, path, dis_mat):
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb
                pdb.set_trace()
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def ga_cross(self, x, y,cross_ratio):
        len_ = len(x)
        if np.random.random()<cross_ratio:
            start = random.randint(0, len_ - 2)
            end = random.randint(start + 1, len_ - 1)
            child1=y[start:end+1]#list切片即使是0：len，也等价于copy
            child2=[]#child2的结尾等于p1的切片
            for i in x:
                if i not in child1:
                    child1.append(i)
            for j in y:
                if j not in x[start:end+1]:
                    child2.append(j)
            child2.extend(x[start:end+1])
        else:
            child1=x.copy()
            child2=y.copy()
        return list(child1), list(child2)

    def ga_elitism(self, scores, ga_elitism_ratio):
        sort_index = np.argsort(-scores).copy()#得到适应度从高到低排序后的索引。
        sort_index = sort_index[0:int(ga_elitism_ratio * len(sort_index))]
        elitism = []
        elitism_score = []
        for index in sort_index:
            elitism.append(self.fruits[index])
            elitism_score.append(scores[index])
        return elitism, elitism_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
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
        return list(genes_choose[index1]), list(genes_choose[index2]) #这里return后面必须加list或者在返回项后加.copy，不然种群会退化

    def ga_mutate(self, gene):
        start = random.randint(0, len(gene) - 2)
        end = random.randint(start + 1, len(gene) - 1)
        child=gene.copy()
        child[start]=gene[end]
        child[end]=gene[start]
        return child.copy()
    def ga_Metropolis(self,gene):
        start=random.randint(0, len(gene) - 2)
        end=random.randint(start, len(gene) - 1)
        interval=gene[start:end+1]
        random.shuffle(interval)
        new_gene=gene.copy()
        new_gene[start:end + 1] = interval
        past_len=self.compute_pathlen(gene, self.dis_mat)
        new_len = self.compute_pathlen(new_gene, self.dis_mat)
        if new_len<past_len:
            child=new_gene.copy()
        else:
            p1=math.exp(-((new_len-past_len)/(self.iteration-0.9*self.gen)))#gen_max为算法最大迭代次数,gen为算法当前迭代次数
            p=np.random.random()
            if p<p1:
                child=new_gene.copy()
            else:
                child=gene.copy()
        return child.copy()
    def ga_reverse(self,gene):
        start = random.randint(0, len(gene) - 2)
        end = random.randint(start, len(gene) - 1)
        new_gene = gene.copy()
        new_gene[start:end + 1] = new_gene[start:end + 1][::-1]
        if self.compute_pathlen(new_gene, self.dis_mat) < self.compute_pathlen(gene, self.dis_mat):
            child=new_gene.copy()
        else:
            child=gene.copy()
        return child.copy()
    def ga(self):
        # 获得优质父代
        scores = self.compute_adp(self.fruits)#计算当前种群的适应度
        f_max=max(scores)#分别计算当前种群适应度的最大值和平均数
        f_avg=sum(scores)/len(scores)
        # 选择部分优秀个体作为精英直接保留到下一代
        elitism, elitism_score = self.ga_elitism(scores, self.ga_elitism_ratio)#这里本质是精英选择机制。
        # 新的种群fruits
        fruits = elitism.copy()
        # 生成新的种群
        while len(fruits) < self.num_total:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(elitism_score, elitism)
            #计算自适应的交叉概率
            f_cross=max(1./self.compute_pathlen(gene_x, self.dis_mat),1./self.compute_pathlen(gene_y, self.dis_mat))#计算两个父代中大的适应度
            if f_cross<f_avg:
                cross_ratio=self.k3
            else:
                cross_ratio=self.k1*math.sin((f_max-f_cross)/(f_max-f_avg))
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y,cross_ratio)
            #对child变异
            f_x=1./self.compute_pathlen(gene_x_new, self.dis_mat)
            if f_x<f_avg:
                mutate_ratio=self.k4
            else:
                mutate_ratio=self.k2*math.sin((f_max-f_x)/(f_max-f_avg))
            if np.random.rand() < mutate_ratio:
                child_x = self.ga_mutate(gene_x_new)
            else:
                child_x=gene_x_new
            f_y=1./self.compute_pathlen(gene_y_new, self.dis_mat)
            if f_y<f_avg:
                mutate_ratio=self.k4
            else:
                mutate_ratio=self.k2*math.sin((f_max-f_y)/(f_max-f_avg))
            if np.random.rand() < mutate_ratio:
                child_y = self.ga_mutate(gene_y_new)
            else:
                child_y=gene_y_new
            child_x_adp = 1. / self.compute_pathlen(child_x, self.dis_mat)
            child_y_adp = 1. / self.compute_pathlen(child_y, self.dis_mat)
            if child_x_adp > child_y_adp  and (not child_x in elitism):
                child=child_x.copy()
            elif child_x_adp <= child_y_adp and (not child_y in elitism):
                child=child_y.copy()
            else:
                continue
            #Metropolis法则
            child=self.ga_Metropolis(child)
            #reverse
            child=self.ga_reverse(child)
            fruits.append(child)
        # new_pop=[]
        # for fruit in fruits:
        #     gene=self.ga_Metropolis(fruit)
        #     gene=self.ga_reverse(gene)
        #     new_pop.append(gene)
        self.fruits = fruits.copy()
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores).copy()
        tmp_best_one = self.fruits[sort_index[0]].copy()
        tmp_best_score = scores[sort_index[0]]
        return tmp_best_one, tmp_best_score

    def run(self):
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        BEST_LIST = self.fruits[sort_index[0]]
        print(type(scores))
        best_score =scores[sort_index[0]]
        self.best_record =[]
        self.best_record.append(1./best_score)
        print(0,1./best_score)
        for i in range(1, self.iteration + 1):
            self.gen+=1
            tmp_best_one, tmp_best_score = self.ga()
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1./best_score)
            print(i,1./best_score)
        return self.location[BEST_LIST], 1. / best_score#返回的location[BEST_LIST]是一个二维数组，关于各点的横纵坐标。例如[[ 80.  39.][ 81.  34.]]

# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data
data = read_tsp(r'D:\Users\qwy\Desktop\tsp算例\st70.tsp\st70.tsp')
data = np.array(data)#为了方便后续切片，所以要转为数组
data = data[:, 1:]#切片，只需后两位数字即横轴坐标,此时的data是一个二维数组，其维度是(70,2)
Best, Best_path = math.inf, None

model = GA(num_city=data.shape[0], num_total=100, iteration=500, data=data.copy(),k1=1,k2=0.5,k3=1,k4=0.5)#根据类GA创建对象model,位置实参必须放在关键字实参之前,所以如果不写k1=1，那么k1的数字得卸载num_city前面。
path, path_len = model.run()
#print(path)
if path_len < Best:
    Best = path_len
    Best_path = path
# 指定默认字体为黑体（Windows 上一般都有）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 让负号能正常显示
matplotlib.rcParams['axes.unicode_minus'] = False
# 加上一行因为会回到起点
# fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
# axs[0].scatter(Best_path[:, 0], Best_path[:,1])
# Best_path = np.vstack([Best_path, Best_path[0]])
# axs[0].plot(Best_path[:, 0], Best_path[:, 1])
# axs[0].set_title('规划结果')
# iterations = range(model.iteration)
# best_record = model.best_record
# axs[1].plot(iterations, best_record)
# axs[1].set_title('收敛曲线')
# plt.show()

