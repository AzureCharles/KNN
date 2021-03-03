#-*- coding:utf-8 -*-

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import operator
import csv

LABELS = ['不喜欢','有些喜欢','非常喜欢']
'''
#准备数据，从文本文件中解析数据
'''
def file2matrix(filename):
    #打开文件
    with open(filename,'r') as fr:
        # 读取文件所有内容
        arrayOLines = fr.readlines()
        # 得到文件行数
        numberOfLines = len(arrayOLines)
        # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
        returnMat = np.zeros((numberOfLines, 3))
        # 返回的分类标签向量
        classLabelVector = []
        # 行的索引值
        index = 0
        for line in arrayOLines:
            # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
            line = line.strip()
            # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
            listFromLine = line.split('\t')
            # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
            returnMat[index, :] = listFromLine[0:3]
            # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表有些喜欢,3代表非常喜欢
            if listFromLine[-1] == 'didntLike':
                classLabelVector.append(1)
            elif listFromLine[-1] == 'smallDoses':
                classLabelVector.append(2)
            elif listFromLine[-1] == 'largeDoses':
                classLabelVector.append(3)
            index += 1
    return returnMat, classLabelVector

'''
#分析数据，数据可视化，使用Matplotlib创建散点图
'''
def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    # sans-serif就是无衬线字体，是一种通用字体族。
    # 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, 中文的幼圆、隶书等等
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    #将fig画布分隔成2行2列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,9))

    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[0][0].set_xlabel('每年获得的飞行常客里程数')
    axs0_ylabel_text = axs[0][0].set_ylabel('玩视频游戏所消耗时间占')
    plt.setp(axs0_title_text, size=12, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=10, weight='bold', color='black')


    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title('每年获得的飞行常客里程数与每周消费的冰激淋公升数',)
    axs1_xlabel_text = axs[0][1].set_xlabel('每年获得的飞行常客里程数')
    axs1_ylabel_text = axs[0][1].set_ylabel('每周消费的冰激淋公升数')
    plt.setp(axs1_title_text, size=12, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=10, weight='bold', color='black')


    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title('玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
    axs2_xlabel_text = axs[1][0].set_xlabel('玩视频游戏所消耗时间占比')
    axs2_ylabel_text = axs[1][0].set_ylabel('每周消费的冰激淋公升数')
    plt.setp(axs2_title_text, size=12, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=10, weight='bold', color='black')

    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label=LABELS[0])
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',markersize=6, label=LABELS[1])
    largeDoses = mlines.Line2D([], [], color='red', marker='.',markersize=6, label=LABELS[2])
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

'''
#准备数据，数据归一化处理
'''
def autoNorm(dataSet):
    #获得每列数据的最小值和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    #normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals



'''
KNN算法分类器 
#  inX - 用于分类的数据(测试集)
#  dataSet - 用于训练的数据(训练集)
#  labes - 训练数据的分类标签
#  k - kNN算法参数,选择距离最小的k个点
#  sortedClassCount[0][0] - 分类结果
'''
def classify0(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加,sum(0)列相加,sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方,计算出距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

'''
#测试算法，计算分类器的准确率，验证分类器
'''
def datingClassTest(datingDataMat, datingLabels, saveCsvName, k, hoRatio = 0.10):
    
    #hoRatio = 0.10表示取所有数据的百分之十
    
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获得normMat的行数
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m], k)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        csvWriter([classifierResult, datingLabels[i]], saveCsvName)
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    errorRate = errorCount/float(numTestVecs)*100
    print("错误率:%f%%" %errorRate)
    csvWriter(["错误率:", errorRate], saveCsvName)
    return errorRate

"""
def dataMatMaker(datingDataMat, datingLabels):
    pass
     
    
"""

def classifyPerson(datingDataMat, datingLabels, testDataMat, resultList, k):
    """
    三参数直接自矩阵化的数据集输入，支持用户指定数据输入
    # testDataMat即三维特征用户输入
    # ffMiles = float()#每年获得的飞行常客里程数
    # precentTats = float()#玩视频游戏所耗时间百分比
    # iceCream = float()#每周消费的冰激淋公升数
    """
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array(testDataMat)
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, k)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))

def csvWriter(record_list, csvname, mode="a"):
    #通用写入csv文件操作
    with open(csvname, mode, encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(record_list)

def kIteration(max_k = 5):
    return [k for k in range(1,max_k+1)]  
    #通过指定k值上限制作k值实验组的迭代器

def knnTest(datingDataMat, datingLabels, resultCsv, k_iter, hoRatio, test_times=10):
    #用以批量测试k值对分类的影响
    #不同的k操作同一个测试集
    csvWriter(["实验次数","k=","错误率"], "errorRate.csv", "w")    #创建错误率记录文件头
    for flag in range(test_times):
        min_error = 100.0
        min_error_flag = 0
        min_error_k = 0
        for k in k_iter:
            saveCsvName = resultCsv+str(flag+1)+"_k_"+str(k)+".csv"
            csvWriter(["分类结果","真实类别"], saveCsvName, "w")    #创建记录文件头
            errorRate = datingClassTest(datingDataMat, datingLabels, saveCsvName, k, hoRatio)
            csvWriter([flag+1,k,errorRate], "errorRate.csv")
            if errorRate < min_error:
                min_error = errorRate
                min_error_flag = flag
                min_error_k = k
        
    csvWriter([("实验次数",min_error_flag+1),("k",min_error_k),("最小错误率",min_error)], "errorRate.csv")
'''
#主函数，测试以上各个步骤，并输出各个步骤的结果
'''
if __name__ == '__main__':
    
    filename = "datingTestSet.txt"  #数据集文件名
    testCsvName = "classify_tests.csv"
    resultCsv = "results/classify_result"   #指定测试保存文件名
    #csvWriter(["分类结果","真实类别"], testCsvName, "w")    #创建测试记录表
    max_k = 5                       #指定测试k值的上限
    hoRatio = 0.15                  #指定测试集比例
    
    k_iter = kIteration(max_k)
    datingDataMat, datingLabels = file2matrix(filename)
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    showdatas(datingDataMat, datingLabels)  #数据可视化
    
    #datingClassTest(datingDataMat, datingLabels, testCsvName, 4)   
    #取k=4验证分类器
    knnTest(datingDataMat, datingLabels, resultCsv, k_iter, hoRatio, test_times=10)
    #使用分类器批量测试
    #classifyPerson(datingDataMat, datingLabels, testDataMat, LABELS, k)
