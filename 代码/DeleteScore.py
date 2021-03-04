import pandas as pd

dirpath = r'C:\\Users\\1\\Desktop\\基于电商评论的情感分析\\爬取数据\\'
productname = input('请输入产品名称：')

inputfile1 = dirpath + productname + '_负面情感结果.txt'
inputfile2 = dirpath + productname + '_正面情感结果.txt'
outputfile1 = dirpath + productname + '_dc_负面情感结果.txt'
outputfile2 = dirpath + productname + '_dc_正面情感结果.txt'

data1 = pd.read_csv(inputfile1, encoding = 'utf-8', header = None)
data2 = pd.read_csv(inputfile2, encoding = 'utf-8', header = None)

data1 = pd.DataFrame(data1[0].str.replace('.*?\d+?\\t ', '')) 
data2 = pd.DataFrame(data2[0].str.replace('.*?\d+?\\t ', ''))

data1.to_csv(outputfile1, index = False, header = False, encoding = 'utf-8') 
data2.to_csv(outputfile2, index = False, header = False, encoding = 'utf-8')