import pandas as pd
import os
import hashlib
import operator

def  judge(l1,l2): 
    if len(l1) != len(l2):
        return False
    else:
        return operator.eq(l1,l2)

def gen_md5(data):
    md5 = hashlib.md5()
    md5.update(data.encode('utf-8'))
    return md5.hexdigest()

def remove_same(input_file,output_file):
    fps = set()
    dirpath = r'C:\\Users\\1\\Desktop\\基于电商评论的情感分析\\爬取数据\\'
    with open(dirpath + input_file,'r',encoding='utf-8') as file1:
        with open(dirpath + output_file,'w',encoding='utf-8') as file2:
            for line in file1:
                string = line.strip()
                length = len(string)
                fp = gen_md5(string)
                if fp not in fps:
                    if length>=6:
                        fps.add(fp)
                        file2.write(line)

def remove_signal(string):
    string = string.replace('\n','')
    string = string.replace('space','')
    string = string.replace('！','')
    string = string.replace('。','')
    string = string.replace('，','')
    string = string.replace('?','')
    string = string.replace('~','')
    string = string.replace('（','')
    string = string.replace('）','')
    string = string.replace('-','')
    string = string.replace('*','')
    string = string.replace('+','')
    return string

if __name__ == "__main__":
    productname = input('请输入产品名称：')
    remove_same(productname + '.txt',productname + '_rs.txt')