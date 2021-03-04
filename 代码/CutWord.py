import jieba
import os

list1 = ['了','呢','的','也','就','很','只要','才','是','又']

def seg_list(sentence):
    sentence1 = jieba.cut(sentence.strip())
    stopwords = list1
    outstr = ''
    for word in sentence1:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ' '
    return outstr

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
    string = string.replace('：','')
    string = string.replace('、','')
    string = string.replace('：','')
    string = string.replace('；','')
    string = string.replace('+','')
    string = string.replace('@','')
    string = string.replace('.','')
    return string

def remove_word(input_file,output_file):
    dirpath = os.getcwd() + '\\'
    with open(dirpath + input_file,'r',encoding='utf-8') as file1:
        with open(dirpath + output_file,'w',encoding='utf-8') as file2:
            for line in file1:
                string = remove_signal(line)
                seg = seg_list(string)
                file2.write(seg)

if __name__ == "__main__":
    remove_word('infor_rs.txt','infor_new.txt')