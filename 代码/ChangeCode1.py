def change_code(input_file,output_file):
    dirpath = r'C:\\Users\\1\\Desktop\\基于电商评论的情感分析\\爬取数据\\'
    with open(dirpath + input_file,'r',encoding='utf-8') as file1:
        with open(dirpath + output_file,'w',encoding='ANSI') as file2:
            for line in file1:
                file2.write(line)

productname = input('请输入产品名称：')
change_code(productname + '_rs.txt', productname + '_rs_ANSI.txt')