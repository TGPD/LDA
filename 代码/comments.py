    import requests
import csv
import time
import json
import random
from lxml import etree
from bs4 import BeautifulSoup

header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
url = 'https://search.jd.com/Search?'

params = {
    'keyword':'笔记本电脑',
    'enc':'utf-8',
    'wq':'笔记本电脑',
    'pvid':'537840179fcc4a08a9164fe24591afdf'
}

res1 = requests.get(url,headers=header,params=params)
selector = etree.HTML(res1.text)
product_list = selector1.xpath('/html/body/div[6]/div[2]/div[2]/div[1]/div/div[2]/ul/li')

for product in product_list:
    p_id = product.xpath('@data-sku')[0]
    detail_url = 'https://item.jd.com/' + p_id + '.html'
    detail_res = requests.get(detail_url, headers=header)
    detail_selector = etree.HTML(detail_res.text)

    with open(r'D:\Desktop\poket\Python\result.txt','a',encoding='utf-8') as txt:
        brand_name= detail_selector.xpath('//*[@id="parameter-brand"]/li/a/text()')[0]
        pro_name = detail_selector.xpath('//*[@id="detail"]/div[2]/div[1]/div[1]/ul[2]/li[1]/text()')[0]
        print('爬取{}的评论'.format(pro_name))

        pro_name = pro_name[5:]

        txt.write(pro_name)
        txt.write('\r\n')
        txt.write(brand_name)
        txt.write('\r\n')

        for i in range(20):
            print('爬取第{}页评论'.format(i))
            comment_url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=' + str(p_id) + '&score=0&sortType=5&page=' + str(i) + '&pageSize=10&isShadowSku=0&fold=1'
            res_comment = requests.get(comment_url,headers=header)
            time.sleep(random.randint(4,6))
            res_comment_text = res_comment.text.replace('fetchJSON_comment98(','').replace(');','')
            comments = json.loads(res_comment_text)['comments']
            for comment in comments:
                commentdata = comment['content']
                txt.write(commentdata)

        txt.write('\r\n')
