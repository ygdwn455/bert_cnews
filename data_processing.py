import os
import numpy as np
import pandas as pd
import jieba
import time



Contradictory_keys = {'经济':['经济','经济增长','总需求','GDP','经济运行','内需','外需','可持续性','经济指标','总体经济','增长','内生',
                            '居民收入','宏观政策','经济体','动能','内生性','马车','财政政策','韧性','经济趋势'],
                    '流动':['流动性','资金面','资金紧张','资金','货币','钱荒','利率','对冲','公开市场','时点','降准','货币政策','压力',
                           'MLF','市场','准备金','中性','正回购','储率','超储率','资金量'],
                    '通胀':['通胀','物价','CPI','物价水平','通胀率','价格水平','货币政策','PCE','PPI','总需求','降息',
                          '商品价格','预期','斜率','锚定','薪资','失业率','能源价格','QE','触顶','名义工资']}
Contradictory_inds = ['经济','流动','通胀']



# set_jinji = set(Contradictory_keys['经济'])
# set_liudong = set(Contradictory_keys['流动'])
# set_tongzhang = set(Contradictory_keys['通胀'])

def Identify_contradiction(text):
    seg_list = jieba.cut(text)

    # word_lst = [word for word in seg_list]
    # list01_num = len(list(set(word_lst).intersection(set_jinji)))
    # list02_num = len(list(set(word_lst).intersection(set_liudong)))
    # list03_num = len(list(set(word_lst).intersection(set_tongzhang)))

    list01_num = 0
    list02_num = 0
    list03_num = 0
    for word in seg_list:
        if word in Contradictory_keys['经济']:
            list01_num += 1
        elif word in Contradictory_keys['流动']:
            list02_num += 1
        elif word in Contradictory_keys['通胀']:
            list03_num += 1
    tmp = np.array([list01_num,list02_num,list03_num])
    return Contradictory_inds[tmp.argmax()]

news_info = pd.read_csv('news_lx/news_info.csv',iterator=True, chunksize=65535)

# 第一种读取所有的chunk块并将所有块拼接成一个DataFrame
chunk_total_len = 0
for chunk in news_info:

    t1 = time.time()
    chunk_len = len(chunk)
    part01 = int(chunk_len*0.7)
    part02 = int(chunk_len*0.9)
    chunk_total_len += chunk_len

    chunk['contra'] = chunk.newsSummary.map(lambda x: str(Identify_contradiction(x) + '\t' + x))
    text_lst = chunk['contra'].to_list()

    cnews_train = "\n".join(text_lst[:part01])
    with open('cnews/cnews.train.txt', 'a', encoding='utf8') as f:
        f.write(cnews_train)

    cnews_test = "\n".join(text_lst[part01:part02])
    with open('cnews/cnews.test.txt', 'a', encoding='utf8') as f:
        f.write(cnews_test)

    cnews_val = "\n".join(text_lst[part02:])
    with open('cnews/cnews.val.txt', 'a', encoding='utf8') as f:
        f.write(cnews_val)
    print(chunk_len)
    print(time.time()-t1)


print(chunk_total_len)







