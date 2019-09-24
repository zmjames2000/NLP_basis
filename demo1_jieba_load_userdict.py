#encoding=utf-8
import jieba
import re
from pyhanlp import *
from jpype import *
# root_path = r'D:\Anaconda3\Lib\site-packages\pyhanlp\static'
# djclass_path = r'-Djava.class.path='+root_path+os.sep+'hanlp-1.7.4.jar;'+root_path
# startJVM(getDefaultJVMPath(),djclass_path,'-Xmslg','-Xmxlg')

dict_path = r'./data/dict.txt'
jieba.load_userdict(dict_path)

dict_fp = open(dict_path,'r',encoding='utf-8')
d = {}
[d.update({line: len(line.split(' ')[0])}) for line in dict_fp]
print(d.items()) #dict_items([('台中\n', 3), ('台中正确', 4)])
f = sorted(d.items(), key=lambda x:x[1], reverse=True) #key是一个函数，key=len按照长度排序
dict_fp.close()

new_dict = open('./data/dict1.txt','w',encoding='utf-8')
[new_dict.write(item[0]+'\n') for item in f]
new_dict.close()

[ jieba.suggest_freq(line.strip(),tune=True) for line in open('./data/dict1.txt','r',encoding='utf-8')]

if __name__ == '__main__':
    string = '台中正确应该不会被切开'
    words = jieba.cut(string, HMM=False)
    result = ' '.join(words)
    print(result)



