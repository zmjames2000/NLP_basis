# encoding: utf-8
from grammer.recursionSearch import *
from grammer.stanfordParse import *

def _replace_c(text):
    """
    将英文标点符号替换成中文标点符号，并去除html语言的一些标志等噪音
    :param text:
    :return:
    """
    intab = ",?!()"  # 句子中有的英文字符
    outtab = "，？！（）"  # 替换成中文的字符
    deltab = " \n<li>< li>+_-.><li \U0010fc01 _" # 需要删除的内容
    trantab=text.maketrans(intab, outtab, deltab)
    return text.translate(trantab)

def parse_sentence(text):
    text = _replace_c(text)  # 文本去噪
    # print(text)
    try:
        if len(text.strip())>6:
            return Tree.fromstring(nlp.parse(text.strip()))
    except:
        pass


if __name__ == '__main__':
    with open('./data/demo2_TextRank/dependency.txt', 'w',encoding='utf-8') as fout,\
         open('./data/demo2_TextRank/text.txt','r', encoding='utf-8') as itera:
        for it in itera:
            # print(it)
            s = parse_sentence(it)  # 得到一个tree
            print('s tree:{}'.format(s))
            # s.draw()
            res = search(s)