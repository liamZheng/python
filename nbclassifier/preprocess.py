__author__ = 'LiamZheng'

import sys
import fileinput
import glob
import os
import jieba

reload(sys)
sys.setdefaultencoding('utf-8')


def extractTitle(inPath, outPath):
    "extract title from texts and put all into one file"
    labels = os.listdir(inPath)
    outFile = open(outPath, 'w')
    num = {}
    for label in labels:
        num[label] = 0
        for line in fileinput.input(glob.glob(inPath+"/"+label+"/*")):
            if not line.startswith('http://'):
                line_uni = unicode(line, 'gbk', 'ignore')
                 #  tokenize
                tokens = " ".join(jieba.cut(line_uni))
                outFile.write( "%s\t%s" % (label, tokens))
                num[label]+=1
                fileinput.nextfile()
        fileinput.close()
    outFile.close()
    print "Extract Files"
    for label, n in num.iteritems():
        print "%-10s : %d" % (label, n)


def partition(inPath, outPath, lim_train=sys.maxint, lim_test=sys.maxint):
    '''partition the data into train and test samples'''
    data = open(inPath, "r")
    train = open(os.path.join(outPath, "train"), "w")
    test = open(os.path.join(outPath, "test"), "w")
    count = {}
    count_train = {}
    count_test = {}
    for line in data:
        label = line.split("\t")[0]
        count[label] = count.get(label, 0) + 1
        if count[label]%2==0 and count_train.get(label,0)<lim_train:
            train.write(line)
            count_train[label] = count_train.get(label, 0) + 1
        elif count_test.get(label,0)<lim_test:
            test.write(line)
            count_test[label] = count_test.get(label, 0) + 1
    data.close()
    train.close()
    test.close()
    print count, count_test, count_train



#  test
if __name__ == "__main__":
    inPath = "data/digital/extracted"
    outPath = "data/digital"
    partition(inPath, outPath, 500, 500)
