__author__ = 'LiamZheng'
# coding=utf-8

import sys
import math
import jieba
import string
import operator
from numpy import *;
import pickle

reload(sys)
sys.setdefaultencoding('utf-8')

def trainBayes(inPath, outPath):
    train = open(inPath, "r")
    tf = {}
    df = {}
    num_docs = {}
    vocal = set()

    #  get the tf in each document and df for each feature
    for line in train :
        #  init
        label, title = line.split("\t")
        if not tf.has_key(label):
            tf[label] = []
        if not df.has_key(label):
            df[label] = {}
        num_docs[label] = num_docs.get(label, 0) + 1

        #  calculate tf and df
        tf_doc = {}
        for feature in title.split():
            tf_doc[feature] = tf_doc.get(feature, 0) + 1
            vocal.add(feature)
        for feature in tf_doc.keys():
            df[label][feature] = df[label].get(feature, 0) + 1
        tf[label].append(tf_doc)
    train.close()

    #  calculate tfidf(normalized) for each doc, and sum up the tfidf of feature for each label
    weight = {}
    for label, docs in tf.iteritems():
        if not weight.has_key(label):
            weight[label] = {}

        for tf_doc in docs:
            tfidf_doc = dict.fromkeys(tf_doc)
            for feature, num in tf_doc.iteritems():
                tfidf_doc[feature] = math.sqrt(num) * (1 + math.log(num_docs[label]*1.0/(df[label][feature] + 1)))
            norm = math.sqrt(sum([w*w for w in tfidf_doc.values()]))
            for feature, tfidf in tfidf_doc.iteritems():
                weight[label][feature] = weight[label].get(feature, 0) + tfidf/norm

    #  calculate weight of feature for each label
    num_features = len(vocal)
    for label, weights_label in weight.iteritems():
        weight_sum = sum(weights_label.values())
        weight[label]["__null__"] = 0
        for feature, w in weights_label.iteritems():
            weight[label][feature] = math.log((w+1)*1.0/(weight_sum + num_features))
    model = open(outPath, 'w')
    pickle.dump(weight,model,0)
    model.close()

def classify(weight, text, tokenized=False):
    '''get label of a text'''
    if tokenized:
        tokens = text.split()
    else:
        tokens = list(jieba.cut(text))
    scores = {}
    for label in weight.keys():
        default = weight[label]["__null__"]
        scores[label] = sum([weight[label].get(token,default) for token in tokens])
    sorted_scores = sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_scores[0][0]

def loadModel(modelPath):
    '''load Model from file'''
    model = open(modelPath, "r")
    weight = pickle.load(model)
    model.close()
    return weight

def printConfuseMat(confuseMat, labels):
    prec_all = trace(confuseMat)/sum(confuseMat)
    print "the precision of all is  ", format(prec_all, ".2%")
    print "The confuse matrix is : "
    print  '%10s' % '',
    for label in labels:
        print "%10s" % label,
    print
    for label_r in labels:
        print  '%-10s' % label_r,
        for label_p in labels:
            print "%10d" % confuseMat[labels.index(label_r)][labels.index(label_p)],
        print

def testBayes(testPath, modelPath):
    test = open(testPath, 'r')
    weight = loadModel(modelPath)
    labelNum = len(weight)
    labels = weight.keys()
    confuseMat = zeros([labelNum, labelNum])
    for line in test:
        label, title = line.split("\t")
        pred = classify(weight, title, tokenized=True)
        indexReal = labels.index(label)
        indexPred = labels.index(pred)
        confuseMat[indexReal][indexPred] +=1
    test.close()
    printConfuseMat(confuseMat, labels)


if __name__ == "__main__":
    trainPath = "data/digital/title-train"
    testPath = "data/digital/title-test"
    modelPath = "data/digital/model"
    #trainBayes(trainPath, modelPath)
    testBayes(testPath, modelPath)
    #text = "【三星】正品惊爆折扣 最全三星特卖信息"
    #weight = loadModel(modelPath)
    #print classify(weight, text)
