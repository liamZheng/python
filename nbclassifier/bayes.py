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

class Bayes:
    """base class for bayes model"""
    def __init__(self):
        self.prob_label = {}

    def train_bayes(self, train_path):
        pass

    def classify(self, text, tokenized):
        pass


    def printConfuseMat(self, confuseMat, labels):
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

    def testBayes(self, test_path, model_path):
        test = open(test_path, 'r')
        numlabels = len(self.prob_label)
        labels = self.prob_label.keys()
        confuseMat = zeros([numlabels, numlabels])
        for line in test:
            label_real, title = line.split("\t")
            label_pred = self.classify(title, tokenized=True)
            confuseMat[labels.index(label_real)][labels.index(label_pred)] +=1
        test.close()
        self.printConfuseMat(confuseMat, labels)

    def load_model(self,model_path):
        """load Model from file"""
        model = open(model_path, "r")
        self.prob_label = pickle.load(model)
        model.close()

    def dump_model(self, model_path):
        """dump model into file"""
        model = open(model_path, 'w')
        pickle.dump(self.prob_label, model,0)
        model.close()


class BayesMn(Bayes):
    """Class for multinomial bayes model """

    def train_bayes(self, train_path):
        train = open(train_path, "r")
        tf = {}
        df_label = {}
        ndoc_label = {}
        vocal_list = set()

        #  get the tf in each document and df for each feature
        for line in train :
            #  init
            label, title = line.split("\t")
            if not tf.has_key(label):
                tf[label] = []
            if not df_label.has_key(label):
                df_label[label] = {}
            ndoc_label[label] = ndoc_label.get(label, 0) + 1

            #  calculate tf and df for each class
            tf_doc = {}
            for feature in title.split():
                tf_doc[feature] = tf_doc.get(feature, 0) + 1
                vocal_list.add(feature)
            for feature in tf_doc.keys():
                df_label[label][feature] = df_label[label].get(feature, 0) + 1
            tf[label].append(tf_doc)
        train.close()

        #  Calculate normalized tfidf in each doc, and sum up the tfidf of feature for each label
        prob_label = {}
        for label, docs in tf.iteritems():
            if not prob_label.has_key(label):
                prob_label[label] = {}
            for tf_doc in docs:
                tfidf_doc = dict.fromkeys(tf_doc)
                for feature, num in tf_doc.iteritems():
                    tfidf_doc[feature] = math.sqrt(num) * (1 + math.log(ndoc_label[label]*1.0/(df_label[label][feature] + 1)))
                norm = math.sqrt(sum([w*w for w in tfidf_doc.values()]))
                for feature, tfidf in tfidf_doc.iteritems():
                    prob_label[label][feature] = prob_label[label].get(feature, 0) + tfidf/norm

        #  Calculate probability that a feature occurs in each label
        num_features = len(vocal_list)
        for label, tfidf in prob_label.iteritems():
            tfidf_sum = sum(tfidf.values())
            prob_label[label]["__null__"] = 0
            for feature, w in tfidf.iteritems():
                prob_label[label][feature] = math.log((w+1)*1.0/(tfidf_sum + num_features))
        self.prob_label = prob_label

    def classify(self, text, tokenized=False):
        '''get label of a text'''
        if tokenized:
            tokens = text.split()
        else:
            tokens = list(jieba.cut(text))
        scores = {}
        for label in self.prob_label.keys():
            default = self.prob_label[label]["__null__"]
            scores[label] = sum([self.prob_label[label].get(token,default) for token in tokens])
        sorted_scores = sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores[0][0]


class BayesBernoulli(Bayes):
    """Class for bernoulli bayes model"""



if __name__ == "__main__":
    train_path = "data/digital/title-train"
    test_path = "data/digital/title-test"
    model_path = "data/digital/model"
    #trainBayes(trainPath, modelPath)

    bayes_mn = BayesMn()
    bayes_mn.train_bayes(train_path)
    #bayes_mn.load_model(model_path)
    bayes_mn.testBayes(test_path, model_path)

    #testBayes(testPath, modelPath)
    #text = "【三星】正品惊爆折扣 最全三星特卖信息"
    #weight = loadModel(modelPath)
    #print classify(weight, text)
