from pyspark import SparkConf,SparkContext
from string import punctuation
from stopwords import get_stopwords
from pyspark.mllib.feature import HashingTF,IDF
from pyspark.ml.feature import StringIndexer,VectorIndexer
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
#from nltk.stem import PorterStemmer
#from nltk import word_tokenize
# from _sqlite3 import *
#from nltk.corpus import stopwords

# Punctuation and Stopwords removal function
def strip_punctuation(s):
    excludepunc=''.join(c for c in s if c not in punctuation)
    excludestopwords=' '.join(c for c in excludepunc.split() if c not in get_stopwords("english"))
    return excludestopwords
def labelDataTransform(s):
    #Return 1 if Spam else 0
    if s=='spam':
        return 1
    else:
        return 0

conf=SparkConf().setAppName('Project').setMaster('local[6]').set('spark.driver.memory','10G')
sc=SparkContext(conf=conf)
rawdata=sc.textFile("/home/cloudera/Downloads/SMSSpamCorpus01/input")
#print(rawdata.take(5))
TotalMessages=rawdata.map(lambda line:line.rsplit(",",1))
print(TotalMessages.take(5))
PunctuationRemoved=TotalMessages.map(lambda line:strip_punctuation(line[0]))
print(PunctuationRemoved.take(20))
# Split data into labels and features, transform
# preservesPartitioning is not really required
# since map without partitioner shouldn't trigger repartitiong
labels = TotalMessages.map(lambda line:labelDataTransform(line[1]))

#print(labels.count())
print(labels.take(5))
#print(labels.count())
documents=PunctuationRemoved.map(lambda doc: doc.split("\t"))
print(documents.count())
hashing=HashingTF()
tf = hashing.transform(documents)
#tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
print(tfidf)
print(tfidf.take(5))
print(tfidf.count())
# Combine using zip
data= labels.zip(tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
training,test=data.randomSplit([0.7,0.3])
print("training data",training.count())
print("testing data",test.count())
# Naive Bayes Model
#NaiveBayesmodel(training,test)
DecisiontreeModel(training,test)



