# coding=utf-8
# !/usr/bin/python
'''
INFO:
DESC:
script options
--------------
--param : parameter list

Created by Samujjwal_Ghosh on 11-Apr-17.

__author__ : Samujjwal Ghosh
__version__ = ": 1 $"
__date__ = "$"
__copyright__ = "Copyright (c) 2017 Samujjwal Ghosh"
__license__ = "Python"

Supervised approaches:
    SVM,

Features:
    # 1. Unigrams, bigrams
    # 2. count of words like (lakh,lakhs,millions,thousands)
    # 3. count of units present (litre,kg,gram)
    # 4. k similar tweets class votes
    # 5. k closest same class distance avg
    # 6. count of frequent words of that class (unique to that class)
    # 7. Length related features.
'''
import os,sys,re,math,json,string,logging
import unicodedata
import heapq
import numpy as np
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn import cluster
from textblob import TextBlob as tb

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
log_file='results.log'
handler=logging.FileHandler(log_file)
handler.setLevel(logging.INFO)

# File Handling-------------------------------------------------------------------------------------
def get_date_time_tag():
    from datetime import datetime
    date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    current_file_name = os.path.basename(__file__)
    return current_file_name+"_"+date_time+"_"

date_time_tag = get_date_time_tag()

from collections import OrderedDict
from collections import defaultdict
#
def save_json(dict,filename,tag=False,dir=False):
    # print("Method: save_json(dict,file)")
    # dir_path=''
    # if dir:
        # dir_path='saved_jsons'
        # try:
            # if not os.path.isdir(dir_path):
                # os.makedirs(dir_path)
            # filename = os.path.join(dir_path,filename)
        # except Exception as e:
            # print("Could not create directory: ",dir_path)
            # print("Failure reason: ",e)

    try:
        if tag:
            # date_time_tag = get_date_time_tag()
            with open(date_time_tag+filename + ".json",'w') as outfile:
                outfile.write(json.dumps(dict,indent=4))
            outfile.close()
            return True
        else:
            with open(filename + ".json",'w') as outfile:
                outfile.write(json.dumps(dict,indent=4))
            outfile.close()
            return True
    except Exception as e:
        print("Could not write to file: ",filename)
        print("Failure reason: ",e)
        print("Writing file as plain text: ",filename + ".txt")
        if tag:
            # date_time_tag = get_date_time_tag()
            with open(date_time_tag+filename + ".txt",'w') as outfile:
                outfile.write(str(dict))
            outfile.close()
            return False
        else:
            with open(date_time_tag+filename + ".txt",'w') as outfile:
                outfile.write(str(dict))
            outfile.close()
            return False

def read_json(filename,alternate=None):
    # print("Reading JSON file: ",filename+".json")
    # if os.path.isdir(dir_path):
        # filename = os.path.join(dir_path,filename)

    print(filename+".json")

    if os.path.isfile(filename+".json"):
        with open(filename+".json","r",encoding="utf-8") as file:
            json_dict=OrderedDict(json.load(file))
        file.close()
        return json_dict
    elif alternate:
        print("Warning:",filename+" does not exist, reading ",alternate)
        alternate=read_json(alternate)
        return alternate
    else:
        print("Warning:",filename+" does not exist.")
        return False

# Globals-------------------------------------------------------------------------------------------
n_classes          =4     # number of classes
k_similar          =15    # hyper-param,# of similar tweets to find based on cosine similarity
k_unique_words     =25    # hyper-param,# of unique words to find using tf-idf per class
acronym_dict       =read_json("acronym")    # dict to hold acronyms
class_names=['RESOURCES AVAILABLE',
             'RESOURCES REQUIRED',
             'INFRASTRUCTURE DAMAGE, RESTORATION AND CASUALTIES REPORTED',
             'RESCUE ACTIVITIES OF VARIOUS NGOs / GOVERNMENT ORGANIZATIONS'
            ]

# Preprocess----------------------------------------------------------------------------------------
emoticons_str=r'''
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )'''
regex_str=[
    emoticons_str,
    r'<[^>]+>',# HTML tags
    r'(?:@[\w_]+)',# @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",# hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',# URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',# numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",# words with - and '
    r'(?:[\w_]+)',# other words
    r'(?:\S)' # anything else
]
tokens_re  =re.compile(r'('+'|'.join(regex_str)+')',re.VERBOSE | re.IGNORECASE)
emoticon_re=re.compile(r'^'+emoticons_str+'$',re.VERBOSE | re.IGNORECASE)

def preprocess(s,lowercase=False):
    # print("Method: preprocess(s,lowercase=False)")
    tokens=tokens_re.findall(str(s))
    if lowercase:
        tokens=[token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def parse_tweet(tweet):
    # print("Method: parse_tweet(tweet)")
    # print(tweet)
    stop=stopwords.words('english') + list(string.punctuation) + ['rt','via','& amp']
    tweet=re.sub(r"http\S+","urlurl",tweet) # replaces hyperlink with urlurl
    terms=preprocess(tweet,True)
    for term_pos in range(len(terms)):
        terms[term_pos]=terms[term_pos].replace("@","")
        terms[term_pos]=terms[term_pos].replace("#","")
        terms[term_pos]=get_acronyms(terms[term_pos])
        terms[term_pos]=contains_phone(terms[term_pos])
        #TODO: pre-process the acronym
    mod_tweet=" ".join([term for term in terms if term not in stop])
    return mod_tweet

def get_acronyms(term):
    '''Check for Acronyms and returns the acronym of the term'''
    # print("Method: get_acronyms(term)",term)
    global acronym_dict
    if term in acronym_dict.keys():
        # print(term," -> ",acronym_dict[term])
        return acronym_dict[term]
    else:
        return term

# Features------------------------------------------------------------------------------------------
def k_similar_tweets(train,test,k_similar):
    '''Finds k_similar tweets in train for each test tweet using cosine similarity'''
    # print("Method: k_similar_tweets(train,test,k_similar)")
    k_sim_twts=OrderedDict()
    for t_twt_id,t_twt_val in test.items():
        i=0
        sim_list=OrderedDict()
        for tr_twt_id,tr_twt_val in train.items():
            if t_twt_id == tr_twt_id:
                # print(t_twt_id,"Already exists in list,ignoring...")
                continue
            if i<k_similar:
                sim_list[tr_twt_id]=get_cosine(t_twt_val["parsed_tweet"],tr_twt_val["parsed_tweet"])
                # sim_list[tr_twt_id]=get_jackard(t_twt_val["parsed_tweet"],tr_twt_val["parsed_tweet"])
                i=i+1
            else:
                new_sim_twt=get_cosine(t_twt_val["parsed_tweet"],tr_twt_val["parsed_tweet"])
                # new_sim_twt=get_jackard(t_twt_val["parsed_tweet"],tr_twt_val["parsed_tweet"])
                for sim_id,sim_val in sim_list.items():
                    if new_sim_twt > sim_val:
                        del sim_list[sim_id]
                        sim_list[tr_twt_id]=new_sim_twt
                        break
        k_sim_twts[t_twt_id]=sim_list
    return k_sim_twts

def sim_tweet_class_vote(train,test,sim_vals):
    '''Returns the vote counts of train tweets of similar tweets for test tweets '''
    # print("Method: sim_tweet_class_vote(train,test,sim_vals)")
    for id,sim_dict in sim_vals.items():
        class_votes=[0] * n_classes
        for t_id in sim_dict.keys():
            for tr_cls in train[t_id]["classes"]:
                class_votes[tr_cls]=class_votes[tr_cls]+1
        test[id]["knn_votes"]=class_votes
    return class_votes

def find_word(tweet_text,word_list):
    tweet_text_blob=tb(tweet_text)
    word_count=0
    for term in word_list:
        if term in tweet_text_blob.words.lower():
            word_count=word_count+1
    return word_count

def contains_phone(text):
    phonePattern=re.compile(r'''
                # don't match beginning of string,number can start anywhere
    (\d{3})     # area code is 3 digits (e.g. '800')
    \D*         # optional separator is any number of non-digits
    (\d{3})     # trunk is 3 digits (e.g. '555')
    \D*         # optional separator
    (\d{4})     # rest of number is 4 digits (e.g. '1212')
    \D*         # optional separator
    (\d*)       # extension is optional and can be any number of digits
    $           # end of string
    ''',re.VERBOSE)
    # return len(phonePattern.findall(text))
    if len(phonePattern.findall(text)) > 0:
        return "phonenumber"
    else :
        return text

def contains(train,unique_words,feature_count=False):
    # print("Method: contains(train,unique_words,feature_count=False)")
    units  =tb('litre liter kg kilogram gram packet kilometer meter pack sets ton meal equipment kit percentage')
    units  =units.words+units.words.pluralize()
    number =tb('lac lakh million thousand hundred')
    number =number.words+number.words.pluralize()
    ra     =tb('treat send sent sending supply offer distribute treat mobilize mobilized donate donated dispatch dispatched')
    ra     =ra.words+ra.words.pluralize()
    rr     =tb('need requirement require ranout shortage scarcity')
    rr     =rr.words+rr.words.pluralize()
    medical=tb('medicine hospital medical doctor injection syringe ambulance antibiotic')
    medical=medical.words+medical.words.pluralize()
    url    =tb('urlurl')
    phone  =tb('phonenumber')
    loc    =tb('at')

    if feature_count:
        units_count = 0
        number_count = 0
        ra_count = 0
        rr_count = 0
        medical_count = 0
        loc_count = 0
        url_count = 0
        phone_count = 0
        feature_names = ['units','number','ra','rr','medical','loc','url','phone']
        feature_count_matrix=np.zeros((n_classes, (len(feature_names) + 1)))

    for id,vals in train.items():
        train[id]['units']  =find_word(vals['text'],units)
        train[id]['number'] =find_word(vals['text'],number)
        train[id]['ra']     =find_word(vals['text'],ra)
        train[id]['rr']     =find_word(vals['text'],rr)
        train[id]['medical']=find_word(vals['text'],medical)
        train[id]['loc']    =find_word(vals['text'],loc)
        train[id]['url']    =find_word(vals['text'],url)
        train[id]['phone']  =find_word(vals['text'],phone)
        train[id]['word']   =len(vals["parsed_tweet"].split())
        train[id]['char']   =len(vals["parsed_tweet"])-vals["parsed_tweet"].count(' ')
        train[id]['unique'] =unique_word_count_class(vals["parsed_tweet"],unique_words)
        train[id]['char_space']=len(vals["parsed_tweet"])
        if feature_count:
            for cls in train[id]['classes']:
                feature_count_matrix[cls][0] = feature_count_matrix[cls][0] + train[id]['units']
                units_count = units_count + train[id]['units']
                feature_count_matrix[cls][1] = feature_count_matrix[cls][1] + train[id]['number']
                number_count = number_count + train[id]['number']
                feature_count_matrix[cls][2] = feature_count_matrix[cls][2] + train[id]['ra']
                ra_count = ra_count + train[id]['ra']
                feature_count_matrix[cls][3] = feature_count_matrix[cls][3] + train[id]['rr']
                rr_count = rr_count + train[id]['rr']
                feature_count_matrix[cls][4] = feature_count_matrix[cls][4] + train[id]['medical']
                medical_count = medical_count + train[id]['medical']
                feature_count_matrix[cls][5] = feature_count_matrix[cls][5] + train[id]['loc']
                loc_count = loc_count + train[id]['loc']
                feature_count_matrix[cls][6] = feature_count_matrix[cls][6] + train[id]['url']
                url_count = url_count + train[id]['url']
                feature_count_matrix[cls][7] = feature_count_matrix[cls][7] + train[id]['phone']
                phone_count = phone_count + train[id]['phone']

    if feature_count:
        print(feature_names)
        print(feature_count_matrix)

        print(units_count)
        print(number_count)
        print(ra_count)
        print(rr_count)
        print(medical_count)
        print(loc_count)
        print(url_count)
        print(phone_count)
        for i in range(len(feature_count_matrix)):
            # for cls in train[id]['classes']:
            feature_count_matrix[i][0] = feature_count_matrix[i][0] / units_count
            feature_count_matrix[i][1] = feature_count_matrix[i][1] / number_count
            feature_count_matrix[i][2] = feature_count_matrix[i][2] / ra_count
            feature_count_matrix[i][3] = feature_count_matrix[i][3] / rr_count
            feature_count_matrix[i][4] = feature_count_matrix[i][4] / medical_count
            feature_count_matrix[i][5] = feature_count_matrix[i][5] / loc_count
            feature_count_matrix[i][6] = feature_count_matrix[i][6] / url_count
            feature_count_matrix[i][7] = feature_count_matrix[i][7] / phone_count
        print(feature_names)
        print(feature_count_matrix)

def unique_word_count_class(text,unique_words):
    cls_counts=[0] * n_classes
    for word in text.split():
        for cls in range(len(unique_words)):
            if word in unique_words[cls]:
                cls_counts[cls]=cls_counts[cls] + 1
    return cls_counts

def create_corpus(data,n_classes):
    # print("Method: create_corpus(data,n_classes)")
    total_corpus=[]
    class_corpuses=dict((key,[]) for key in range(n_classes))
    for id,vals in data.items():
        total_corpus.append(vals["parsed_tweet"])
        class_corpuses[vals["classes"][0]].append(vals["parsed_tweet"])
    return total_corpus,class_corpuses

def most_freq_words(corpus,k_most_common):
    return FreqDist(corpus).most_common(k_most_common)

def tf(word,blob):
    '''computes "term frequency" which is the number of times a word appears in a document blob,
    normalized by dividing by the total number of words in blob.'''
    return blob.words.count(word) / len(blob.words)

def n_containing(word,bloblist):
    '''number of documents containing word'''
    return sum(1 for blob in bloblist if word in blob)

def idf(word,bloblist):
    '''computes "inverse document frequency" which measures how common a word is among all documents in bloblist. The more common a word is, the lower its idf'''
    return math.log(len(bloblist) / (1 + n_containing(word,bloblist)))

def tfidf(word,blob,bloblist):
    '''computes the TF-IDF score. It is simply the product of tf and idf'''
    return tf(word,blob) * idf(word,bloblist)

def get_cosine(tweet1,tweet2):
    '''calculates the cosine similarity between 2 tweets'''
    # print("Method: get_cosine(tweet1,tweet2)")
    from collections import Counter
    WORD=re.compile(r'\w+')
    vec1=Counter(WORD.findall(tweet1))
    vec2=Counter(WORD.findall(tweet2))

    intersection=set(vec1.keys()) & set(vec2.keys())
    numerator=sum([vec1[x] * vec2[x] for x in intersection])

    sum1=sum([vec1[x]**2 for x in vec1.keys()])
    sum2=sum([vec2[x]**2 for x in vec2.keys()])
    denominator=math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def unique_words_class(class_corpuses):
    ''' Finds unique words for each class'''
    # print("Method: unique_words_class(class_corpuses)")
    bloblist=[]
    unique_words=defaultdict()
    for cls_id,text in class_corpuses.items():
        bloblist.append(tb(" ".join(text)))
    for i,blob in enumerate(bloblist):
        unique_words[i]=[]
        # print("\nTop words in class {}".format(i))
        scores={word: tfidf(word,blob,bloblist) for word in blob.words}
        sorted_words=sorted(scores.items(),key=lambda x: x[1],reverse=True)
        for word,score in sorted_words[:k_unique_words]:
            # print("{},TF-IDF: {}".format(word,round(score,5)))
            unique_words[i].append(word)
    return unique_words

def create_tf_idf(train,test,n_gram=1):
    '''Calculates tf-idf vectors for train and test'''
    # print("Method: create_tf_idf(train,test)")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer=TfidfVectorizer(strip_accents='unicode',decode_error='ignore',ngram_range=(1,n_gram))
    train_tfidf_matrix=tfidf_vectorizer.fit_transform([vals["parsed_tweet"] for twt_id,vals in train.items()])
    test_tfidf_matrix =tfidf_vectorizer.transform([vals["parsed_tweet"] for twt_id,vals in test.items()])
    return train_tfidf_matrix,test_tfidf_matrix

# Supervised----------------------------------------------------------------------------------------
def supervised(train,test,train_tfidf_matrix,test_tfidf_matrix,init_C=10,probability=True,metric=False,grid=True):
    # print("Method: supervised(train,test,train_tfidf_matrix,test_tfidf_matrix)")
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier

    mlb=MultiLabelBinarizer()
    train_labels=[vals["classes"] for id,vals in train.items()]
    train_labels_bin=mlb.fit_transform(train_labels)

    if metric or grid:
        result=OrderedDict()
        test_labels=[vals["classes"] for id,vals in test.items()]

    print("\nAlgorithm: \t \t \t SVM")
    SVM =OneVsRestClassifier(SVC(kernel='linear',C=init_C,probability=True))
    if grid:
        SVM_params = [
            {'estimator__C': [10000,1000,100,10,1]},
        ]
        SVM_grid = grid_search(SVM,SVM_params,train_tfidf_matrix,train_labels_bin,test_tfidf_matrix,mlb.fit_transform(test_labels),cv=5,score='f1')
        SVM =OneVsRestClassifier(SVC(kernel='linear',C=SVM_grid['params']['estimator__C'],probability=True))
        SVM_fit =SVM.fit(train_tfidf_matrix,train_labels_bin)
        SVM_pred =SVM_fit.predict(test_tfidf_matrix)
        SVM_proba =SVM_fit.predict_proba(test_tfidf_matrix)
        result["SVM_grid"] = SVM_grid
    else:
        SVM_fit =SVM.fit(train_tfidf_matrix,train_labels_bin)
        SVM_pred =SVM_fit.predict(test_tfidf_matrix)
        SVM_proba =SVM_fit.predict_proba(test_tfidf_matrix)

    if metric or grid:
        accuracy_multi(test_labels,mlb.inverse_transform(SVM_pred))
        # result["SVM_metric"]=sklearn_metrics(mlb.fit_transform(test_labels),SVM_pred)
        return result,mlb.inverse_transform(SVM_pred),SVM_proba
    return None,mlb.inverse_transform(SVM_pred),SVM_proba

# Accuracy------------------------------------------------------------------------------------------
def sklearn_metrics(actual,predicted,target_names=class_names,digits=4):
    # print("Method: sklearn_metrics(actual,predicted,target_names=class_names,digits=4)")
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_recall_fscore_support

    results = OrderedDict()

    results["accuracy"] = accuracy_score(actual,predicted)
    results["precision_macro"] = precision_score(actual,predicted,average='macro')
    results["precision_micro"] = precision_score(actual,predicted,average='micro')
    results["recall_macro"] = recall_score(actual,predicted,average='macro')
    results["recall_micro"] = recall_score(actual,predicted,average='micro')
    results["f1_macro"] = f1_score(actual,predicted,average='macro')
    results["f1_micro"] = f1_score(actual,predicted,average='micro')
    results["Precision_class"] = precision_recall_fscore_support(actual,predicted)[0].tolist()
    results["Recall_class"] = precision_recall_fscore_support(actual,predicted)[1].tolist()
    results["F1_class"] = precision_recall_fscore_support(actual,predicted)[2].tolist()

    from termcolor import colored, cprint
    # text = colored('accuracy_score: ', 'green', attrs=['blink'])
    text = 'accuracy_score: '
    print(text,'\x1b[1;31m',results["accuracy"],'\x1b[0m')
    print("\t\t\t Macro,\t\t\t Micro")
    print("\t\t\t -----,\t\t\t -----")
    print("Precision:\t\t",results["precision_macro"],"\t",results["precision_micro"])
    print("Recall:\t\t\t",results["recall_macro"],"\t",results["recall_micro"])
    print("f1:\t\t\t",results["f1_macro"],"\t",results["f1_micro"])
    # print("Precision: ",results["Precision"])
    # print("Recall: ",results["Recall"])
    # print("F1: ",results["F1"])
    print(classification_report(y_true=actual,y_pred=predicted,target_names=class_names,digits=digits))
    print("\n")
    return results

def accuracy_multi(actual,predicted,multi=True):
    '''Calculates (Macro,Micro) precision,recall'''
    # print("Method: accuracy_multi(actual,predicted,multi=True)")
    if len(actual) != len(predicted):
        print("** length does not match: ",len(actual),len(predicted))
    class_count=[0] * n_classes
    for i in range(len(actual)):
        if multi:
            for pred_label in predicted[i]:
                if pred_label in actual[i]:
                    class_count[pred_label]=class_count[pred_label]+1
        else:
            if actual[i] == predicted[i]:
                class_count[predicted[i]]=class_count[predicted[i]]+1
    print("Predicted counts per class:\t",class_count)

def split_data(lab_tweets,test_size,random_state=0):
    ''' splits the data based on test_size'''
    # print("Method: split_data(lab_tweets,test_size)")
    from sklearn.model_selection import train_test_split
    all_list=list(lab_tweets.keys())
    train_split,test_split=train_test_split(all_list,test_size=test_size,random_state=random_state)
    train=OrderedDict()
    test=OrderedDict()
    for id in train_split:
        train[id]=lab_tweets[id]
    for id in test_split:
        test[id]=lab_tweets[id]
    return train,test

def add_features_matrix(train,train_matrix,derived=True,manual=True,length=True):
    # print("Method: add_features_matrix(train,train_matrix,lengths=False)")
    if manual:
        # print("Adding Manual features...")

        loc = np.matrix([[val["loc"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((train_matrix,loc), axis=1)

        medical = np.matrix([[val["medical"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,medical), axis=1)

        number = np.matrix([[val["number"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,number), axis=1)

        ra = np.matrix([[val["ra"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,ra), axis=1)

        rr = np.matrix([[val["rr"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,rr), axis=1)

        units = np.matrix([[val["units"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,units), axis=1)

    if derived:
        # print("Adding Derived features...")

        retweet_count_max = max([val["retweet_count"] for id,val in train.items()])
        retweet_count = np.matrix([[val["retweet_count"] / retweet_count_max] for id,val in train.items()])
        new = np.concatenate((train_matrix,retweet_count), axis=1)

        url = np.matrix([[val["url"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,url), axis=1)

        phone = np.matrix([[val["phone"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,phone), axis=1)

        for i in range(n_classes):
            unique = np.matrix([val["unique"][i] / k_unique_words for id,val in train.items()])
            new = np.concatenate((new,unique.T), axis=1)

        for i in range(n_classes):
            knn_votes = np.matrix([val["knn_votes"][i] / k_unique_words for id,val in train.items()])
            new = np.concatenate((new,knn_votes.T), axis=1)

    if length:
        # print("Adding Length features...")

        char_max = max([val["char"] for id,val in train.items()])
        char = np.matrix([[(val["char"] / char_max)] for id,val in train.items()])
        new = np.concatenate((train_matrix,char), axis=1)

        char_space_max = max([val["char_space"] for id,val in train.items()])
        char_space = np.matrix([[(val["char_space"] / char_space_max)] for id,val in train.items()])
        new = np.concatenate((new,char_space), axis=1)

        word_max = max([val["word"] for id,val in train.items()])
        word = np.matrix([[val["word"] / word_max] for id,val in train.items()])
        new = np.concatenate((new,word), axis=1)

    return new

from plotly.graph_objs import *
def plot(results, iter):
    import plotly.plotly as py
    # print("Method: plot(results, iter)")

    # py.sign_in('samujjwal86', 'U3gIQsZHKYNN5q3fqKF0')
    py.sign_in('samiith', 'KKbfgXade8SAc8VrI30z')

    metrics = ['Precision', 'Recall', 'F1']
    for index,metric in enumerate(metrics):
        class_values = []
        for algo,values in results.items():
            class_values.append({
              "x": ["1","2","3","4","5","6","7",],
              "y": values[metric],
              "name": algo,
              "type": "bar",
              "xaxis": "x",
              "yaxis": "y",
            })
        data = Data(class_values)

        layout = {
            "autosize": True,
            "barmode": "group",
            "dragmode": "zoom",
            "hovermode": "x",
            "legend": {"orientation": "h"},
            "margin": {
                "t": 40,
                "b": 110
            },
            "showlegend": True,
            "title": metric,
            "xaxis": {
                "anchor": "y",
                "autorange": True,
                "domain": [-1, 1],
                "dtick": 1,
                "exponentformat": "none",
                "fixedrange": False,
                "nticks": 1,
                "range": [-0.5, 6.5],
                "showgrid": False,
                "showline": False,
                "showticklabels": True,
                "ticks": "",
                "title": "<b>Classes</b>",
                "type": "category",
                "zeroline": False
            },
            "yaxis": {
                "anchor": "x",
                "autorange": True,
                "dtick": 0.05,
                "range": [0, 1],
                "showgrid": False,
                "tick0": 0,
                "tickangle": "auto",
                "tickmode": "linear",
                "tickprefix": "",
                "ticks": "",
                "title": "<b>Values</b>",
                "type": "linear",
                "zeroline": False
          },
        }
        fig = Figure(data=data, layout=layout)
        filename=metric+'_'+iter
        print(filename)
        try:
            py.image.save_as(fig, filename=filename+'.png')
            plot_url = py.plot(fig,filename= metric)
        except Exception as e:
            print("Could not plot graph. Failure reason: ",e)

def features(train,test,feature_count=False):
    # print("Method: features(train,test)")
    sim_vals_train=k_similar_tweets(train,train,k_similar)
    sim_vals_test=k_similar_tweets(train,test,k_similar)
    sim_tweet_class_vote(train,train,sim_vals_train)
    sim_tweet_class_vote(train,test,sim_vals_test)

    total_corpus,class_corpuses=create_corpus(train,n_classes)
    unique_words=unique_words_class(class_corpuses) # TODO: unique word list can be increased by iteration on test data as followed in "SMERP paper 1"

    contains(train,unique_words,feature_count=feature_count)
    contains(test,unique_words,feature_count=feature_count)

def write_file(data,file_name,mode='w',tag=False):
    if tag:
        # date_time_tag = get_date_time_tag()
        with open(date_time_tag+file_name+".txt",mode, encoding="utf-8") as out_file:
            out_file.write(str(data))
            out_file.write("\n")
            out_file.write("\n")
        out_file.close()
    else:
        with open(file_name+".txt",mode, encoding="utf-8") as out_file:
            out_file.write(str(data))
            out_file.write("\n")
            out_file.write("\n")
        out_file.close()

def parse_tweets(train):
    # print("Method: parse_tweets(train)")
    for id,val in train.items():
        val['parsed_tweet'] = parse_tweet(val['text'])
    return train

def open_word2vec(word2vec):
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(word2vec, binary=True)
    return model

def use_word2vec(train,w2v):
    train_vec=OrderedDict()
    for id,val in train.items():
        s_vec = np.zeros(300)
        for word in val['parsed_tweet'].split(" "):
            if word in w2v.vocab:
                # train_vec[id][word] = w2v[word].tolist()
                s_vec = np.add(s_vec, w2v[word])
            else:
                pass
                # print("Word [",word,"] not in vocabulary")
            # print("\n")
        train_vec[id]=s_vec
    return train_vec

def expand_tweet(w2v,tweet):
    new_tweet = []
    for word in tweet.split(" "):
        new_tweet= new_tweet+[word]
        if word in w2v.vocab:
            w2v_words=w2v.most_similar(positive=[word], negative=[], topn=3)
            for term,val in w2v_words:
                new_tweet= new_tweet+[term]
    return new_tweet

def expand_tweets(w2v,dict):
    # print("Method: expand_tweets(dict)")
    for id,val in dict.items():
        val['expanded_tweet'] = "".join(expand_tweet(w2v,val['parsed_tweet']))
    return dict

def grid_search(model,params,X_train,y_train,X_test,y_test,cv=5,score='f1'):
    # print("Method: grid_search(model,params,X_train,y_train,X_test,y_test,cv=5,score='f1')")
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import RandomizedSearchCV

    grid_results = OrderedDict()
    print("# Cross Validation set size: %s \n" % cv)
    print("Params: ",params)
    clf = GridSearchCV(model,params,cv=cv,scoring='%s_macro' % score)
    print("Fitting...")
    clf.fit(X_train, y_train)
    grid_results['params'] = clf.best_params_
    grid_results['score'] = clf.best_score_
    print("\nGrid scores on development set: ")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.6f (+/-%0.06f) for %r"
              % (mean, std * 2, params))
    print()
    print("Best parameters set found on development set: ",clf.best_params_)

    y_true, y_pred = y_test, clf.predict(X_test)
    grid_results["report"] = sklearn_metrics(y_true,y_pred,digits=4)
    return grid_results

def read_unlabeled_json(unlabeled_file):
    # print("Method: read_unlabeled_json(unlabeled_file)")
    unlabeled_tweets_dict = OrderedDict()
    with open(unlabeled_file+".json", 'r', encoding="utf-8") as f:
        for line in f:
            urls = []
            hashtags = []
            users = []
            line = json.loads(line)
            # print(json.dumps(line,indent=4,sort_keys=True))
            try:
                tweet_text = line["retweeted_status"]["text"]
            except KeyError:
                tweet_text = line["text"]
            tweet_text = unicodedata.normalize('NFKD',tweet_text).encode('ascii','ignore').decode("utf-8")
            parsed_tweet = parse_tweet(tweet_text)
            tweet_id = line["id"]
            # retweet_count = line["retweet_count"]
            # for url in line["entities"]["urls"]:
                # urls.append(url["expanded_url"])
            # for hashtag in line["entities"]["hashtags"]:
                # hashtags.append(hashtag["text"])
            # for user in line["entities"]["user_mentions"]:
                # users.append(user["screen_name"])
            unlabeled_tweets_dict[str(tweet_id)] = line
            # unlabeled_tweets_dict[str(tweet_id)] = OrderedDict()
            # unlabeled_tweets_dict[str(tweet_id)]['text'] = tweet_text
            # unlabeled_tweets_dict[str(tweet_id)]['parsed_tweet'] = parsed_tweet
            # unlabeled_tweets_dict[str(tweet_id)]['retweet_count'] = retweet_count
            # unlabeled_tweets_dict[str(tweet_id)]['urls'] = urls
            # unlabeled_tweets_dict[str(tweet_id)]['hashtags'] = hashtags
            # unlabeled_tweets_dict[str(tweet_id)]['users'] = users
            unlabeled_tweets_dict[str(tweet_id)]['classes'] = []
            # print(json.dumps(unlabeled_tweets_dict[str(tweet_id)],indent=4,sort_keys=True))
    f.close()
    return unlabeled_tweets_dict

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    # print("Method: merge_dicts(*dict_args)")
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def count_class(list):
    """count new tweets class poportions"""
    # print("Method: count_class(list)")
    class_count=[0] * n_classes
    for i in range(len(list)):
        for cls in list[i]:
            class_count[cls]=class_count[cls]+1
    # print(class_count)
    return class_count

def select_tweets(test_unlabeled,unlabeled_pred,unlabeled_proba,threshold=0.8):
    '''Select new tweets with threshold to be added to train set'''
    # print("Method: select_tweets(test_unlabeled,unlabeled_pred,unlabeled_proba,threshold=0.8)")
    labelled_selected = OrderedDict()
    for i, (id, val) in enumerate(test_unlabeled.items()):
        if unlabeled_pred[i]:
            sel_cls=[]
            for pos,prob in enumerate(unlabeled_proba[i]):
                if prob > threshold:
                    sel_cls.append(pos)

            if len(sel_cls): # adding only if at least one class has proba > threshold
                val["classes"] = sel_cls
                val["probabilities"] = list(unlabeled_proba[i])
                labelled_selected[id] = val

    return labelled_selected

def max_add_unlabeled(train,test_unlabeled,unlabeled_pred,unlabeled_proba,max_add_portion=0.3,threshold=0.8,iter=0):
    # print("Method: max_add_unlabeled(train,test_unlabeled,unlabeled_pred,unlabeled_proba,max_add_portion=0.3,threshold=0.8,iter=0)")
    if threshold < 0.5:
        print("Threshold value very low: ", threshold)

    if len(test_unlabeled)!=len(unlabeled_pred)!=len(unlabeled_proba):
        print("Lengths does not match: ",len(test_unlabeled),len(unlabeled_pred),len(unlabeled_proba))
        exit(0)

    print("Predicted class proportions:",count_class(unlabeled_pred))

    print("Selecting tweets with threshold = ",threshold," to be added to train set")
    new_labeled_threshold = select_tweets(test_unlabeled,unlabeled_pred,unlabeled_proba,threshold=threshold)
    if len(new_labeled_threshold) < 1:
        print("No new tweet got selected, dict length: ",len(new_labeled_threshold))
        print("Returning original train set.")
        return train

    train_class_counts = count_class([val["classes"] for id,val in train.items()])
    allowed_class_counts = [int(math.ceil(x * max_add_portion)) for x in train_class_counts]
    new_labeled_threshold_class_counts = count_class([val["classes"] for id,val in new_labeled_threshold.items()])
    print("Count class portions of selected tweets : ",new_labeled_threshold_class_counts)
    print("Allowed class portions : ",allowed_class_counts)
    sel_new=OrderedDict()
    for cls in range(n_classes): # add maximum 30% of training data per class
        # print(count_class([val["classes"] for id,val in new_labeled_threshold.items()]))
        if new_labeled_threshold_class_counts[cls] > allowed_class_counts[cls]:
            print("Current selected tweets count",new_labeled_threshold_class_counts[cls],"crosses maximum allowed",int(train_class_counts[cls] * max_add_portion),", for class ",cls,". Removing extra tweets.")
            i=0
            for id,val in new_labeled_threshold.items():
                if i < allowed_class_counts[cls] and cls in val['classes']:
                    sel_new[id]=val
                    i=i+1
    print(len(sel_new))
    new_labeled_threshold_class_counts = count_class([val["classes"] for id,val in sel_new.items()])
    # save_json(sel_new,"sel_new_"+str(iter),tag=True)
    print("Count class portions of selected tweets : ",new_labeled_threshold_class_counts)
    print("Adding ",len(sel_new)," new selected tweets to train set")
    print("Selected {:3.2f}% of new labeled tweets to be added to train set.".format((len(sel_new) / len(test_unlabeled))*100))
    train = merge_dicts(train,sel_new)
    return train

def test_f1(labelled_train_new,test,iter=0):
    # print("Method: test_f1(labelled_train_new,test,iter)")
    train_tfidf_matrix_1,test_tfidf_matrix_1=create_tf_idf(labelled_train_new,test,1)
    train_tfidf_matrix_1 = train_tfidf_matrix_1.todense()
    test_tfidf_matrix_1 = test_tfidf_matrix_1.todense()
    features(labelled_train_new,test)
    train_tf_idf1_manual=add_features_matrix(labelled_train_new,train_tfidf_matrix_1)
    test_tf_idf1_manual=add_features_matrix(test,test_tfidf_matrix_1)
    test_f1_param_result,predictions,probabilities=supervised(labelled_train_new,test,train_tfidf_matrix_1,test_tfidf_matrix_1,metric=True,grid=True)
    return test_f1_param_result

def init_w2v(nlp=True):
    if nlp:
        nlp_path = '/home/cs16resch01001/data/crisisNLP_word2vec_model/'
        nlp_file = 'crisisNLP_word_vector.bin'
        w2v = open_word2vec(os.path.join(nlp_path,nlp_file))
        print("NLP Word2Vec selected")
    else:
        g_path   = '/home/cs16resch01001/data/'
        g_file   = 'GoogleNews-vectors-negative300.bin'
        w2v = open_word2vec(os.path.join(g_path,g_file))
        print("Google Word2Vec selected")
    return w2v

def read_labeled_data(file_name,test_size=0.3,validation_size=0.3):
    # print("Method: read_labeled_data(file_name,test_size=0.3,validation_size=0.3)")
    ## Reading Labelled Data:
    print("Reading Labelled Data (Randomized)")
    lab_tweets=read_json(file_name)
    train,test=split_data(lab_tweets,test_size)
    print("train size:",len(train))
    print("test size:",len(test))
    train,validation=split_data(train,validation_size)
    print("validation size:",len(validation))
    return train,validation,test

def read_json_array(json_array_file):
    # print("Method: read_json_array(json_array_file)")
    json_array = OrderedDict()
    # print(json_array_file)
    data=open(json_array_file+'.json')
    f = json.load(data)
    for line in f:
        urls = []
        hashtags = []
        users = []
        # print(json.dumps(line,indent=4,sort_keys=True))
        try:
            tweet_text = line["retweeted_status"]["text"]
        except KeyError:
            tweet_text = line["text"]
        tweet_text = unicodedata.normalize('NFKD',tweet_text).encode('ascii','ignore').decode("utf-8")
        parsed_tweet = parse_tweet(tweet_text)
        tweet_id = line["id"]
        retweet_count = line["retweet_count"]
        for url in line["entities"]["urls"]:
            urls.append(url["expanded_url"])
        for hashtag in line["entities"]["hashtags"]:
            hashtags.append(hashtag["text"])
        for user in line["entities"]["user_mentions"]:
            users.append(user["screen_name"])
        json_array[str(tweet_id)] = OrderedDict()
        json_array[str(tweet_id)]['text'] = tweet_text
        json_array[str(tweet_id)]['parsed_tweet'] = parsed_tweet
        json_array[str(tweet_id)]['retweet_count'] = retweet_count
        json_array[str(tweet_id)]['urls'] = urls
        json_array[str(tweet_id)]['hashtags'] = hashtags
        json_array[str(tweet_id)]['users'] = users
        json_array[str(tweet_id)]['classes'] = []
        # print(json.dumps(json_array[str(tweet_id)],indent=4,sort_keys=True))
    return json_array

def read_smerp_labeled():
    file_names = [0,1,2,3]
    lab = OrderedDict()
    for file in file_names:
        # print("Reading file: ","smerp"+str(file)+".json")
        single = read_json_array("smerp"+str(file))
        for id, val in single.items():
            if id in lab:
                lab[id]["classes"].append(file)
            else:
                lab[id]=val
                lab[id]["classes"]=[]
                lab[id]["classes"].append(file)
            # print(val["classes"])
        # print("Finished file: ","smerp"+str(file)+".json")
        # lab = merge_dicts(lab,single)
    return lab

def read_smerp_unlabeled():
    file_names = [0,1,2,3]
    lab = OrderedDict()
    single = read_json_array("smerp"+str(file))
    for id, val in single.items():
        if id in lab:
            lab[id]["classes"].append(file)
        else:
            lab[id]=val
            lab[id]["classes"]=[]
            lab[id]["classes"].append(file)
        # print(val["classes"])
    # print("Finished file: ","smerp"+str(file)+".json")
    # lab = merge_dicts(lab,single)
    return lab

# Main----------------------------------------------------------------------------------------------
def main():
    # print("Method: main()")
    np.set_printoptions(threshold=np.inf,precision=4,suppress=True)
    # algo_list=["SVM_Linear"]
    # train_class_counts = [163, 96, 988, 175]
    # validation_class_counts = [56, 42, 440, 71]
    # test_class_counts = [91, 75, 588, 114]

    # to use word2vec
    # w2v = init_w2v()

    ## Reading SMERP labeled Data
    print("Reading SMERP labeled Data")
    smerp_labeled_file = 'smerp_labeled_'
    if os.path.isfile(smerp_labeled_file+"train"+".json") and os.path.isfile(smerp_labeled_file+"validation"+".json") and os.path.isfile(smerp_labeled_file+"test"+".json"):
        train = read_json(smerp_labeled_file+"train")
        validation = read_json(smerp_labeled_file+"validation")
        test = read_json(smerp_labeled_file+"test")
    elif os.path.isfile(smerp_labeled_file+".json"):
        train,validation,test = read_labeled_data(smerp_labeled_file,test_size=0.3,validation_size=0.3)
        save_json(train,smerp_labeled_file+"train")
        save_json(validation,smerp_labeled_file+"validation")
        save_json(test,smerp_labeled_file+"test")
    else:
        smerp_labeled = read_smerp_labeled()
        save_json(smerp_labeled,smerp_labeled_file)
        train,validation,test = read_labeled_data(smerp_labeled_file,test_size=0.3,validation_size=0.3)
        print("Number of SMERP labeled tweets: ",len(smerp_labeled))
        save_json(train,smerp_labeled_file+"train")
        save_json(validation,smerp_labeled_file+"validation")
        save_json(test,smerp_labeled_file+"test")

    print(count_class([val["classes"] for id,val in train.items()]))
    print(count_class([val["classes"] for id,val in validation.items()]))
    print(count_class([val["classes"] for id,val in test.items()]))

    train = parse_tweets(train)
    # train = expand_tweets(w2v,train)
    validation = parse_tweets(validation)
    # validation = expand_tweets(w2v,validation)
    test = parse_tweets(test)
    # test = expand_tweets(w2v,test)
    train = merge_dicts(train,validation)

    ## recording supervised result
    # print("Recording supervised result: ")
    # initial_result_train_param=test_f1(train,test)
    # print("Initial result with train and test with parameter tuning: ",json.dumps(initial_result_train_param,indent=4))
    # save_json(initial_result_train_param,"initial_result_train_param",tag=True)
    save_json(train,"train_mod_"+str(i),tag=True)

    ## Reading unlabeled Data
    print("Reading Unlabeled Data")
    smerp_unlabeled_file = 'smerp_unlabeled_l2'
    unlabeled_file_path = '/home/cs16resch01001/datasets/SMERP2017-dataset'
    unlabeled_file_name = 'SMERP2017-data-challenge-tweetids-level2.txt'
    if os.path.isfile(unlabeled_file_name+".json"):
        unlabeled_data = read_json(smerp_unlabeled_file)
    else:
        unlabeled_data = read_json_array(os.path.join(unlabeled_file_path,unlabeled_file_name))
        save_json(unlabeled_data,smerp_unlabeled_file)

    print("Number of unlabeled tweets: ",len(unlabeled_data))

    result_all = OrderedDict()
    labelled_train_new = train
    part_size = 5000
    threshold = 0.8
    max_add_portion = 0.10
    part = int(math.ceil(len(unlabeled_data) / part_size))
    result_old = 0.0
    print("Processing ",part_size," tweets in each iteration. Need to iterate ",part," times.")
    for i in range(part):
        print("----------------------------------------------------------------------------------------")
        test_size = part_size / len(unlabeled_data)
        if test_size <1.0:
            unlabeled_data,test_unlabeled=split_data(unlabeled_data,test_size)
        else: # Number of available tweets are less than part size
            test_unlabeled = unlabeled_data
        print("Iteration: ",int(i)," Processing ",len(test_unlabeled)," tweets...")
        print("Training data length: ",len(labelled_train_new))
        print("Count class portions in Training data : ",count_class([val["classes"] for id,val in labelled_train_new.items()]))

        train_tfidf_matrix_1,test_unlabeled_tfidf_matrix_1=create_tf_idf(labelled_train_new,test_unlabeled,1)

        print('train_tfidf_matrix_1 shape: ',train_tfidf_matrix_1.shape)
        print('test_unlabeled_tfidf_matrix_1 shape: ',test_unlabeled_tfidf_matrix_1.shape)

        train_tfidf_matrix_1 = train_tfidf_matrix_1.todense()
        test_unlabeled_tfidf_matrix_1 = test_unlabeled_tfidf_matrix_1.todense()

        features(labelled_train_new,test_unlabeled)

        train_tf_idf1_manual=add_features_matrix(labelled_train_new,train_tfidf_matrix_1)
        test_unlabeled_tf_idf1_manual=add_features_matrix(test_unlabeled,test_unlabeled_tfidf_matrix_1)

        result,unlabeled_pred,unlabeled_proba = supervised(labelled_train_new,test_unlabeled,train_tf_idf1_manual,test_unlabeled_tf_idf1_manual,grid=False)

        old_len = len(labelled_train_new)

        labelled_train_new = max_add_unlabeled(labelled_train_new,test_unlabeled,unlabeled_pred,unlabeled_proba,max_add_portion=max_add_portion,threshold=threshold,iter=i)

        ## Validation result
        result_now_param = test_f1(labelled_train_new,test,i)
        save_json(result_now_param,"result_now_param_"+str(i),tag=True)
        result_append={}
        result_append[i] = result_now_param
        write_file(json.dumps(result_append,indent=4),'SVM_grid_all_appended',mode='a',tag=True)
        if result_now_param["SVM_grid"]["report"]["f1_macro"] < result_old:
            text="Warning: Current f1_macro is lower than last run, now:",result_now_param["SVM_grid"]["report"]["f1_macro"],"previously:",result_old
            print(text)
            write_file(text,'SVM_grid_all_appended',mode='a',tag=True)
        result_old = result_now_param["SVM_grid"]["report"]["f1_macro"]

        ## Increase threshold value by 0.02 in each iteration until 0.96 is reached.
        # if threshold <= 0.96:
            # threshold = threshold + 0.02

        ## If no new data were added, stop loop.
        if old_len == len(labelled_train_new):
            text="Warning: No new tweets were added to the test set in iteration",i,"Current training data size: ",len(labelled_train_new),",previously:",old_len
            print(text)
            write_file(text,'SVM_grid_all_appended',mode='a',tag=True)
            break

    # save_json(result_all,"result_all",tag=True)

if __name__ == "__main__": main()