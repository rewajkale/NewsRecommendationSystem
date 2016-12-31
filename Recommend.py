# -*- coding: utf-8 -*-
import logging

from gensim.models import LdaModel
from gensim import corpora
from utils.utils import *
import pickle
import argparse
import matplotlib
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
from flask import Flask, json,url_for,render_template, flash, redirect, session, url_for, request, g
from flask_login import login_user, logout_user, current_user, login_required
import requests
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import httplib2
import random
from itertools import repeat

indices = [];
app = Flask(__name__)
class Predict():
    def __init__(self):
        # current_working_dir = '/home/etu/eason/nodejs/Semantic_Aware_RecSys'
        current_working_dir = '.'
        os.chdir(current_working_dir)
        lda_model_path = "./LDAmodel/final_ldamodel"

        self.lda = LdaModel.load(lda_model_path)
        self.no_of_recommendation = 10
        self.omit_topic_below_this_fraction = 0.1
        self.mapping = self.__init_mapping()
        self.linkMapping = self.__init_Link_mapping()
        self.doc_topic_matrix = loadPickleFile('doc_topic_matrix')
    
    def __init_mapping(self):
        path_mappingfile= './LDAmodel/documentmapping.pickle'
        mappingFile = open(path_mappingfile,'r')
        mapping = pickle.load(mappingFile)
        mappingFile.close()

        return mapping

    def __init_Link_mapping(self):
        path_mappingfile= './LDAmodel/linkmapping.pickle'

        if os.path.isfile(path_mappingfile):
            mappingFile = open(path_mappingfile,'r')
            mapping = pickle.load(mappingFile)
            mappingFile.close()
            return mapping
        else:
            return {}
    
    def constructDocToTopicMatrix(self,lda,corpus):
        '''
        This code snippet could be easily done by one-liner dict comprehension:
        {key:value for key,value in anylist}
        '''        
        doc_topic_matrix = {}
        count = 0
        for doc in corpus:
            if len(doc)>0:
                count = count+1
                vector = convertListToDict(lda[doc])
                doc_topic_matrix[count]=vector
        return doc_topic_matrix
    
    def constructUserToTopicMatrix(self,user_dict,verbose=False):
        """ Construct user-topic vector(dictionary)
        args:
            user_dict: a dictionary of user-doc and doc-topic 
        """
        user_topic_vector = {}
        length = len(user_dict)
        for seen_doc in user_dict:
            for seen_topic in user_dict[seen_doc]:
                weight = user_dict[seen_doc][seen_topic]
                if user_topic_vector.has_key(seen_topic):
                    current_weight = user_topic_vector[seen_topic]
                    current_weight = current_weight + weight/length
                    user_topic_vector[seen_topic] = current_weight
                else:
                    user_topic_vector[seen_topic] = weight/length
        
        # Remove topic less than weight : omit_topic_below_this_fraction/2
        lightweight_user_topic_vector = {}
        for k,v in user_topic_vector.iteritems():
            if v > self.omit_topic_below_this_fraction/2:
                lightweight_user_topic_vector[k] = v
        
        denominator = sum(lightweight_user_topic_vector.values())

        for topic in lightweight_user_topic_vector:
            lightweight_user_topic_vector[topic] = lightweight_user_topic_vector[topic] / denominator
            
        if verbose:    
            print 'Topic distribution for current user : {0}'.format(lightweight_user_topic_vector)
            print 'Normalized topic distribution for current user : {0}'.format(lightweight_user_topic_vector)
        
        return lightweight_user_topic_vector
    
    
    def getLink(self,sort,no_of_recommendation):
        for i in sort.keys()[:no_of_recommendation]:
            print 'Recommend document: {0} '.format(self.mapping[i])        
    
    def run(self, user_dict,verbose=False):
        '''
        Get recommendations from the user_dict which describes the topic distribution attibutes to a user/item 
        If verbose = True, return the result in a verbose way.
        '''        
        user_topic_matrix = self.constructUserToTopicMatrix(user_dict,verbose)
        recommend_dict = {}
        
        # Pearson correlation appears to be the most precise 'distance' metric in this case
        for doc in self.doc_topic_matrix:
            #sim = cossim(user_topic_matrix,doc_topic_matrix[doc])  # cosine similarity
            #sim = KLDbasedSim(user_topic_matrix,doc_topic_matrix[doc])  # KLD similarity
            sim = pearson_correlation(user_topic_matrix,self.doc_topic_matrix[doc],self.lda.num_topics)
            if sim > 0.7 and doc not in user_dict.keys():  # 0.7 is arbitrary, subject to developer's judge
                if verbose:
                    print 'Recommend document {0} of similarity : {1}'.format(doc,sim)
                recommend_dict[doc] = sim
        
        sort = getOrderedDict(recommend_dict)
        recommend_str = (str(sort.keys()[:self.no_of_recommendation])
                        .replace('[','')
                        .replace(']','')
                        )
	titleDict= reduce(lambda x,y:x+';'+y,map(lambda i: self.mapping[int(i)],recommend_str.split(','))).split(';')
	print titleDict
	urlDict = reduce(lambda x,y:x+';'+y,map(lambda i: self.linkMapping[int(i)],recommend_str.split(','))).split(';')
	print urlDict
        if verbose:        
            for title in user_dict:
                    print 'You viewed : {0}'.format(self.mapping[title])
            self.getLink(sort,self.no_of_recommendation)
        else:
            print 'You viewed : [' + reduce(lambda x,y: x+'] & ['+y,map(lambda title:self.mapping[title],user_dict)) +']; Your Recommendations : ;'+ reduce(lambda x,y:x+';'+y,map(lambda i: self.mapping[int(i)],recommend_str.split(','))) + ' &&'+  reduce(lambda x,y:x+';'+y,map(lambda i: self.linkMapping[int(i)],recommend_str.split(',')))
    
    def getNews(self, indices):
	p =0
	d = [[] for i in repeat(None, 10)]
	for i in range(len(indices)):
		d[p].append(indices[i])
		d[p].append(self.mapping[int(indices[i])])
		d[p].append(self.linkMapping[int(indices[i])])
		p = p+1
	return d;

    def getRecommendationList(self, index):
	recommend_dict = {}
	user_dict = {}
	index = int(index)
	path_doc_topic_matrix = './LDAmodel/doc_topic_matrix.pickle'
    	mappingFile = open(path_doc_topic_matrix,'r')
    	doc_topic_matrix = pickle.load(mappingFile)
    	mappingFile.close()
	if matplotlib.cbook.is_numlike(index):
             user_dict[index] = doc_topic_matrix[index]
	print user_dict
	verbose=False
        user_topic_matrix = self.constructUserToTopicMatrix(user_dict,verbose)
        # Pearson correlation appears to be the most precise 'distance' metric in this case
        for doc in self.doc_topic_matrix:
            sim = pearson_correlation(user_topic_matrix,self.doc_topic_matrix[doc],self.lda.num_topics)
            if sim > 0.7 and doc != index:  # 0.7 is arbitrary, subject to developer's judge
                recommend_dict[doc] = sim
        
        sort = getOrderedDict(recommend_dict)
	print sort
        recommend_str = (str(sort.keys()[:self.no_of_recommendation])
                        .replace('[','')
                        .replace(']','')
                        )
	print recommend_str
	lisst = recommend_str.split(",")
	news = [];
	for i in lisst:
	     dic = {}
	     dic['headline'] = self.mapping[int(i)];
             dic['url'] = self.linkMapping[int(i)];
	     news.append(dic);
        return news;
    
    def getMainURL(self, index):
	return self.linkMapping[int(index)]
    def getTitle(self, index):
	return self.mapping[int(index)]
    # def get(self, user_dict):
    #     load_path = './LDAmodel/corpus.pickle'
    #     mappingFile = open(load_path,'r')
    #     corpus = pickle.load(mappingFile)
    #     mappingFile.close()
        
    #     doc_topic_matrix = loadPickleFile('doc_topic_matrix')
    #     user_topic_matrix = self.constructUserToTopicMatrix(user_dict)
    #     recommend_dict = {}
        
    #     for doc in doc_topic_matrix:
    #         #sim = cossim(user_topic_matrix,doc_topic_matrix[doc])  # cosine similarity
    #         #sim = KLDbasedSim(user_topic_matrix,doc_topic_matrix[doc])  # KLD similarity
    #         sim = pearson_correlation(user_topic_matrix,doc_topic_matrix[doc],self.lda.num_topics)
    #         if sim > 0.7 and doc not in user_dict.keys():  # 0.7 is arbitrary, subject to developer's judge
    #             #print 'Recommend document {0} of similarity : {1}'.format(doc,sim)
    #             recommend_dict[doc] = sim
        
    #     return recommend_dict

@app.route('/',methods=['GET', 'POST'])
@app.route('/index')
def index():
    del indices[:]
    p =Predict();
    for i in range(10):
	ind = random.randint(1, 5896)
        indices.append(ind)
    d = p.getNews(indices);
    return render_template('index.html',
                           title='Home',
                           tags=d)

@app.route('/recommendation', methods=['POST'])
def recommendation():
    p =Predict();
    checked = request.form.getlist('channel')
    news = p.getRecommendationList(checked[0])
    from_url = request.form.get('from_url')
    to_url = request.form.get('to_url')
    checked = request.form.getlist('channel')
    password = request.form.get('password')
    to_cell = request.form.get('to_cell')
    print from_url;
    print to_url;
    msg = MIMEMultipart()
    msg['From'] = from_url
    msg['To'] = to_url
    msg['Subject'] = "You would like to read this news"
    body = "The news is: "+p.getTitle(checked[0])+"\nClick on this link:  "+p.getMainURL(checked[0])
    msg.attach(MIMEText(body, 'plain'))
	# Use sms gateway provided by mobile carrier:
	# at&t:     number@mms.att.net
	# t-mobile: number@tmomail.net
	# verizon:  number@vtext.com
	# sprint:   number@page.nextel.com
    try:
		if from_url!="" and to_url!="" and checked[0]!=None:
			server = smtplib.SMTP('smtp.gmail.com', 587)
			server.starttls()
			server.login(from_url, password)
			text = msg.as_string()
			server.sendmail(from_url, to_url, text)
			server.quit()
		if from_url!="" and to_cell!="":
			server = smtplib.SMTP('smtp.gmail.com', 587)
			server.starttls()
			server.login(from_url, password)
			text = msg.as_string()
			server.sendmail( from_url, to_cell+'@tmomail.net', text )
			server.quit()
    except:
		print "Authentication Error";
    return render_template('recommendation.html',tags = checked,dat=news, mainURL = p.getMainURL(checked[0]))

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", default=False, help="Interactive environment for recommending items")
    parser.add_argument("--SBCF", default=0, help="Under development")
    parser.add_argument("--api",default=False,  help="Article # that you want to get recommendation out of")
    args = parser.parse_args()
    predict = Predict()
    
    path_doc_topic_matrix = './LDAmodel/doc_topic_matrix.pickle'
    mappingFile = open(path_doc_topic_matrix,'r')
    doc_topic_matrix = pickle.load(mappingFile)
    mappingFile.close()
    
    if args.api:
        user = {}

        for arg in args.api.split(','):
            arg=int(arg)
            if matplotlib.cbook.is_numlike(arg):
                user[arg] = doc_topic_matrix[arg]  
        
        predict.run(user)
        sys.stdout.flush()

    if args.shell:
        user = {}
        while True:
            try:
                articles = raw_input('What articles you\'ve viewed? : ')
                for arg in articles.split(','):
                    arg=int(arg)
                    if matplotlib.cbook.is_numlike(arg):
                        user[arg] = doc_topic_matrix[arg]  
                    else:
                        print 'you entered NaN : {0}'.format(arg)
            
                    
                predict.run(user,True)
                print '=========================================='

            except KeyboardInterrupt:
                print '\nDone....exiting....'
                sys.exit(1)
    
    ## Under construction
    # if args.SBCF=='1':
    #     from sqlitedict import SqliteDict
    #     mydict = SqliteDict('./my_db.sqlite', autocommit=True)
    #     #my = predict.get(itemProfile)
    #     count = 0
    #     for topic in doc_topic_matrix:
    #         itemProfile = {}
    #         itemProfile[topic] = doc_topic_matrix[topic]    
    #         if not mydict.has_key(topic):
    #             my = predict.get(itemProfile)
    #             mydict[topic] = my
    #         count = count+1
    #         if count % 100 ==0:
    #             print 'Processed ',count,' documents...'
    #     mydict.close()

if __name__ == '__main__':
    main()
