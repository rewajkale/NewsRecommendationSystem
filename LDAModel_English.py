# -*- coding: utf-8 -*-
from __future__ import division
from gensim import corpora, models, parsing
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from glob import glob
import warnings
import os,sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re,string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

class LDAModel:
	def __init__(self,path_to_corpora):
		## Built-in dictionary for word-parser, and path to corpora
		self.stopword = stopwords.words('english')
		self.path_to_corpora = path_to_corpora
		warnings.filterwarnings("ignore")
		print 'Initialize LDAModel....path to corpora : ',path_to_corpora

		## Hyperparameters for training model
		# Minimun length of single document
		self.min_length = 200
		# Num_topics in LDA
		self.num_topics = 90
		# Filter out tokens that appear in less than `no_below` documents (absolute number)
		self.no_below_this_number = 50
		# Filter out tokens that appear in more than `no_above` documents (fraction of total corpus size, *not* absolute number).
		self.no_above_fraction_of_doc = 0.2
		# Remove topic which weights less than this number
		self.remove_topic_so_less = 0.05
		# Number of iterations in training LDA model, the less the documents in total, the more the iterations for LDA model to converge
		self.num_of_iterations = 1000
		# Number of passes in the model
		self.passes = 3
		#Print all hyperparameters
		parameters = {}
		parameters['min_length'] = self.min_length
		parameters['num_topics'] = self.num_topics
		parameters['no_below_this_number'] = self.no_below_this_number
		parameters['no_above_fraction_of_doc'] = self.no_above_fraction_of_doc
		parameters['remove_topic_so_less'] = self.remove_topic_so_less
		parameters['num_of_iterations'] = self.num_of_iterations
		parameters['passes'] = self.passes
		for k in parameters:
		    print "Parameter for {0} is {1}".format(k,parameters[k])
		print 'Finished initializing....'

	def __tokenizeWholeCorpora(self,pathToCorpora):
	    print 'Start tokenzing the corpora: %s' % (pathToCorpora)
	    punct = re.compile('[%s]' % re.escape(string.punctuation))
	    wnl = WordNetLemmatizer()
	    doc_count=0
	    train_set = []
	    doc_mapping = {}
	    link_mapping = {}

	    for f in glob(pathToCorpora+'/*'):
	            filereader = open(f, 'r')
	            article = filereader.readlines();filereader.close()
	            text = ''
	            try:
	            	link = article[0]
	            	title = article[1]
	            	text = article[2].lower()
	            except IndexError:
	            	continue

	            # Skip document length < min_length
	            if len(text) < self.min_length:
	                continue
	            text = punct.sub("",text)  # Remove all punctuations
	            tokens = nltk.word_tokenize(text)  # Tokenize the whole text
	            # Lemmatize every word and add to tokens list if the word is not in stopword
	            train_set.append([wnl.lemmatize(word) for word in tokens if word not in self.stopword]) 
	            # Build doc-mapping
	            doc_mapping[doc_count] = title
	            link_mapping[doc_count] = link
	            doc_count = doc_count+1
	            if doc_count % 10000 == 0:
	            	print 'Have processed %i documents' % (doc_count)

	    print 'Finished tokenzing the copora: %s' % (pathToCorpora)
	    return doc_count,train_set,doc_mapping,link_mapping

	def __convertListToDict(self,anylist):
	    '''
	    This code snippet could be easily done by one-liner dict comprehension:
	    {key:value for key,value in anylist}
	    '''
	    convertedDict = {}
	    for pair in anylist:
	        topic = pair[0]
	        weight = pair[1]
	        convertedDict[topic] = weight
	    return convertedDict
	    
	def __savePickleFile(self,fileName,objectName):
	    '''
  	    Serialize objects into pickle files 
	    '''		
	    fileName= './LDAmodel/'+fileName+'.pickle'
	    mappingFile = open(fileName,'w')
	    pickle.dump(objectName,mappingFile)
	    mappingFile.close()

	def saveModel(self,lda,doc_mapping,link_mapping,corpus):
		'''
		Saving models and maps for later use

		:param lda: the LDA model
		:param doc_mapping: index-document mapping
		:param link_mapping: index-link mapping
		:param corpus: the whole corpus in list[list[tokens]]
		'''
		print 'Start saving LDA models & maps....'
		# Save model output
		save_path = './LDAmodel/final_ldamodel'
		lda.save(save_path)
		print 'Model saved at {0}'.format(save_path)
		# Save the whole corpus
		save_path = 'corpus'
		self.__savePickleFile(save_path,corpus)
		print 'Corpus saved at {0}'.format(save_path)
		# Save index to document mapping
		save_path = 'documentmapping'
		self.__savePickleFile(save_path,doc_mapping)
		print 'Document mapping saved at {0}'.format(save_path)
		# Save index to link mapping
		save_path = 'linkmapping'
		self.__savePickleFile(save_path,link_mapping)
		print 'Link mapping saved at {0}'.format(save_path)
		# Save doc to topic matrix
		doc_topic_matrix = {}
		count = 0
		for doc in corpus:
		    dense_vector = {}
		    vector = self.__convertListToDict(lda[doc])
		    # remove topic that is so irrelevant
		    for topic in vector:
		        if vector[topic] > self.remove_topic_so_less:
		            dense_vector[topic] = vector[topic]
		    doc_topic_matrix[count]=dense_vector
		    count = count+1
		save_path = 'doc_topic_matrix'
		self.__savePickleFile(save_path,doc_topic_matrix)
		print 'doc to topic mapping saved at {0}'.format(save_path)

		print 'Finished saving LDA models & maps....' 	

	def trainModel(self):
		'''
		Train a LDA model, inclusive of 4 steps:
		1. Parse the whole corpora into unigram token collections and document mapping (for later use)
		2. Filter tokens which are not common (no_below_this_number), and too common (no_above_fraction_of_doc)
		3. Indexing the token collections and do TF-IDF transformation
		4. Call gensim.models.LdaModel and generate topic distributions of the corpora
		'''
		print 'Start preparing unigram tokens....'		
		## Start of preparing list of documents and tokens [[words_in_1st_doc],[words_in_2nd_doc]....], which comprise Bag-Of-Words (BOW)
		# Get document_count, tokens, and document-index mapping from the corpora
		doc_count,train_set,doc_mapping,link_mapping = self.__tokenizeWholeCorpora(path_corpora) 
		# Put the training data into gensim.corpora for later use
		dic = corpora.Dictionary(train_set) 
		denominator = len(dic)
		# Filtering infrequent words & common stopwords, thus reducing the dimension of terms (which prevents curse of dimensionality)
		dic.filter_extremes(no_below=self.no_below_this_number, no_above=self.no_above_fraction_of_doc)
		nominator = len(dic)
		corpus = [dic.doc2bow(text) for text in train_set]  # transform every token into BOW
		print 'There are %i documents in the pool' % (doc_count)
		print "In the corpus there are ", denominator, " raw tokens"
		print "After filtering, in the corpus there are", nominator, "unique tokens, reduced ", (1-(nominator/denominator)),"%"
		print 'Finished preparing unigram tokens....'	
		##END 

		print 'Start training LDA model....'
		## Implementing TF-IDF as a vector for each document, and train LDA model on top of that
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		lda = models.LdaModel(corpus_tfidf, id2word = dic, num_topics = self.num_topics,iterations=self.num_of_iterations,passes = self.passes)
		corpus_lda = lda[corpus_tfidf]
		# Once done training, print all the topics and related words
		print 'Finished training LDA model.......Here is the list of all topics & their most frequent words' 	
		for i in range(self.num_topics):
		    print 'Topic %s : ' % (str(i)) + lda.print_topic(i)
		# Exhibit perplexity of current model under specific topic hyperparameter : k. The lower the better
		print '==============================='
		print 'Model perplexity : ',lda.bound(corpus_lda),' when topic k =', str(self.num_topics)
		print '==============================='   
		
		return lda,doc_mapping,link_mapping,corpus		

if __name__ == '__main__':
	def parseArgs(argv=None): 
	    '''Command line options.
	    '''
	    if argv is None:
	        argv = sys.argv
	    else:
	        sys.argv.extend(argv)
	 
	    parser = ArgumentParser(description="LDAModel", formatter_class=RawDescriptionHelpFormatter)
	    parser.add_argument("-i","--dir", dest="directory", help="Directory to which articles stored", required=True)
	    args = parser.parse_args()
	    directory = args.directory
	   
	    return directory

	path_corpora = parseArgs()  # parse the path to corpora
	LDAmodel = LDAModel(path_corpora)  # instantiate the LDAModel class
	lda,doc_mapping,link_mapping,corpus = LDAmodel.trainModel()  # train a LDA model using the assgined corpora
	LDAmodel.saveModel(lda,doc_mapping,link_mapping,corpus)  # save model for recommendations use
