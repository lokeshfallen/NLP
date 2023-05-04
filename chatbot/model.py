import numpy as np
import tflearn
from tflearn import metrics
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
from MonitorCallback import MonitorCallback
import matplotlib.pyplot as plt


class create_model:
	
	def __init__(self, train, output, tags, all_questions_words):
		acc = metrics.Accuracy(name="acc")
		r2 = metrics.R2("r2_error")
		tf.reset_default_graph()
		self.tags = tags
		self.words = all_questions_words
		self.network = tflearn.input_data(shape=[None,len(train[0])])
		self.network = tflearn.fully_connected(self.network, 8)
		self.network = tflearn.fully_connected(self.network, 8)
		self.network = tflearn.fully_connected(self.network, 8)
		self.network = tflearn.fully_connected(self.network, 8)
		self.network = tflearn.fully_connected(self.network, len(output[0]), activation= "softmax")
		self.network = tflearn.regression(self.network, metric=acc)	
		self.model = tflearn.DNN(self.network, tensorboard_dir=r'C:\Users\Kumar\Documents\SEM4\ML_AI\Project\Chatbot-cum-voice-Assistant\tf_learn_logs',tensorboard_verbose=0)

	def fit_model(self, train, output, n=400, batch = 8, metric=True):
		self.monitorCallback = MonitorCallback()
		self.model.fit(train, output, n_epoch = n, batch_size=batch, show_metric=metric)
		self.model.save("chatbot.tfl")
	def input_words(self, sentence):
		bag_of_words = [0 for _ in range(len(self.words))]
		stemmer = LancasterStemmer()
		sentence_words = nltk.word_tokenize(sentence)
		sentence_words = [stemmer.stem(w.lower()) for w in sentence_words]

		for s in sentence_words:
			for i,j in enumerate(self.words):
				if j == s:
					bag_of_words[i] = 1

		return np.array(bag_of_words)

	def predict_tag(self, sentence):
		results = self.model.predict([self.input_words(sentence)])
		# tag = self.tags[np.argmax(results)]
		return np.argmax(results)

	def get_tags(self):
		return self.tags








