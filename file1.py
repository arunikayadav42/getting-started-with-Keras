from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
import numpy as np
import numbers

corpus_path = 'data/training/training-data.csv'

#prepare training data for bow(corpus)
X_training = []
df = pd.read_csv(corpus_path, header=None)
for i in range(len(df[0])):
	X_training.append(df[0][i])

sentences = np.array(X_training)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences).todense()

for key in vectorizer.vocabulary_.keys():
	value = vectorizer.vocabulary_[key]
	value = int(value)
	vectorizer.vocabulary_[key] = value

with open('vocabulary.json', 'w') as vocabFile:
	json.dump(vectorizer.vocabulary_ , vocabFile)

print("Vocabulary is printed")