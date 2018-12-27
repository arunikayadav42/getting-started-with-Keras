from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from keras.models import Sequential
from .tobow import tobow

np.random.seed(7)

model_path = "classifier.json"
weights_path = "weights.h5"
encoder_path = "encoder.npy"	

training_data = "data/training/training-data.csv"
training_label = "data/training/training-label.csv"

X_df = pd.read_csv(training_data, header=None)
X = X_df.values
Y_df = pd.read_csv(training_label, header=None)
Y = Y_df.values

dummy_x = []
for text in X:
	dummy_x.append(np.array(tobow(text[0])[0]))

bow = np.array(dummy_x)

#encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
np.save(encoder_path, encoder.classes_)
encoded_Y = encoder.transform(Y)

#convert integers to one hot encoded variables
dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():
	model = Sequential()
	model.add(Dense(5000, input_shape=(len(bow[0]), ), activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(len(dummy_y[0]), activation='softmax'))

	model.summary()
	model.compile(loss='categorical_crossentropy',
		          optimizer='adam',
		          metrics=['accuracy'])
	return model


model = baseline_model()

#train
model.fit(bow, dummy_y, epochs=30)

#save to json
model_json = model.to_json()
with open(model_path, "w") as json_file:
	json_file.write(model_json)

model.save_weights(weights_path)

print("Model has been saved")