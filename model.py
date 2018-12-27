def baseline_model():
	model = Sequential()
	model.add(Dense(5000, input_shape=(len(bow[0]), ), activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(len(dummy_y[0]), activation='softmax'))

	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model