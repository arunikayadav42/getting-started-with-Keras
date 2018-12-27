training_data = "data/training/training-data.csv"
training_label = "data/training/training-label.csv"

X_df = pd.read_csv(training_data, header=None)
X = X_df.values
Y_df = pd.read_csv(training_label, header=None)
Y = Y_df.values

dummy_x = []
for test in X:
	dummy_x.append(np.array(tobow(text[0])[0]))

bow = np.array(dummy_x)

encoder = LabelEncoder()
encoder.fit(Y)
np.save(encoder_path, encoder.classes_)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)