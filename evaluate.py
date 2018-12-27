import pandas as pd
import numpy as np
from tobow import tobow
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.models import model_from_json

def evaluate(model_path, weights_path, test_data_path, test_label_path, encoder_path):
    X_test = pd.read_csv(test_data_path, header=None)
    Y_test = pd.read_csv(test_label_path, header=None)
    test_data = X_test.values
    test_label = Y_test.values

    #load class encoder
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder_path)
    json_file = open(model_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()

    #load classifier model
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    #classification process
    labels = []
    for sentence in test_data:
        vector = tobow(sentence[0])
        predict = model.predict(np.array([vector[0]]))
        index = np.argmax(predict[0])
        labels.append(encoder.classes_[index])

    #evaluate accuracy
    correct = 0
    for i in range(len(labels)):
        print(test_label[i]+"===>"+labels[i])
        if test_label[i] == labels[i]:
            correct += 1
    print("correct"+str(correct)+" from "+str(len(test_data)))
    print("Accuracy: "+str(float(correct)/len(test_data)*100))
    return correct/len(test_data)

model_path = "classifier.json"
weights_path = "weights.h5"
encoder_path = "encoder.npy"

test_data_path = "data/test/test-data.csv"
test_label_path = "data/test/test-label.csv"
evaluate(model_path,weights_path,test_data_path,test_label_path,encoder_path)
