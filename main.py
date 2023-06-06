import json
import pickle
import random

import nltk
import numpy
# import speech_recognition as sr
# from nltk.stem import LancasterStemmer
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from nlp_id.stopword import StopWord
from nlp_id.lemmatizer import Lemmatizer  # library pendeteksi kata yang berhimbuan awalan kata dan akhiran kata
from nlp_id.tokenizer import Tokenizer  # pemisahan kalimat yang akan diformat bentuk tag dengan acuan tiap spasi (per character)
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential  # library pelatihan menggunakan metode sequential
from tensorflow.python.keras.models import model_from_yaml  # library perubahan format data training ke yaml

nltk.download('punkt')

# stemmer = LancasterStemmer()
stopwords = StopWord()
lemmatizer = Lemmatizer()  # pemanggilan fungsi lemmatizer
tokenizer = Tokenizer()  # pemanggilan fungsi tokenizer

with open("intents.json") as file:
    data = json.load(file)

delete_word = ['?', '(', ')', ',', '!', '@', '$', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
               '']

try:
    with open("chatbot.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # mengambil kata dan tokenize (pecah)
            wrds = tokenizer.tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        # jika label diluar list
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stopwords.remove_stopword(ws.lower()) for ws in words]
    words = [lemmatizer.lemmatize(w) for w in words if w not in delete_word]

    words = sorted(list(set(words)))
    labels = sorted(list(set(labels)))

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    random.shuffle(training)
    random.shuffle(output)
    training = numpy.array(training)
    output = numpy.array(output)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

try:
    yaml_file = open('chatbotmodel.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    myChatModel = model_from_yaml(loaded_model_yaml)
    myChatModel.load_weights("chatbotmodel.h5")
    print("Loaded model from disk")

except:
    # pembuatan layer neural network

    myChatModel = Sequential()
    myChatModel.add(Dense(128, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dense(64, activation='relu'))
    myChatModel.add(Dropout(0.5))
    myChatModel.add(Dense(len(labels), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # optimize model
    myChatModel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    epochs = 1000
    # train model
    myChatModel.summary()
    history = myChatModel.fit(training, output, epochs=epochs, batch_size=15, verbose=1)

    # ploting data
    epochs_range = range(epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], color='g', label='Accuracy')
    # plt.ylim(ymax=1)
    plt.legend(loc='upper right')
    plt.title('Training Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history.history['loss'], color='r', label='Loss')
    plt.legend(loc='upper right')
    # plt.ylim(ymax=1)
    plt.title('Training Loss')
    plt.show()

    # serialize model to yaml and save it to disk
    model_yaml = myChatModel.to_yaml()
    with open("chatbotmodel.yaml", "w") as y_file:
        y_file.write(model_yaml)

    # serialize weights to HDF5
    myChatModel.save_weights("chatbotmodel.h5")
    print("Saved model from disk")


def bag_of_words(s, word):
    list_check = 0
    bags = [0 for _ in range(len(word))]
    # pemisahan kalimat mejadi per kata dari input suara
    s_words = tokenizer.tokenize(s)
    s_words = [stopwords.remove_stopword(txt.lower()) for txt in s_words]
    s_words = [lemmatizer.lemmatize(word) for word in s_words if word not in delete_word]
    print(s_words)

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                list_check += 1
                bags[i] = 1

    # kata yang valid dan cocok dengan dataset
    list_error = (((len(s_words)) - list_check) / (len(s_words))) * 100
    list_acc = (list_check / (len(s_words))) * 100
    print("Data Sukses : {:0.2f}%".format(list_acc))
    print("Data Loss : {:0.2f}%".format(list_error))
    return numpy.array(bags), s_words


# input = data dari speech recognition yang akan diuabh menjadi text
# proses = data text akan diolah menggunakan nlp
# output = pencocokan presentase job skill

def chat_with_bot(input_text):
    currentText, wordz = bag_of_words(input_text, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)

    if numpy.all((numpyCurrentText == 0)):
        return "I didn't get that, try again"

    result = myChatModel.predict(numpyCurrentText[0:1])
    result_index = result[-1]
    dataindex = len(result_index)

    presx = []
    respx = []
    countJobs = 0

    for datai in range(dataindex):
        tag = labels[datai]
        # tagdata = result[0][datai]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                patternx = tg['patterns']
                responses = tg['responses']
                wordsx, txtz = classificate(patternx, wordz)
                numpola = len(wordsx)

                print("========================")
                print("Data pattern Raw: {}".format(numpola))
                print("Data pattern Valid: {}".format(txtz))
                if len(responses) > 1:
                    print("Data Responses: {}".format(len(responses)))
                    print(responses)
                    print("========================")
                    for xmax in range(len(responses)):
                        presx.append(((txtz + int(numpola*0.3)) / numpola) * 100)
                        respx.append(responses[xmax])
                        countJobs += 1
                else:
                    print("Data Responses: {}".format(len(responses)))
                    print(responses)
                    print("========================")
                    presx.append(((txtz + int(numpola*0.3)) / numpola) * 100)
                    respx.append(responses[0])
                    countJobs += 1

    return respx, presx, countJobs

    # else:
    # print("I didn't get that, try again")


def classificate(patternsc, wordsc):
    wordx = []
    checkdata = 0

    for patterns in patternsc:
        dataPattern = tokenizer.tokenize(patterns)
        wordx.extend(dataPattern)

    wordx = [stopwords.remove_stopword(text.lower()) for text in wordx]
    wordx = [lemmatizer.lemmatize(txt) for txt in wordx if txt not in delete_word]

    for wc in wordsc:
        if wc in wordx:
            checkdata += 1

    return wordx, checkdata


def chat():
    print("Start talking with the chatbot (try quit to stop)")
    count = 0
    while count < 1:
        print("Pelajaran yang kamu sukai ?")
        while True:
            text = input("You : ")
            if text != "":
                resp, pres, maks = chat_with_bot(text)
                for ada in range(maks):
                    print("Referensi Skill: {} dengan presentasi {:0.2f}".format(resp[ada], pres[ada]))
                break

        print("Pelajaran yang kamu bisa kerjakan ?")
        while True:
            text = input("You : ")
            if text != "":
                resp, pres, maks = chat_with_bot(text)
                for ada in range(maks):
                    print("Referensi Skill: {} dengan presentasi {:0.2f}".format(resp[ada], pres[ada]))
                break

        print("Apa pelajaran yang kamu tidak sebarapa kamu kuasai ?")
        while True:
            text = input("You : ")
            if text != "":
                resp, pres, maks = chat_with_bot(text)
                for ada in range(maks):
                    print("Referensi Skill: {} dengan presentasi {:0.2f}".format(resp[ada], pres[ada]))
                break

        print("Apa pelajaran yang kamu tidak sukai ?")
        while True:
            text = input("You : ")
            if text != "":
                resp, pres, maks = chat_with_bot(text)
                for ada in range(maks):
                    print("Referensi Skill: {} dengan presentasi {:0.2f}".format(resp[ada], pres[ada]))
                break

        count += 1


# chat()
