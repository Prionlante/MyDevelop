
from Model import get_medel
import random
import json
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

class NLPneural():
    # Конструктор
    def __init__(self, intents, model_name="modelNLP"):
        self.intents = intents
        self.model_name = model_name

        if intents.endswith(".json"):
            self.load_from_json(intents)

        self.lemmatizer = WordNetLemmatizer()

    # Загрузка датасета из json
    def load_from_json(self, intents):
        self.intents = json.loads(open(intents, encoding='utf-8').read())

    # Обучение модели
    def train_model(self, epoh = 250, batchSZ = 5, log = True):

        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        # Разбиваем датасет на обучающие слова (word)
        # выходной результат (intent['tag'])
        # записываем обочающие слова и выходной резултат в documents
        for intent in self.intents['context']:
            for pattern in intent['examples']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_data = list(training[:, 0])
        train_lables = list(training[:, 1])

        # для тестирования
        _, test_data, __, test_lables = train_test_split(train_data, train_lables, test_size=0.05, random_state=415)

        self.model = get_medel(len(train_data[0]), len(train_lables[0]))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

        if(log): print(documents)

        self.hist = self.model.fit(np.array(train_data), np.array(train_lables),
                                   validation_data=(test_data, test_lables),
                                   epochs=epoh, batch_size=batchSZ, verbose=1)
        if(log):
            # Оценка сетки
            print('\n# Оцениваем на тестовых данных')
            results = self.model.evaluate(test_data, test_lables)
            print('\ntest loss:', (round(results[0] * 100, 1)))
            print('test acc:', round(results[1] * 100, 1))

            # (Опцианально) Вывод графиков для мониторинга
            N = np.arange(0, epoh)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.hist.history["loss"], label="train_loss")
            plt.plot(N, self.hist.history["categorical_accuracy"], label="train_acc")
            plt.plot(N, self.hist.history["val_loss"], label="val_loss")
            plt.plot(N, self.hist.history["val_categorical_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()

    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"C:\\Users\Dark\\Desktop\\Save\\{self.model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'C:\\Users\Dark\\Desktop\\Save\\{self.model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'C:\\Users\Dark\\Desktop\\Save\\{self.model_name}_classes.pkl', 'wb'))
        else:
            self.model.save(f"C:\\Users\Dark\\Desktop\\Save\\{model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'C:\\Users\Dark\\Desktop\\Save\\{model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'C:\\Users\Dark\\Desktop\\Save\\{model_name}_classes.pkl', 'wb'))

    def load_model(self, model_name=None):
        if model_name is None:
            self.words = pickle.load(open(f'C:\\Users\Dark\\Desktop\\Save\\{self.model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'C:\\Users\Dark\\Desktop\\Save\\{self.model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'C:\\Users\Dark\\Desktop\\Save\\{self.model_name}.h5')
        else:
            self.words = pickle.load(open(f'C:\\Users\Dark\\Desktop\\Save\\{model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'C:\\Users\Dark\\Desktop\\Save\\{model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'C:\\Users\Dark\\Desktop\\Save\\{model_name}.h5')


    def bag_of_words(self, sentence, words):

        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]

        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1

        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, ints, intents_json):
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['context']
            for i in list_of_intents:
                if i['tag']  == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "Я не пониаю!"
        return result

    def request(self, message):
        ints = self.predict_class(message)
        return self.get_response(ints, self.intents)
