import os
import re
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from sklearn.metrics import classification_report

# Шаг 1: Предварительная обработка кода
def preprocess_code(code):
    # Удаление комментариев
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)

    # Удаление импортов
    code = re.sub(r'^\s*import.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*from.*$', '', code, flags=re.MULTILINE)

    # Удаление пустых строк
    code = re.sub(r'^\s*$', '', code, flags=re.MULTILINE)

    # Удаление строковых литералов
    code = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', '', code)

    # Удаление многоточий
    code = re.sub(r'\.\.\.', '', code)

    # Удаление пробельных символов
    code = code.strip()

    return code


# Шаг 2: Токенизация кода
def tokenize_code(code):
    code = preprocess_code(code)
    tokens = code.split()
    return tokens


# Шаг 3: Подготовка данных для обучения
def pad_data(data, max_sequence_length):
    padded_data = pad_sequences(data, maxlen=max_sequence_length, padding='post')
    return padded_data


# Шаг 4: Создание модели
def create_model(vocab_size, embedding_dim, num_filters, filter_size, hidden_dim, max_sequence_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    model.add(Conv1D(num_filters, filter_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


# Шаг 5: Обучение модели
def train_model(train_dir, vocab_size, embedding_dim, num_filters, filter_size, hidden_dim, max_sequence_length):
    train_data = []
    for file_name in os.listdir(train_dir):
        if file_name.endswith('.py'):
            file_path = os.path.join(train_dir, file_name)
            with open(file_path, 'r') as file:
                code = file.read()
                preprocessed_code = preprocess_code(code)
                tokenized_code = tokenize_code(preprocessed_code)
                train_data.append((tokenized_code, 1))

    X_train = [data[0] for data in train_data]
    y_train = [data[1] for data in train_data]

    if len(X_train) > 0 and len(X_train[0]) >= filter_size:
        X_train = pad_data(X_train, max_sequence_length)
        model = create_model(vocab_size, embedding_dim, num_filters, filter_size, hidden_dim, max_sequence_length)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=16)

        # Вычисление метрик на обучающем наборе
        train_predictions = model.predict(X_train)
        train_predictions = [1 if pred >= 0.5 else 0 for pred in train_predictions]

        train_report = classification_report(y_train, train_predictions)
        print("Метрики на обучающем наборе:")
        print(train_report)

        return model
    else:
        print("Размер входных данных меньше размера фильтра свертки. Обучение модели невозможно.")
        return None


# Шаг 6: Поиск плагиата в файле
def find_plagiarized_lines(file_path, model, vocab_size, embedding_dim, max_sequence_length):
    with open(file_path, 'r') as file:
        code = file.read()
        preprocessed_code = preprocess_code(code)
        tokenized_code = tokenize_code(preprocessed_code)

        X_test = pad_data([tokenized_code], max_sequence_length)
        predictions = model.predict(X_test)[0]
        plagiarized_lines = [line for line, pred in zip(tokenized_code, predictions) if pred >= 0.5]

        return plagiarized_lines


def main():
    train_dir = 'D:/lib'
    test_dir = 'D:/analyze/'

    vocab_size = 10000
    embedding_dim = 100
    num_filters = 128
    filter_size = 3
    hidden_dim = 64
    max_sequence_length = 100

    model = train_model(train_dir, vocab_size, embedding_dim, num_filters, filter_size, hidden_dim, max_sequence_length)

    if model:
        for file_name in os.listdir(test_dir):
            if file_name.endswith('.py'):
                file_path = os.path.join(test_dir, file_name)
                plagiarized_lines = find_plagiarized_lines(file_path, model, vocab_size, embedding_dim, max_sequence_length)
                if plagiarized_lines:
                    print(f"Плагиатные строки в файле {file_name}:")
                    for line in plagiarized_lines:
                        print(line)
                    print()


if __name__ == '__main__':
    main()
