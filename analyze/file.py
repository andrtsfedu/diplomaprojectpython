import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Создание модели сверточной нейронной сети
model = keras.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Генерация случайных входных данных
X_train = np.random.random((100, 10, 1))
y_train = np.random.randint(2, size=(100, 1))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Генерация случайных тестовых данных
X_test = np.random.random((10, 10, 1))

# Предсказание с помощью обученной модели
predictions = model.predict(X_test)

# Вывод предсказаний
print(predictions)
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Создание модели сверточной нейронной сети
model = keras.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Генерация случайных входных данных
X_train = np.random.random((100, 10, 1))
y_train = np.random.randint(2, size=(100, 1))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Генерация случайных тестовых данных
X_test = np.random.random((10, 10, 1))

# Предсказание с помощью обученной модели
predictions = model.predict(X_test)

# Вывод предсказаний
print(predictions)
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Создание модели сверточной нейронной сети
model = keras.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Генерация случайных входных данных
X_train = np.random.random((100, 10, 1))
y_train = np.random.randint(2, size=(100, 1))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Генерация случайных тестовых данных
X_test = np.random.random((10, 10, 1))

# Предсказание с помощью обученной модели
predictions = model.predict(X_test)

# Вывод предсказаний
print(predictions)
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Создание модели сверточной нейронной сети
model = keras.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Генерация случайных входных данных
X_train = np.random.random((100, 10, 1))
y_train = np.random.randint(2, size=(100, 1))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Генерация случайных тестовых данных
X_test = np.random.random((10, 10, 1))

# Предсказание с помощью обученной модели
predictions = model.predict(X_test)

# Вывод предсказаний
print(predictions)
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Создание модели сверточной нейронной сети
model = keras.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Генерация случайных входных данных
X_train = np.random.random((100, 10, 1))
y_train = np.random.randint(2, size=(100, 1))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Генерация случайных тестовых данных
X_test = np.random.random((10, 10, 1))

# Предсказание с помощью обученной модели
predictions = model.predict(X_test)

# Вывод предсказаний
print(predictions)
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Создание модели сверточной нейронной сети
model = keras.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Генерация случайных входных данных
X_train = np.random.random((100, 10, 1))
y_train = np.random.randint(2, size=(100, 1))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Генерация случайных тестовых данных
X_test = np.random.random((10, 10, 1))

# Предсказание с помощью обученной модели
predictions = model.predict(X_test)

# Вывод предсказаний
print(predictions)
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Создание модели сверточной нейронной сети
model = keras.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Генерация случайных входных данных
X_train = np.random.random((100, 10, 1))
y_train = np.random.randint(2, size=(100, 1))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Генерация случайных тестовых данных
X_test = np.random.random((10, 10, 1))

# Предсказание с помощью обученной модели
predictions = model.predict(X_test)

# Вывод предсказаний
print(predictions)
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Создание модели сверточной нейронной сети
model = keras.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Генерация случайных входных данных
X_train = np.random.random((100, 10, 1))
y_train = np.random.randint(2, size=(100, 1))

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Генерация случайных тестовых данных
X_test = np.random.random((10, 10, 1))

# Предсказание с помощью обученной модели
predictions = model.predict(X_test)

# Вывод предсказаний
print(predictions)
