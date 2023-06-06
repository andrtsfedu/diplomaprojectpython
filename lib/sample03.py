import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Загрузка данных
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Предобработка данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Определение модели CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Обучение модели
model.fit(x_train[..., tf.newaxis], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(x_test[..., tf.newaxis], y_test, verbose=2)
print('Точность модели на тестовых данных:', test_accuracy)
