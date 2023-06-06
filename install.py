import os
import shutil
import subprocess

# Установка необходимых пакетов
def install_packages():
    packages = ['tensorflow', 'numpy', 'scikit-learn']  # Здесь перечислите все необходимые пакеты
    for package in packages:
        subprocess.call(['pip', 'install', package])

# Загрузка обучающих и тестовых файлов
def load_files(train_dir, test_dir):
    # Создание директорий, если они не существуют
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

# Основная функция
def main():
    train_dir = 'path/to/train_directory'  # укажи путь к обучающей директории
    test_dir = 'path/to/test_directory'  # укажи путь к тестовой директории

    # Установка необходимых пакетов
    install_packages()

    # Загрузка файлов
    load_files(train_dir, test_dir)

    # Запуск программы
    # Вставьте сюда код для запуска вашей программы

# Вызов основной функции
if __name__ == '__main__':
    main()
