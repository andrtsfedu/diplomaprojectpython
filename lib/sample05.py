import random

def guess_number():
    secret_number = random.randint(1, 100)
    attempts = 0

    print("Привет! Я загадал число от 1 до 100.")
    print("Попробуй угадать!")

    while True:
        try:
            guess = int(input("Введи число: "))
        except ValueError:
            print("Некорректный ввод. Попробуй еще раз.")
            continue

        attempts += 1

        if guess < secret_number:
            print("Загаданное число больше.")
        elif guess > secret_number:
            print("Загаданное число меньше.")
        else:
            print("Поздравляю! Ты угадал число за", attempts, "попыток.")
            break

    play_again = input("Хочешь сыграть еще? (да/нет): ")
    if play_again.lower() == "да":
        guess_number()
    else:
        print("Спасибо за игру! Пока!")

guess_number()
