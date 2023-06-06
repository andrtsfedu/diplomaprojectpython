def read_notes():
    with open("notes.txt", "r") as file:
        notes = file.read()
    return notes

def write_notes(notes):
    with open("notes.txt", "w") as file:
        file.write(notes)

def add_note():
    note = input("Введите заметку: ")
    existing_notes = read_notes()
    updated_notes = existing_notes + note + "\n"
    write_notes(updated_notes)
    print("Заметка добавлена!")

def view_notes():
    notes = read_notes()
    print("Ваши заметки:")
    print(notes)

def clear_notes():
    confirmation = input("Вы уверены, что хотите очистить все заметки? (да/нет): ")
    if confirmation.lower() == "да":
        write_notes("")
        print("Все заметки удалены!")
    else:
        print("Очистка заметок отменена.")

def main():
    print("Добро пожаловать в простой блокнот!")

    while True:
        print("\nВыберите действие:")
        print("1. Просмотреть заметки")
        print("2. Добавить заметку")
        print("3. Очистить заметки")
        print("4. Выйти")

        choice = input("Введите номер действия: ")

        if choice == "1":
            view_notes()
        elif choice == "2":
            add_note()
        elif choice == "3":
            clear_notes()
        elif choice == "4":
            break
        else:
            print("Некорректный выбор. Попробуйте еще раз.")

    print("Спасибо за использование блокнота. До свидания!")

if __name__ == "__main__":
    main()
