import requests
import json

def get_weather(city):
    api_key = "YOUR_API_KEY"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = json.loads(response.text)

    if data["cod"] == "404":
        return None

    weather_description = data["weather"][0]["description"]
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]

    weather_info = {
        "description": weather_description,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed
    }

    return weather_info

def main():
    print("Прогноз погоды")

    while True:
        city = input("Введите название города (или 'выход' для завершения): ")

        if city.lower() == "выход":
            break

        weather = get_weather(city)

        if weather:
            print("\nПогода в городе", city)
            print("Описание:", weather["description"])
            print("Температура:", weather["temperature"], "°C")
            print("Влажность:", weather["humidity"], "%")
            print("Скорость ветра:", weather["wind_speed"], "м/с")
        else:
            print("Город не найден. Попробуйте еще раз.")

    print("Спасибо за использование виджета погоды. До свидания!")

if __name__ == "__main__":
    main()
