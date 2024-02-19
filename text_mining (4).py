import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import speech_recognition as sr


def capture_voice_input():
    with sr.Microphone() as source:
        print("Говоріть...")
        audio = recognizer.listen(source)
    return audio


def convert_voice_to_text(audio):
    try:
        text = recognizer.recognize_google(audio, language="uk-UA")
        print("Ви сказали: " + text)
    except sr.UnknownValueError:
        text = ""
        print("Вибачте, я Вас не розумію.")
    except sr.RequestError as e:
        text = ""
        print("Error; {0}".format(e))
    return text


def process_voice_command(text, product_data):
    if "найдешевший товар" in text.lower():
        cheapest_product = product_data.loc[product_data['price'].idxmin()]
        print("\nНайдешевший товар:")
        print(cheapest_product)
    elif "найдорожчий товар" in text.lower():
        expensive_product = product_data.loc[product_data['price'].idxmax()]
        print("\nНайдорожчий товар:")
        print(expensive_product)
    elif "найбільш популярні товари" in text.lower():
        product_data['likes'] = pd.to_numeric(product_data['likes'], errors='coerce')
        popular_products = product_data[product_data['likes'] > 1]
        print("\nНайбільш популярні товари:")
        print(popular_products)
    elif any(char.isdigit() for char in text):
        indices = [int(i) for i in text.split() if i.isdigit()]
        valid_indices = [idx for idx in indices if 0 <= idx < len(product_data)]
        if valid_indices:
            selected_products = product_data.iloc[valid_indices]
            print("\nВибрані товари:")
            print(selected_products)
        else:
            print("\nІндекси виходять за межі діапазону товарів.")
    elif "вихід" in text.lower():
        print("До побачення! Гарного дня!")
        return True
    else:
        print("Команда не зрозуміла. Будь ласка, спробуйте ще раз.")
    return False


def monitor_product_changes(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }

    r = requests.get(url, headers=headers)

    result_list = {'title': [], 'price': [], 'likes': []}
    print(r.status_code)

    soup = bs(r.text, "html.parser")
    products = soup.find_all('div', class_='b-tile-item__content')

    for product in products:
        title = product.find('a', class_='b-tile-item-name-product js-b-tile-item').text.strip()
        price = product.find('div', class_='b-tile-item__price').text.strip()
        likes = product.find('span', class_='b-tile-item__favorite-inner').text.strip()

        result_list['title'].append(title)
        result_list['price'].append(price)
        result_list['likes'].append(likes)

    print(result_list['title'])
    print(result_list['price'])
    print(result_list['likes'])

    # Завантаження попереднього файлу
    try:
        previous_df = pd.read_excel("product_changes.xlsx")
    except FileNotFoundError:
        previous_df = pd.DataFrame()

    # Додавання всіх товарів до Excel-файлу
    all_products = pd.DataFrame(data=result_list)
    all_products.to_excel("product_changes.xlsx", index=False)
    print("Всі товари додано до Excel-файлу.")

    # Порівняння та виведення нових товарів у консоль
    new_products = all_products[~all_products.isin(previous_df)].dropna()
    if not new_products.empty:
        print("\nЗнайдено нові товари:")
        print(new_products)
    else:
        print("\nНових товарів не знайдено.")

    # Голосові команди
    end_program = False
    while not end_program:
        audio = capture_voice_input()
        text = convert_voice_to_text(audio)
        end_program = process_voice_command(text, all_products)


if __name__ == "__main__":
    recognizer = sr.Recognizer()
    URL_TEMPLATE = "https://shafa.ua/uk/sport/sport/turizm-i-kemping"
    monitor_product_changes(URL_TEMPLATE)
