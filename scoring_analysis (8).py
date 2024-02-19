from typing import Any

import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------------- І. ПІДГОТОВКА ВХІДНИХ ДАНИХ ------------------------------
# ----------------------------------- парсинг файлу вхідних даних ------------------------------
d_sample_data = pd.read_excel('sample_data.xlsx', parse_dates=['birth_date'])
print('d_sample_data=', d_sample_data)

# ------------------------------- аналіз структури вхідних даних ------------------------------
Title_d_sample_data = d_sample_data.columns
print('-------------  назви стовпців DataFrame  -----------')
print(Title_d_sample_data)                                           # заголовок стовпців таблиці DataFrame
print('---------  типи даних стовпців DataFrame  -----------')
print(d_sample_data.dtypes)                                          # визначення типу даних таблиці DataFrame
print('---------  пропущені значення стовпців (суми)  ------')
print(d_sample_data.isnull().sum())                                  # визначення пропусків даних DataFrame

# ---------------------------- парсинг файлу пояснень параметрів ------------------------------
d_data_description = pd.read_excel('data_description.xlsx')
print('---------------  d_data_description  ---------------')
print('d_data_description = ', d_data_description)
print('----------------------------------------------------')

# -------------------------- сегментація ознак клієнта та кредиту  -------------------------------
d_segment_data_description_client_bank = \
    d_data_description[(d_data_description.Place_of_definition == 'Вказує позичальник') |
                       (d_data_description.Place_of_definition == 'параметри, повязані з виданим продуктом')]
n_client_bank = d_segment_data_description_client_bank['Place_of_definition'].size  # розмір стовпця
d_segment_data_description_client_bank.index = range(0, len(d_segment_data_description_client_bank))
print('---------  d_segment_data_description_client_bank  -----------')
print('d_segment_data_description_client_bank = ', d_segment_data_description_client_bank)
print('----------------------------------------------------')

print('перевірка наявності індексів клієнта та кредиту з d_data_description в даних d_sample_data')
# ------ 'Field_in_data' - стовпчик із назвами індикаторів скорінгу
b = d_segment_data_description_client_bank['Field_in_data']

# ------ кількість співпадінь
n_columns = d_segment_data_description_client_bank['Field_in_data'].size
j = 0
for i in range(0, n_columns):
    a = d_segment_data_description_client_bank['Field_in_data'][i]
    if set([a]).issubset(d_sample_data.columns):
        j = j+1  # кількість співпадінь для кожного columns
print('Кількість співпадінь = ', j)
# ------ індекси співпадінь
Columns_Flag_True = np.zeros(j)
j = 0
for i in range(0, n_columns):
    a = d_segment_data_description_client_bank['Field_in_data'][i]
    if set([a]).issubset(d_sample_data.columns):  # перевірка кожного columns
        Flag = 'Flag_True'
        Columns_Flag_True[j] = i
        j = j+1
    else:
        Flag = 'Flag_False'
print('Індекси співпадінь', Columns_Flag_True)

# ------ DataFrame співпадінь
# Формування
d_segment_data_description_client_bank_True = d_segment_data_description_client_bank.iloc[Columns_Flag_True]
# Балансування індексів
d_segment_data_description_client_bank_True.index = range(0, len(d_segment_data_description_client_bank_True))
print('------------ DataFrame співпадінь -------------')
print(d_segment_data_description_client_bank_True)
print('-----------------------------------------------')

# ------- формування сегменту вхідних даних за рейтингом клієнт + банк --------
b = d_segment_data_description_client_bank_True['Field_in_data']
d_segment_sample_data_client_bank = d_sample_data[b]
print('---- пропуски даних сегменту DataFrame --------')
print(d_segment_sample_data_client_bank.isnull().sum())  # визначення пропусків даних DataFrame
print('-----------------------------------------------')

# ------ вилучення строк та індикаторів з пропусками - СКОРИНГОВА КАРТА -------
# Очищення індикаторів скорингової таблиці
d_segment_data_description_cleaning = d_segment_data_description_client_bank_True.loc[
      (d_segment_data_description_client_bank_True['Field_in_data'] != 'fact_addr_start_date')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'position_id')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'employment_date')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'has_prior_employment')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'prior_employment_start_date')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'prior_employment_end_date')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'income_frequency_other')]
d_segment_data_description_cleaning.index = range(0, len(d_segment_data_description_cleaning))
d_segment_data_description_cleaning.to_excel('d_segment_data_description_cleaning.xlsx')

# Очищення вхідних даних
d_segment_sample_cleaning = d_segment_sample_data_client_bank.drop(columns=['fact_addr_start_date', 'position_id',
                                                                            'employment_date', 'has_prior_employment',
                                                                            'prior_employment_start_date',
                                                                            'prior_employment_end_date',
                                                                            'income_frequency_other'])
d_segment_sample_cleaning.index = range(0, len(d_segment_sample_cleaning))
d_segment_sample_cleaning.to_excel('d_segment_sample_cleaning.xlsx')
print('--- Контроль наявності пропусків даних після очищення на індикаторах ---')
print(d_segment_sample_cleaning.isnull().sum())
print('---------- DataFrame вхідних даних - скорингова карта -----------')
print(d_segment_sample_cleaning)
print('----------------- DataFrame індикатори скорингу  ----------------')
print(d_segment_data_description_cleaning)
print('-----------------------------------------------------------------')

# ----------------------------------- ІІ. ФОРМУВАННЯ СКОРИНГОВОЇ МОДЕЛІ ------------------------------
# --------------------------- Скорингова модель багатокритеріального оцінювання ----------------------
d_segment_data_description_minimax = pd.read_excel('d_segment_data_description_minimax.xlsx')  # інфологічна модель

print('----------------- DataFrame d_segment_data_description_minimax  ----------------')
print(d_segment_data_description_minimax)
print('-----------------------------------------------------------------')

# відбір даних за критеріями
d = d_segment_data_description_minimax['Field_in_data']
cols = d.values.tolist()  # перетворення стовпчика DataFrame у строку
d_segment_sample_minimax: Series | None | ndarray | DataFrame | Any = d_segment_sample_cleaning[cols]
print('----------------- DataFrame d_segment_sample_minimax  ----------------')
print(cols)
print(d_segment_sample_minimax)
print('-----------------------------------------------------------------')
d_segment_sample_minimax.to_excel('d_segment_sample_minimax.xlsx')

# мінімальні, максимальні значення стовпчиків DataFrame
d_segment_sample_min = d_segment_sample_minimax[cols].min()
d_segment_sample_max = d_segment_sample_minimax[cols].max()
print('----------------- DataFrame: d_segment_sample_min  ----------------')
print(d_segment_sample_min)
print('----------------- DataFrame: d_segment_sample_max  ----------------')
print(d_segment_sample_max)

# нормування критеріїв
m = d_segment_sample_minimax['loan_amount'].size
n = d_segment_data_description_minimax['Field_in_data'].size
d_segment_sample_minimax_Normal = np.zeros((m, n))  # перехід в розрахунках до масиву numpy

delta_d = 0.3  # коефіцієнт запасу при нормуванні
for j in range(0, len(d_segment_data_description_minimax)):
    columns_d = d_segment_data_description_minimax['Minimax'][j]
    if columns_d == 'min':
        columns_m = d_segment_data_description_minimax['Field_in_data'][j]
        for i in range(0, len(d_segment_sample_minimax)):
            max_max = d_segment_sample_max[j]+(2*delta_d)
            d_segment_sample_minimax_Normal[i, j] = (delta_d + d_segment_sample_minimax[columns_m][i]) / max_max
    else:
        for i in range(0, len(d_segment_sample_minimax)):
            min_min = d_segment_sample_max[j] + (2*delta_d)
            d_segment_sample_minimax_Normal[i, j] = (1 / (delta_d + d_segment_sample_minimax[columns_m][i])) / min_min

print(d_segment_sample_minimax_Normal)
np.savetxt('d_segment_sample_minimax_Normal.txt', d_segment_sample_minimax_Normal)   # файл нормованих параметрів


# інтегрована багатокритеріальна оцінка - SCOR
def Voronin(d_segment_sample_minimax_Normal, n, m=150):
    Integro = np.zeros(m)
    Scor = np.zeros(m)
    for i in range(0, m):
        Sum_Voronin = 0
        for j in range(0, n):
            Sum_Voronin = Sum_Voronin + ((1 - d_segment_sample_minimax_Normal[i, j]) ** (-1))
        Integro[i] = Sum_Voronin
        Scor[i] = 1000  # порог прийняття рішень про видачу кредиту
        np.savetxt('Integro_Scor.txt', Integro)  # файл інтегрованого показника - СКОРУ
    plt.title('Integro_Scor')
    plt.plot(Integro)
    plt.plot(Scor)
    plt.show()
    return Integro


# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ------------------------------
if __name__ == '__main__':
    Voronin(d_segment_sample_minimax_Normal, n)

    # логістична регресія на основі результатів функції Voronin
    Integro = Voronin(d_segment_sample_minimax_Normal, n, m)
    print(f'Кількість позичальників, яким буде відмовлено у кредиті: {sum(Integro < 1000)}')

    Y = np.array([0 if y < 1000 else 1 for y in Integro])
    X = d_segment_sample_minimax.values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    print("Actual Labels: ", Y_test)
    print("Predictions: ", predictions)

    accuracy = accuracy_score(Y_test, predictions)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # ------------------------- # Виявлення аномалій (шахрайство) -----------------------------------
    minimax_data = pd.read_excel('d_segment_sample_minimax.xlsx')

    # Використання моделі Isolation Forest
    model = IsolationForest(contamination=0.05)  # Contamination is the proportion of outliers in the data
    model.fit(minimax_data)

    minimax_data['fraud'] = model.predict(minimax_data)
    n_fraud = minimax_data[minimax_data['fraud'] == -1].shape[0]
    print(f'Кількість позичальників, якs підозрюються на шахрайство: {n_fraud}')

    # Використання PCA for зменшення вимірів
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(minimax_data.drop('fraud', axis=1))

    reduced_data_with_anomaly = pd.DataFrame(reduced_data, columns=['principal_component_1', 'principal_component_2'])
    reduced_data_with_anomaly['fraud'] = minimax_data['fraud']

    sns.scatterplot(x='principal_component_1', y='principal_component_2', hue='fraud', data=reduced_data_with_anomaly,
                    palette='viridis')
    plt.title('Fraud detection')
    plt.tight_layout()
    plt.show()
