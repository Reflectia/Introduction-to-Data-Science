"""
    Розробити програмний скрипт мовою Python що забезпечує аналіз властивостей іхарактеристик вихідних даних
відповідно до етапів:
    1. Модель генерації випадкової величини за заданим у табл.1 додатку 1 закону розподілу;
    2. Модель зміни (ідеальний тренд) досліджуваного процесу за заданим у табл.1 додатку 1 законом;
    3. Адитивна модель статистичної вибірки відповідно до синтезованих в п.1,2 моделей випадкової (стохастична)
і невипадкової складових. Параметри закону розподілу та закону зміни досліджуваного процесу обрати самостійно.
    4. Визначення статистичних (числових) характеристик сформованих в п.1,3 вибірок (дисперсія,
середньоквадратичне відхилення, математичне очікування, гістограма закону розподілу).
    5. Визначення статистичних характеристик реальних даних, заданих файлом oschadbank (USD).xls
за умов табл. 1 додатку 1.
    6. Провести аналіз отриманих результатів та верифікацію розробленого скрипта.
"""

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd
import xlrd


# 1 Закон зміни похибки – нормальний, експоненційний.
def random_norm(dm, dsig, size):
    S = np.random.normal(dm, dsig, size)
    mS = np.mean(S)
    dS = np.var(S)  # дисперсія
    scvS = mt.sqrt(dS)  # середнє квадратичне відхилення
    print('------- статистичні характеристики НОРМАЛЬНОГО закону розподілу ВВ -----')
    print('матиматичне сподівання ВВ = ', mS)
    print('дисперсія ВВ = ', dS)
    print('СКВ ВВ = ', scvS)
    print('------------------------------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S


def random_exponential(alfa, size):
    S = np.random.exponential(alfa, size)
    mS = np.mean(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('------- статистичні характеристики ЕКСПОНЕНЦІЙНОГО закону розподілу ВВ -----')
    print('математичне сподівання ВВ = ', mS)
    print('дисперсія ВВ = ', dS)
    print('СКВ ВВ = ', scvS)
    print('----------------------------------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S


# 2 Закон зміни досліджуваного процесу(тренду) – постійна, квадратичний.
def Model_const(n):
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.0000005
    return S0


def Model_square(n):
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = (0.0000005*i*i)
    return S0


def Model_NORM(SN, S0N, n):
    SV = np.zeros(n)
    for i in range(n):
        SV[i] = S0N[i]+SN[i]
    return SV


def Model_NORM_AV(S0, SV, n):
    SV_AV = SV
    Q_AV = 3  # коефіцієнт переваги АВ
    dsig = 4  # СКВ
    nAVv = 10  # кількість АВ в абсолютних одиницях
    nAV = int((n * nAVv) / 100)  # кількість АВ у відсотках
    SSAV = np.random.normal(1, (Q_AV * dsig), nAV)  # аномальна випадкова похибка з нормальним законом
    for i in range(nAV):
        k = mt.ceil(np.random.randint(1, n))  # рівномірний розкид номерів АВ в межах вибірки розміром 0-n
        SV_AV[k] = S0[k] + SSAV[i]  # аномальні вимірів з рівномірно розподіленими номерами
    return SV_AV


def Plot_AV(S0_L, SV_L, Text, is_legend=False):
    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L, label='лінія тренду')
    if is_legend:
        plt.legend()
    plt.ylabel(Text)
    plt.show()
    return


def MNK_Stat_characteristics(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout


def Stat_characteristics(SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print('------------', Text, '-------------')
    print('кількість елементів вибірки = ', iter)
    print('математичне сподівання ВВ = ', mS)
    print('дисперсія ВВ = ', dS)
    print('СКВ ВВ = ', scvS)
    print('-----------------------------------------------------')
    return


# 5
def file_parsing(URL, File_name, Data_name):
    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print('Джерело даних: ', URL)
    return S_real


size = 10000
S = np.array([])
S0 = np.array([])
print('Оберіть закон розподілу ВВ:')
print('1 - нормальний')
print('2 - експоненційний')
mode = input('mode: ')

if mode == '1':
    print('------------------ Обрано: нормальний закон розподілу ВВ -----------------')
    print('------------------- ВХІДНІ параметри закону розподілу ВВ:-----------------')
    dm = 0
    dsig = 3
    print('математичне сподівання ВВ = ', dm)
    print('СКВ ВВ = ', dsig)
    print("об'єм вибірки ВВ = ", size)
    S = random_norm(dm, dsig, size)
elif mode == '2':
    print('----------------- Обрано: експоненційний закон розподілу ВВ --------------')
    print('------------------- ВХІДНІ параметри закону розподілу ВВ:-----------------')
    alfa = 0.5
    print('параметр alfa = ', alfa)
    print("об'єм вибірки ВВ = ", size)
    S = random_exponential(alfa, size)
else:
    pass

print('Оберіть закон зміни тренду:')
print('1 - постійна величина')
print('2 - квадратичний')
mode = input('mode: ')

if mode == '1':
    S0 = Model_const(size)
elif mode == '2':
    S0 = Model_square(size)
else:
    pass

if S.any() and S0.any():
    SV = Model_NORM(S, S0, size)
    Plot_AV(S0, SV, 'Адитивна модель статистичної вибірки', True)
    Stat_characteristics(SV, 'Вибірка + Норм. шум')

    SV_AV = Model_NORM_AV(S0, SV, size)  # модель тренда + нормальних помилок + АВ
    Plot_AV(S0, SV_AV, 'Адитивна модель статистичної вибірки + АВ', True)
    Stat_characteristics(SV_AV, 'Вибірка + Норм. шум + АВ')

print('Оберіть, які дані отримати:')
print('1 - купівля')
print('2 - продаж')
print('3 - курсНБУ')
mode = input('mode: ')

SV_AV = np.array([])
if mode == '1':
    SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls', 'Купівля')
elif mode == '2':
    SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls', 'Продаж')
elif mode == '3':
    SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls', 'КурсНбу')
else:
    pass

if SV_AV.any():
    Plot_AV(SV_AV, SV_AV, 'Коливання курсу USD в 2022 році за даними Ощадбанк')
    Stat_characteristics(SV_AV, 'Коливання курсу USD в 2022 році за даними Ощадбанк')
