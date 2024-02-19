import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


def Model_square(n):
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = -(0.0000005*i*i)
    return S0


def random_exponential(alfa, size):
    S = np.random.exponential(alfa, size)
    mS = np.mean(S)
    dS = np.var(S)
    scvS = math.sqrt(dS)
    print('------- статистичні характеристики ЕКСПОНЕНЦІЙНОГО закону розподілу ВВ -----')
    print('математичне сподівання ВВ = ', mS)
    print('дисперсія ВВ = ', dS)
    print('СКВ ВВ = ', scvS)
    print('----------------------------------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S


def Model_NORM(SN, S0N, n):
    SV = np.zeros(n)
    for i in range(n):
        SV[i] = S0N[i]+SN[i]
    return SV


def Model_NORM_AV(S0, SV, n):
    SV_AV = SV
    Q_AV = 4  # коефіцієнт переваги АВ
    dsig = 2.5  # СКВ
    nAVv = 8  # кількість АВ в абсолютних одиницях
    nAV = int((n * nAVv) / 100)  # кількість АВ у відсотках
    SSAV = np.random.normal(1, (Q_AV * dsig), nAV)  # аномальна випадкова похибка з нормальним законом
    for i in range(nAV):
        k = math.ceil(np.random.randint(1, n))  # рівномірний розкид номерів АВ в межах вибірки розміром 0-n
        SV_AV[k] = S0[k] + SSAV[i]  # аномальні вимірів з рівномірно розподіленими номерами
    return SV_AV


def Plot_AV(S0_L, SV_L, Text):
    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return


def file_parsing(URL, File_name, Data_name):
    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print('Джерело даних: ', URL)
    return S_real


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


def Stat_characteristics_in(SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = math.sqrt(dS)
    print('------------', Text,'-------------')
    print('кількість елементів вибірки = ', iter)
    print('математичне сподівання ВВ = ', mS)
    print('дисперсія ВВ = ', dS)
    print('СКВ ВВ = ', scvS)
    print('-----------------------------------------------------')
    return


def Stat_characteristics_out(SL_in, SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = math.sqrt(dS)
    # глобальне лінійне відхилення оцінки - динамічна похибка моделі
    Delta = 0
    for i in range(iter):
        Delta = Delta + abs(SL_in[i] - Yout[i, 0])
    Delta_average_Out = Delta / (iter + 1)
    print('------------', Text, '-------------')
    print('кількість елементів вибірки = ', iter)
    print('математичне сподівання ВВ = ', mS)
    print('дисперсія ВВ = ', dS)
    print('СКВ ВВ = ', scvS)
    print('Динамічна похибка моделі = ', Delta_average_Out)
    print('-----------------------------------------------------')
    return


# ------------- «навчання» параметрів відомих алгоритмів «бачити» властивості статистичної вибірки -------------
def Find_n_Wind(S0):
    iter = len(S0)
    dS = np.var(S0)
    scvS = math.sqrt(dS)

    scope = int(scvS) * 3
    n_scope = int(iter / scope)

    diffs = []
    for i in range(n_scope):
        S0_scope = []
        for j in range(scope):
            S0_scope.append(S0[i*scope + j])
        max_value = max(S0_scope)
        min_value = min(S0_scope)
        diff = max_value - min_value
        diffs.append(diff)

    # чим більше великих різниць, тим більше ковзне вікно
    mS = np.median(diffs)
    n_Wind = int(mS)
    return n_Wind


# ------------------------------ Виявлення АВ за алгоритмом sliding window -------------------------------------
def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = math.ceil(iter-n_Wind)+1
    S0_Wind = np.zeros(n_Wind)
    Midi = np.zeros(iter)
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j+i)
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        Midi[l] = np.median(S0_Wind)
    # ---- очищена вибірка  -----
    S0_Midi = np.zeros(iter)
    for j in range(iter):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi


# ------------------------------ МНК згладжування -------------------------------------
def MNK(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    print('Регресійна модель:')
    print('y(t) = ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2')
    return Yout


def r2_score(SL, Yout, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    iter = len(Yout)
    numerator = 0
    denominator_1 = 0
    for i in range(iter):
        numerator = numerator + (SL[i] - Yout[i, 0]) ** 2
        denominator_1 = denominator_1 + SL[i]
    denominator_2 = 0
    for i in range(iter):
        denominator_2 = denominator_2 + (SL[i] - (denominator_1 / iter)) ** 2
    R2_score_our = 1 - (numerator / denominator_2)
    print('------------', Text, '-------------')
    print('кількість елементів вбірки=', iter)
    print('Коефіцієнт детермінації (ймовірність апроксимації)=', R2_score_our)

    return R2_score_our


# ---------------------------  МНК ПРОГНОЗУВАННЯ -------------------------------
def MNK_Extrapol(S0, koef):
    iter = len(S0)
    Yout_Extrapol = np.zeros((iter+koef, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    print('Регресійна модель:')
    print('y(t) = ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2')
    for i in range(iter+koef):
        Yout_Extrapol[i, 0] = C[0, 0]+C[1, 0]*i+(C[2, 0]*i*i)   # проліноміальна крива МНК - прогнозування
    return Yout_Extrapol


# ----- статистичні характеристики екстраполяції  --------
def Stat_characteristics_extrapol(koef, SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros(iter)
    for i in range(iter):
        SL0[i] = SL[i,0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = math.sqrt(dS)
    #  довірчий інтервал прогнозованих значень за СКВ
    scvS_extrapol = scvS * koef
    print('------------', Text, '-------------')
    print('кількість елементів вибірки=', iter)
    print('математичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('Довірчий інтервал прогнозованих значень за СКВ=', scvS_extrapol)
    print('-----------------------------------------------------')
    return


# ------------------------------ Джерело вхідних даних ---------------------------
print('Оберіть джерело вхідних даних: ')
print('1 - модель')
print('2 - реальні дані')
Data_mode = int(input('mode:'))

if Data_mode == 1:
    # ------------------------------ сегмент констант ---------------------------
    print('----------------- Обрано: експоненційний закон розподілу ВВ --------------')
    print('------------------- ВХІДНІ параметри закону розподілу ВВ:-----------------')
    alfa = 5
    size = 10000  # кількість реалізацій ВВ
    print('параметр alfa = ', alfa)
    print("об'єм вибірки ВВ = ", size)
    # ------------------------------ сегмент даних ---------------------------
    # ------------ виклики функцій моделей: тренд, аномального та нормального шуму  ----------
    S0 = Model_square(size)  # модель ідеального тренду (квадратичний закон)
    S = random_exponential(alfa, size)  # модель помилок за експоненційним законом
    # ----------------------------- Нормальні похибки ------------------------------------
    SV = Model_NORM(S, S0, size)  # модель тренда + нормальних помилок
    Plot_AV(S0, SV, 'квадратична модель + Норм. шум')
    Stat_characteristics_in(SV, 'Вибірка + Норм. шум')
    # ----------------------------- Аномальні похибки ------------------------------------
    SV_AV = Model_NORM_AV(S0, SV, size)  # модель тренда + нормальних помилок + АВ
    Plot_AV(S0, SV_AV, 'квадратична модель + Норм. шум + АВ')
    Stat_characteristics_in(SV_AV, 'Вибірка з АВ')

if Data_mode == 2:
    url = 'https://meteostat.net/en/place/ua/kyiv?s=33345&t=2023-01-01/2023-10-01'
    SV_AV = file_parsing(url, 'Kyiv_weather.xlsx', 'tavg')

    S0 = SV_AV
    size = len(S0)  # кількість реалізацій ВВ
    Plot_AV(SV_AV, SV_AV, 'Коливання температури у 2023 році')
    Stat_characteristics_in(SV_AV, 'Коливання температури у 2023 році')

print('Вибірка очищена від АВ метод sliding_wind')
# --------------- Очищення від аномальних похибок sliding window -------------------
# n_Wind = 15  # розмір ковзного вікна для виявлення АВ
n_Wind = Find_n_Wind(SV_AV)
print('n_Wind =', n_Wind)
StartTime = time.time()                 # фіксація часу початку обчислень
S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
totalTime = (time.time() - StartTime)   # фіксація часу, на очищення від АВ
print('totalTime =', totalTime, 's')
Stat_characteristics_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind')
Plot_AV(S0, S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind')

# --------------- MNK згладжування -------------------------
Yout_SV_AV_Detect_sliding_wind = MNK(S_AV_Detect_sliding_wind)
Stat_characteristics_out(SV_AV, Yout_SV_AV_Detect_sliding_wind,
                         'MNK згладжена, вибірка очищена від АВ алгоритм sliding_wind')

# --------------- Оцінювання якості моделі та візуалізація -------------------------
r2_score(S_AV_Detect_sliding_wind, Yout_SV_AV_Detect_sliding_wind, 'MNK модель згладжування')
Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
        'MNK згладжена, вибірка очищена від АВ алгоритм sliding_wind')

print('MNK ПРОГНОЗУВАННЯ')
# --------------- Очищення від аномальних похибок sliding window -------------------
koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
koef = math.ceil(size * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки

Yout_SV_AV_Detect_sliding_wind = MNK_Extrapol(S_AV_Detect_sliding_wind, koef)
Stat_characteristics_extrapol(koef, Yout_SV_AV_Detect_sliding_wind,
                              'MNK ПРОГНОЗУВАННЯ, вибірка очищена від АВ алгоритм sliding_wind')
Plot_AV(Yout_SV_AV_Detect_sliding_wind, S_AV_Detect_sliding_wind,
        'MNK ПРОГНОЗУВАННЯ: Вибірка очищена від АВ алгоритм sliding_wind')
