import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt


# ------------------------------ МНК згладжування -------------------------------------
def MNK(Y_coord):
    # ---- формування структури вхідних матриць МНК ------
    iter = Y_coord.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)
    # ------------- алгоритм МНК ------------------------
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    return Yout


# ------------- МНК згладжуваннядля визначення стат. характеристик -------------
def MNK_Stat_characteristics(S0):
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
    return Yout


def Stat_characteristics(SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter ))
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print('------------', Text ,'-------------')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------')
    plt.title(Text)
    plt.hist(SL,  bins=30, facecolor="blue", alpha=0.5) # гістограма закону розподілу ВВ
    plt.show()
    return


# ------------------------------ МНК прогноз -------------------------------------
def MNK_extrapolation(Y_coord, koef):
    # ---- формування структури вхідних матриць МНК ------
    iter = Y_coord.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)
    # ------------- алгоритм МНК ------------------------
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    j=koef
    for i in range(0, koef):
        Yout[i, 0] = C[0, 0] + C[1, 0]*j + (C[2, 0]*j*j) + (C[3, 0]*j*j*j) + (C[4, 0]*j*j*j*j)
        j = j+1

    return Yout


# ------------------------------------парсинг вхідного файла -------------------------------

# пряме зчитування
dateparse = lambda x: pd.to_datetime(x + '-1-1', errors='coerce')
d = pd.read_excel('Data_Set_12.xlsx', parse_dates=['Year'], date_parser=dateparse)
print('d = ', d)

index = 'Value'
plt.title(index)
d[index].plot()
plt.show()

print('\n---------------- Value -------------------')
print(d['Value'])
print(type(float(d['Value'][0])))

# Розподіл значень Value
plt.hist(d['Value'], bins=20, color='cyan', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Values')
plt.show()

# Групування за колонкою 'Year'
d_year = d[['Year', 'Value']].groupby('Year').sum()
print('\nd_year = ', d_year)

d_year['Value'].plot(kind='bar', color='cyan')
plt.title('Values by Year')
plt.xlabel('Year')
plt.ylabel('Value')
plt.tight_layout()
plt.show()

# Групування за колонкою 'Sales_Rep_Name'
d_name = d[['Sales_Rep_Name', 'Value']].groupby('Sales_Rep_Name').sum()
print('\nd_name = ', d_name)

d_name['Value'].plot(kind='bar', color='cyan')
plt.title('Values by Sales_Rep_Name')
plt.xlabel('Sales_Rep_Name')
plt.ylabel('Value')
plt.tight_layout()
plt.show()

# Кількість замовлень за колонкою 'Postcode'
d_counts = d['Postcode'].value_counts()
print('\nd_counts = ', d_counts)

# Розподіл значень Postcode
plt.hist(d['Postcode'], bins=20, color='cyan', edgecolor='black')
plt.xlabel('Postcode')
plt.ylabel('Frequency')
plt.title('Distribution of Postcodes')
plt.show()

# Кількість унікальних значень Postcode
n_postcode = d['Postcode'].nunique()
print('\nn_postcode = ', n_postcode)

# Grouping by 'Sales_Rep_Name' and getting the count of unique postcodes
d_postcode_count = d.groupby('Sales_Rep_Name')['Postcode'].nunique()
print(d_postcode_count)

# Plotting the count of postcodes
d_postcode_count.plot(kind='bar', color='cyan')
plt.title('Postcodes by Sales_Rep_Name')
plt.xlabel('Sales_Rep_Name')
plt.ylabel('Postcode')
plt.tight_layout()
plt.show()

# Групування за колонками 'Year' та 'Sales_Rep_Name'
d_year_name = d[['Year', 'Sales_Rep_Name', 'Value']].groupby(['Year', 'Sales_Rep_Name']).sum()
print(d_year_name)

# Візуалізація графіка
d_year_name['Value'].unstack().plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Values by Year and Sales_Rep_Name')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend(title='Sales_Rep_Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---------------------------- ОЦІНЮВАННЯ ------------------------------
# ---------------------- МНК для колонки 'Value' -----------------------
Yout0 = MNK(d['Value'])
print('------------ вхідна вибірка  ----------')
Stat_characteristics(d['Value'], 'вхідна вибірка')
print('-------------- МНК оцінка  ------------')
Stat_characteristics(Yout0, 'МНК оцінка')
plt.title('MNK_value')
plt.plot(d['Value'])
plt.plot(Yout0)
plt.show()

# ---------------------- МНК для колонки 'Value' за 'Year' -------------------
Yout1 = MNK(d_year['Value'])
print('------------ вхідна вибірка  ----------')
Stat_characteristics(d_year['Value'], "вхідна вибірка за 'Year'")
print('-------------- МНК оцінка  ------------')
Stat_characteristics(Yout1, "МНК оцінка за 'Year'")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(d_year['Value'])
axs[0].set_title('Values by Year')
axs[1].plot(Yout1)
axs[1].set_title('MNK_value')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# ---------------------- МНК для колонки 'Value' за 'Sales_Rep_Name' -------------------
Yout2 = MNK(d_name['Value'])
print('------------ вхідна вибірка  ----------')
Stat_characteristics(d_name['Value'], "вхідна вибірка за 'Sales_Rep_Name'")
print('-------------- МНК оцінка  ------------')
Stat_characteristics(Yout2, "МНК оцінка за 'Sales_Rep_Name'")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(d_name['Value'])
axs[0].set_title('Values by Sales_Rep_Name')
axs[1].plot(Yout2)
axs[1].set_title('MNK_value')

plt.tight_layout()
plt.show()

# ----------------------------- ПРОГНОЗ --------------------------------
# ---------------------- МНК для колонки 'Value' -----------------------
prognoz = int(0.5*d['Value'].size)
Graf_Yout0 = np.zeros(prognoz)
Graf_Yout0_1 = np.zeros(6)
Yout0 = MNK_extrapolation(d['Value'], prognoz)
Graf_Yout0_1[0] = Yout0[1, 0]
Graf_Yout0_1[1] = Yout0[(int(prognoz/6)), 0]
Graf_Yout0_1[2] = Yout0[(int((2*prognoz/6))), 0]
Graf_Yout0_1[3] = Yout0[(int((3*prognoz/6))), 0]
Graf_Yout0_1[4] = Yout0[(int((4*prognoz/6))), 0]
Graf_Yout0_1[5] = Yout0[(int((5*prognoz/6))), 0]

print('------------ MNK_extrapolation_value ---------------')
for i in range(0, 6):
    print('Yout0[', i, ',0]=', Graf_Yout0_1[i])
plt.title('MNK_extrapolation_value')
for i in range(0, prognoz):
    Graf_Yout0[i] = Yout0[i, 0]
plt.plot(Graf_Yout0)
plt.show()

# ---------------------- МНК для колонки 'Value' за 'Year' -------------------
prognoz = 3
Graf_Yout1 = np.zeros(prognoz)
Yout1 = MNK_extrapolation(d_year['Value'], prognoz)
print('------ MNK_extrapolation_value_year --------')
for i in range(prognoz):
    print('Yout1[', i, ',0]=', Yout1[i, 0])
plt.title('MNK_extrapolation_value_year')
for i in range(0, prognoz):
    Graf_Yout1[i] = Yout1[i, 0]
plt.plot(Graf_Yout1)
plt.show()

# ---------------------- МНК для колонки 'Value' за 'Sales_Rep_Name' -------------------
prognoz = 3
Graf_Yout2 = np.zeros(prognoz)
Yout2 = MNK_extrapolation(d_name['Value'], prognoz)
print('------ MNK_extrapolation_value_name --------')
for i in range(prognoz):
    print('Yout2[', i, ',0]=', Yout2[i, 0])
plt.title('MNK_extrapolation_value_name')
for i in range(0, prognoz):
    Graf_Yout2[i] = Yout2[i, 0]
plt.plot(Graf_Yout2)
plt.show()

# -------------------------------------------------------------------------
plt.title("MNK_extrapolation за 'Value', 'Value' за 'Year', 'Value' за 'Sales_Rep_Name'")
plt.plot(Graf_Yout0_1)
plt.plot(Graf_Yout1)
plt.plot(Graf_Yout2)
plt.show()
