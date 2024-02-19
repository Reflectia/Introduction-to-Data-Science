from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import numpy as np


def cp_model_solver():
    # Оптимізаційна математична модель
    model = cp_model.CpModel()
    num_vals = 6

    x1 = model.NewIntVar(0, num_vals - 1, 'x1')
    x2 = model.NewIntVar(0, num_vals - 1, 'x2')
    x3 = model.NewIntVar(0, num_vals - 1, 'x3')
    x4 = model.NewIntVar(0, num_vals - 1, 'x4')
    x5 = model.NewIntVar(0, num_vals - 1, 'x5')
    x6 = model.NewIntVar(0, num_vals - 1, 'x6')

    # Обмеження
    model.Add(3 * x1 + 4 * x2 - 2 * x4 <= 24)
    model.Add(x1 + 2 * x2 - x3 <= 8)
    model.Add(4 * x1 - x5 <= 16)
    model.Add(4 * x2 - x6 <= 12)

    # Цільова функція ефективності
    efficiency_function = -2 * x1 - 2 * x2

    # model.Maximize(efficiency_function)  # Максимізоція цільової функції
    model.Minimize(efficiency_function)  # Мінімізація цільової функції

    # Вирішувач
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print()
        print('Minimum of objective function: %i' % solver.ObjectiveValue())
        x1out = solver.Value(x1)
        x2out = solver.Value(x2)
        print('x1 = ', x1out)
        print('x2 = ', x2out)

        print('---------------Оптимальна-ефективність---------------------')
        MinimusOPT = -2 * x1out - 2 * x2out
        print('MinimusOPT= ', MinimusOPT)

        # -------------- Аналіз отриманого рішення ---------------------------
        '''
        ---------------Максимум-Максиморум для заданих обмежень ------------
        MaxMaxX1:  x_1+2x_2-x_3=8, x_2=0,x_3=0, x1max=8    (2 нерівність)
        MaxMaxX2:  3x_1+4x_2-2x_4=24, x_1=0,x_4=0, x2max=6    (1 нерівність)
        '''
        print('---------------- Максимум-Максиморум ------------------------')
        x1max = 8
        x2max = 6

        MaximusP = -2 * x1max - 2 * x2max
        print('MaximusP= ', MaximusP)

        '''
        --------------- Мінімум-Мініморум для заданих ообмежень -----------
        MinMinX1:  4x_1-x_5=16, x_5=0, x1min=4    (3 нерівність)
        MinMinX2:  4x_2-x_6=12, x_6=0, x2min=3    (4 нерівність)  
        '''
        print('---------------- Мінімум-Мініморум ------------------------')
        x1min = 4
        x2min = 3

        MinimusM = -2 * x1min - 2 * x2min
        print('MinimusM= ', MinimusM)

    return


def plot_solver():
    # -------- Графічна інтерпретація рішення "точкове" ----------
    # ------------------ оптималье рішення-------------------------
    plt.scatter(5, 4, color='green')
    # ------------------ максимальне рішення-----------------------
    plt.scatter(8, 6, color='blue')
    # ------------------ мінімальне рішення -----------------------
    plt.scatter(4, 3, color='red')
    plt.show()

    # -------- Графічна інтерпретація рішення "прямі" ----------
    x = np.arange(-5, 15)
    y1 = (24 - 3 * x) / 4
    y2 = (8 - x) / 2
    y3 = x
    y4 = np.full(20, 3)
    y5 = (-2 * x) / 2

    plt.plot(x, y1, color='blue')
    plt.plot(x, y2, color='red')
    plt.plot(np.full(20, 4), y3, color='green')
    plt.plot(x, y4, color='yellow')
    plt.plot(x, y5, color='purple')
    # ------------------ оптималье рішення-------------------------
    plt.scatter(5, 4, color='green')
    # ------------------ максимальне рішення-----------------------
    plt.scatter(8, 6, color='blue')
    # ------------------ мінімальне рішення -----------------------
    plt.scatter(4, 3, color='red')

    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.show()

    return


# ----------------------- головні виклики -------------------------
if __name__ == '__main__':
    cp_model_solver()
    plot_solver()
