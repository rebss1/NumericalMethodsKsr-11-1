import matplotlib
import matplotlib.pyplot as plt
import math
import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import ttk
from PIL.ImageChops import constant
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def rhs_u1(x, u, u1, g, l):
    return u1
def rhs_var1(x, u, u1, g, l):
    return - g * math.sin(u) / l
def rhs_var2(x, u, u1, g, l):
    return - g * u / l

def rk4_step_var1(v1, v2, x, h, l, g):
    k1_u1 = rhs_u1(x, v1, v2, g, l)
    k1_u2 = rhs_var1(x, v1, v2, g, l)
    k2_u1 = rhs_u1(x, v1 + h/2 * k1_u1, v2 + h/2 * k1_u2, g, l)
    k2_u2 = rhs_var1(x, v1 + h/2 * k1_u1, v2 + h/2 * k1_u2, g, l)
    k3_u1 = rhs_u1(x, v1 + h/2 * k2_u1, v2 + h/2 * k2_u2, g, l)
    k3_u2 = rhs_var1(x, v1 + h/2 * k2_u1, v2 + h/2 * k2_u2, g, l)
    k4_u1 = rhs_u1(x, v1 + h*k3_u1, v2 + h*k3_u2, g, l)
    k4_u2 = rhs_var1(x, v1 + h*k3_u1, v2 + h*k3_u2, g, l)

    x = x + h
    v1 = v1 + (h/6)*(k1_u1 + 2*k2_u1 + 2*k3_u1 + k4_u1)
    v2 = v2 + (h/6)*(k1_u2 + 2*k2_u2 + 2*k3_u2 + k4_u2)
    return x, v1, v2

def rk4_step_var2(v1, v2, x, h, l, g):
    k1_u1 = rhs_u1(x, v1, v2, g, l)
    k1_u2 = rhs_var2(x, v1, v2, g, l)
    k2_u1 = rhs_u1(x, v1 + h/2 * k1_u1, v2 + h/2 * k1_u2, g, l)
    k2_u2 = rhs_var2(x, v1 + h/2 * k1_u1, v2 + h/2 * k1_u2, g, l)
    k3_u1 = rhs_u1(x, v1 + h/2 * k2_u1, v2 + h/2 * k2_u2, g, l)
    k3_u2 = rhs_var2(x, v1 + h/2 * k2_u1, v2 + h/2 * k2_u2, g, l)
    k4_u1 = rhs_u1(x, v1 + h*k3_u1, v2 + h*k3_u2, g, l)
    k4_u2 = rhs_var2(x, v1 + h*k3_u1, v2 + h*k3_u2, g, l)

    x = x + h
    v1 = v1 + (h/6)*(k1_u1 + 2*k2_u1 + 2*k3_u1 + k4_u1)
    v2 = v2 + (h/6)*(k1_u2 + 2*k2_u2 + 2*k3_u2 + k4_u2)
    return x, v1, v2

def get_s(v1, v2, v1_, v2_):
    s1 = abs(v1_ - v1) / 15
    s2 = abs(v2_ - v2) / 15
    if s1 >= s2:
        s = s1
    else: s = s2
    return s

def control(s, eps, h):
    if abs(s) < eps / 32:
        return 2
    if abs(s) >= eps:
        return 0
    else:
        return 1

def sol_var1(x_0, v1_0, v2_0, l, g, h_0, max_steps, eps, max_periods, border_eps):
    x = x_0
    v1 = v1_0
    v2 = v2_0
    h = h_0
    v1_ = 0.0
    v2_ = 0.0
    s = 0.0
    doub = int(0)
    div = int(0)
    periods = 0
    res = [[x, v1, v2, 0, 0, 0, 0, h_0, 0, 0, 0]]

    for i in range(1, max_steps + 1):
        flag = 0
        while flag == 0:
            point = rk4_step_var1(v1, v2, x, h, l, g)
            point1_1 = rk4_step_var1(v1, v2, x, h / 2, l, g)
            point1_2 = rk4_step_var1(point1_1[1], point1_1[2], point1_1[0], h / 2, l, g)

            x, v1, v2 = point[0], point[1], point[2]
            v1_, v2_ = point1_2[1], point1_2[2]
            s = get_s(v1, v2, v1_, v2_)
            flag = control(s, eps, h)

            if flag == 0:
                h = h / 2
                div += 1
            elif flag in [1, 2]:
                row = [x, v1, v2, v1_, v2_, abs(v1-v1_), abs(v2-v2_), h, s * 16, div, doub]
                res.append(row)
                if flag == 2:
                    h = h * 2
                    doub += 1

        if i > 100:
            check_one = v1_0 - border_eps
            check_two = v1_0 + border_eps
            numb = res[-1][1]  # последний вычисленный v1
            if check_one <= numb <= check_two:
                periods += 1
            if periods >= max_periods:
                break

    return res

def sol_var2(x_0, v1_0, v2_0, l, g, h_0, max_steps, eps, max_periods, border_eps):
    x = x_0
    v1 = v1_0
    v2 = v2_0
    h = h_0
    v1_ = 0.0
    v2_ = 0.0
    s = 0.0
    doub = int(0)
    div = int(0)
    periods = 0
    res = [[x, v1, v2, 0, 0, 0, 0, h_0, 0, 0, 0]]

    for i in range(1, max_steps + 1):
        flag = 0
        while flag == 0:
            point = rk4_step_var2(v1, v2, x, h, l, g)
            point1_1 = rk4_step_var2(v1, v2, x, h / 2, l, g)
            point1_2 = rk4_step_var2(point1_1[1], point1_1[2], point1_1[0], h / 2, l, g)

            x, v1, v2 = point[0], point[1], point[2]
            v1_, v2_ = point1_2[1], point1_2[2]
            s = get_s(v1, v2, v1_, v2_)
            flag = control(s, eps, h)

            if flag == 0:
                h = h / 2
                div += 1
            elif flag in [1, 2]:
                row = [x, v1, v2, v1_, v2_, abs(v1-v1_), abs(v2-v2_), h, s * 16, div, doub]
                res.append(row)
                if flag == 2:
                    h = h * 2
                    doub += 1

        if i > 100:
            check_one = v1_0 - border_eps
            check_two = v1_0 + border_eps
            numb = res[-1][1]  # последний вычисленный v1
            if check_one <= numb <= check_two:
                periods += 1
            if periods >= max_periods:
                break

    return res

def show_summary_and_table(df, summary_text):
    # Создание главного окна
    root = tk.Tk()
    root.title("Справка и Таблица")

    # Создание фрейма для справки
    frame_summary = ttk.Frame(root)
    frame_summary.pack(side="top", fill="x", padx=10, pady=10)

    label = tk.Label(frame_summary, text=summary_text, justify="left", font=("Arial", 12))
    label.pack()

    # Создание фрейма для таблицы
    frame_table = ttk.Frame(root)
    frame_table.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

    table = ttk.Treeview(frame_table, columns=list(df.columns), show="headings")
    for col in df.columns:
        table.heading(col, text=col)
        table.column(col, width=100)

    for _, row in df.iterrows():
        formatted_row = [
            f"{row['x']:.6f}" if col == "x" else int(row[col]) if col in ["doub", "div"] else row[col]
            for col in df.columns
        ]
        table.insert("", "end", values=formatted_row)

    table.pack(fill="both", expand=True)

    # Запуск интерфейса
    root.mainloop()

def show_table():
    border_eps = float(border_eps_entry.get())
    max_periods = float(max_periods_entry.get())
    eps = float(maxError_entry.get())
    h_0 = float(h0_entry.get())
    max_steps = int(max_steps_entry.get())
    x_0 = float(x0_entry.get())
    v1_0 = float(v10_entry.get())
    v2_0 = float(v20_entry.get())
    l = float(l_entry.get())
    g = 9.8  # ускорение свободного падения

    # Вычисление результатов
    results = sol_var1(x_0, v1_0, v2_0, l, g, h_0, max_steps, eps, max_periods, border_eps)
    df = pd.DataFrame(results,
                      columns=["x", "v1", "v2", "v1_1/2", "v2_1/2", "|v1-v1_1/2|", "|v2-v2_1/2|", "h", "Норма ОЛП", "divided", "doubled"])

    # Вычисление необходимых параметров
    final_step_size = len(df["h"])
    final_angle = df["v1"].iloc[-1]
    final_x = df["x"].iloc[-1]
    total_divisions = df["divided"].iloc[-1]
    total_doublings = df["doubled"].iloc[-1]
    max_local_error = df["Норма ОЛП"].max()

    # Текст справки
    summary_text = (
        f"Справка:\n\n"
        f"x - время (с)\n"
        f"v1 - угол отклонения маятника целым шагом(рад)\n"
        f"v1_1/2 - угол отклонения маятника половинным шагом  (рад)\n"
        f"v2 - угловая скорость маятника целым шагом(рад/c)\n"
        f"v2_1/2 - угловая скорость маятника половинным шагом  (рад/c)\n"
        f"h - шаг методом РК 4 порядка\n"
        f"Количество шагов h: {final_step_size:.6f}\n"
        f"Конечный угол v1 (рад): {final_angle}\n"
        f"Конечное время: {final_x}\n"
        f"Количество делений шага (divided): {total_divisions}\n"
        f"Количество умножений шага (doubled): {total_doublings}\n"
        f"Максимальная оценка локальной погрешности (Норма ОЛП): {max_local_error}\n"
    )

    show_summary_and_table(df, summary_text)

def show_graph1():
    border_eps = float(border_eps_entry.get())
    max_periods = float(max_periods_entry.get())
    eps = float(maxError_entry.get())
    h_0 = float(h0_entry.get())
    max_steps = int(max_steps_entry.get())
    x_0 = float(x0_entry.get())
    v10 = float(v10_entry.get())
    v20 = float(v20_entry.get())
    l = float(l_entry.get())
    g = 9.8  # ускорение свободного падения

    results = [sol_var1(x_0, v10, v20, l, g, h_0, max_steps, eps, max_periods, border_eps)]

    dfs = [pd.DataFrame(r, columns=["x", "v1", "v2", "v1_1/2", "v2_1/2", "|v1-v1_1/2|", "|v2-v2_1/2|", "h", "Норма ОЛП", "div",
                                    "doub"]) for r in results]

    def plot_graphs():
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        axs[0].plot(dfs[0]["x"], dfs[0]["v1"], label="Численное решение (рад)", color="red")
        axs[0].set_title("График зависимости угла от времени")
        axs[0].set_xlabel("Время (с)")
        axs[0].set_ylabel("Угол (рад)")
        axs[0].legend(loc='lower right')

        axs[1].plot(dfs[0]["x"], dfs[0]["v2"], label="Численное решение", color="red")
        axs[1].set_title("График зависимости угловой скорости от времени")
        axs[1].set_xlabel("Время (с)")
        axs[1].set_ylabel("Угловая скорость (рад/с)")
        axs[1].legend(loc='lower right')

        axs[2].plot(dfs[0]["v1"], dfs[0]["v2"], label="Численное решение", color="red")
        axs[2].set_title("Фазовый портрет")
        axs[2].set_xlabel("Угол (рад)")
        axs[2].set_ylabel("Угловая скорость (рад/с)")
        axs[2].legend(loc='lower right')

        plt.show()

    plot_graphs()

def show_graph2():
    border_eps = float(border_eps_entry.get())
    max_periods = float(max_periods_entry.get())
    eps = float(maxError_entry.get())
    h_0 = float(h0_entry.get())
    max_steps = int(max_steps_entry.get())
    x_0 = 0
    l = 0.1
    g = 9.8

    initial_conditions = [
        {"v1_0": 0.6, "v2_0": 0, "color": "red"},
        {"v1_0": 0.3, "v2_0": 0, "color": "purple"},
        {"v1_0": 0.6, "v2_0": 1, "color": "orange"}
    ]

    results = [sol_var1(x_0, condition["v1_0"], condition["v2_0"], l, g, h_0, max_steps, eps, max_periods, border_eps)
               for condition in initial_conditions]

    dfs = [pd.DataFrame(r, columns=["x", "v1", "v2", "v1_1/2", "v2_1/2", "|v1-v1_1/2|", "|v2-v2_1/2|", "h", "Норма ОЛП", "div",
                                    "doub"]) for r in results]

    def truncate_data_for_period(df, period):
        # Оставляем данные только для одного периода
        return df[df["x"] <= period]

    def plot_graphs():
        fig, axs = plt.subplots(2, 3, figsize=(18, 18))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # Первая строка графиков
        for condition in initial_conditions:
            df = dfs[initial_conditions.index(condition)]
            label = f'v1_0={condition["v1_0"]}, v2_0={condition["v2_0"]}'

            period = 2 * math.pi * (0.1 / g) ** 0.5  # Период для l = 0.1
            truncated_df = truncate_data_for_period(df, period)

            axs[0, 0].plot(truncated_df["x"], truncated_df["v1"], label=label, color=condition["color"])
            axs[0, 1].plot(truncated_df["x"], truncated_df["v2"], label=label, color=condition["color"])
            axs[0, 2].plot(truncated_df["v1"], truncated_df["v2"], label=label, color=condition["color"])

        axs[0, 0].set_title("Угол от времени для разных н.у")
        axs[0, 0].set_xlabel("Время (с)")
        axs[0, 0].set_ylabel("Угол (рад)")
        axs[0, 0].legend(loc='lower right')
        axs[0, 1].set_title("Угловая скорость от времени для разных н.у")
        axs[0, 1].set_xlabel("Время (с)")
        axs[0, 1].set_ylabel("Угловая скорость (рад/с)")
        axs[0, 1].legend(loc='lower right')

        axs[0, 2].set_title("Фазовые портреты для разных н.у")
        axs[0, 2].set_xlabel("Угол (рад)")
        axs[0, 2].set_ylabel("Угловая скорость (рад/с)")
        axs[0, 2].legend(loc='lower right')

        # Параметры для расчета
        conditions = [
            {"v1_0": 0.3, "v2_0": 0, "g": 9.8, "l": 0.1, "label": "g=9.8, l=0.1", "color": "red"},
            {"v1_0": 0.3, "v2_0": 0, "g": 2, "l": 0.1, "label": "g=2, l=0.1", "color": "purple"},
            {"v1_0": 0.3, "v2_0": 0, "g": 9.8, "l": 2, "label": "g=9.8, l=2", "color": "orange"}
        ]

        # Вторая строка графиков
        for condition in conditions:
            # Вычисление решения для каждой пары параметров
            results = sol_var1(0, condition["v1_0"], condition["v2_0"], condition["l"], condition["g"], 0.01, 10000, 1e-6, 3,
                          1e-4)
            df = pd.DataFrame(results, columns=["x", "v1", "v2", "v1_1/2", "v2_1/2", "|v1-v1_1/2|", "|v2-v2_1/2|", "h", "Норма ОЛП", "div", "doub"])

            axs[1, 0].set_xlim(-0.1, 2)  # Ограничиваем временную ось
            axs[1,0].set_ylim(-1, 1)  # Ограничиваем углы/скорости
            axs[1, 1].set_xlim(-0.1, 2)  # Ограничиваем временную ось
            axs[1, 1].set_ylim(-6, 6)  # Ограничиваем углы/скорости
            axs[1, 2].set_xlim(-0.4, 0.4)  # Ограничиваем временную ось
            axs[1, 2].set_ylim(-3.5, 3.5)  # Ограничиваем углы/скорости

            period = 2 * math.pi * (condition["l"] / condition["g"]) ** 0.5
            truncated_df = truncate_data_for_period(df, period)

            axs[1, 0].plot(truncated_df["x"], truncated_df["v1"], label=condition["label"], color=condition["color"])
            axs[1, 1].plot(truncated_df["x"], truncated_df["v2"], label=condition["label"], color=condition["color"])
            axs[1, 2].plot(truncated_df["v1"], truncated_df["v2"], label=condition["label"], color=condition["color"])

        axs[1, 0].set_title("Угол от времени для разных парам")
        axs[1, 0].set_xlabel("Время (с)")
        axs[1, 0].set_ylabel("Угол (рад)")
        axs[1, 0].legend(loc='lower right')

        axs[1, 1].set_title("Угловая скорость от времени для разных парам")
        axs[1, 1].set_xlabel("Время (с)")
        axs[1, 1].set_ylabel("Угловая скорость (рад/с)")
        axs[1, 1].legend(loc='lower right')

        axs[1, 2].set_title("Фазовый портрет для разных парам")
        axs[1, 2].set_xlabel("Угол (рад)")
        axs[1, 2].set_ylabel("Угловая скорость (рад/с)")
        axs[1, 2].legend(loc='lower right')

        plt.show()

    plot_graphs()

def show_graph3():
    border_eps = float(border_eps_entry.get())
    max_periods = float(max_periods_entry.get())
    eps = float(maxError_entry.get())
    h_0 = float(h0_entry.get())
    max_steps = int(max_steps_entry.get())
    x_0 = float(x0_entry.get())
    v10 = float(v10_entry.get())
    v20 = float(v20_entry.get())
    l = float(l_entry.get())
    g = 9.8  # ускорение свободного падения

    # Для первого варианта
    results_var1 = [sol_var1(x_0, v10, v20, l, g, h_0, max_steps, eps, max_periods, border_eps)]
    dfs_var1 = [pd.DataFrame(r, columns=["x", "v1", "v2", "v1_1/2", "v2_1/2", "|v1-v1_1/2|", "|v2-v2_1/2|", "h", "Норма ОЛП", "div",
                                    "doub"]) for r in results_var1]

    # Для второго варианта
    results_var2 = [sol_var2(x_0, v10, v20, l, g, h_0, max_steps, eps, max_periods, border_eps)]
    dfs_var2 = [pd.DataFrame(r, columns=["x", "v1", "v2", "v1_1/2", "v2_1/2", "|v1-v1_1/2|", "|v2-v2_1/2|", "h", "Норма ОЛП", "div",
                                         "doub"]) for r in results_var2]

    def plot_graphs():
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # Для первого варианта
        axs[0].plot(dfs_var1[0]["x"], dfs_var1[0]["v1"], label="1 вариант (рад)", color="red")
        axs[0].set_title("График зависимости угла от времени")
        axs[0].set_xlabel("Время (с)")
        axs[0].set_ylabel("Угол (рад)")
        axs[0].legend(loc='lower right')

        axs[1].plot(dfs_var1[0]["x"], dfs_var1[0]["v2"], label="1 вариант", color="red")
        axs[1].set_title("График зависимости угловой скорости от времени")
        axs[1].set_xlabel("Время (с)")
        axs[1].set_ylabel("Угловая скорость (рад/с)")
        axs[1].legend(loc='lower right')

        axs[2].plot(dfs_var1[0]["v1"], dfs_var1[0]["v2"], label="1 вариант", color="red")
        axs[2].set_title("Фазовый портрет")
        axs[2].set_xlabel("Угол (рад)")
        axs[2].set_ylabel("Угловая скорость (рад/с)")
        axs[2].legend(loc='lower right')

        # Для второго варианта
        axs[0].plot(dfs_var2[0]["x"], dfs_var2[0]["v1"], label="2 вариант (рад)", color="green")
        axs[0].set_title("График зависимости угла от времени")
        axs[0].set_xlabel("Время (с)")
        axs[0].set_ylabel("Угол (рад)")
        axs[0].legend(loc='lower right')

        axs[1].plot(dfs_var2[0]["x"], dfs_var2[0]["v2"], label="2 вариант", color="green")
        axs[1].set_title("График зависимости угловой скорости от времени")
        axs[1].set_xlabel("Время (с)")
        axs[1].set_ylabel("Угловая скорость (рад/с)")
        axs[1].legend(loc='lower right')

        axs[2].plot(dfs_var2[0]["v1"], dfs_var2[0]["v2"], label="2 вариант", color="green")
        axs[2].set_title("Фазовый портрет")
        axs[2].set_xlabel("Угол (рад)")
        axs[2].set_ylabel("Угловая скорость (рад/с)")
        axs[2].legend(loc='lower right')

        plt.show()

    plot_graphs()

def show_graph4():
    border_eps = float(border_eps_entry.get())
    max_periods = float(max_periods_entry.get())
    max_steps = int(max_steps_entry.get())
    v1_0 = 0.3
    v2_0 = 0.0
    x_0 = 0
    l = 0.1
    g = 9.8

    def truncate_data_for_period(df, period):
        # Оставляем данные только для одного периода
        return df[df["x"] <= period]

    def plot_graphs():
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # Параметры для расчета
        conditions = [
            {"h": 0.0001, "eps": 1e-13, "label": "h=0.0001, eps=1e-13", "color": "red"},
            {"h": 0.0001, "eps": 1e-3, "label": "h=0.0001, eps=1e-3", "color": "purple"},
            {"h": 0.05, "eps": 1e-13, "label": "h=0.05, eps=1e-13", "color": "orange"}
        ]

        for condition in conditions:
            # Вычисление решения для каждой пары параметров
            results = sol_var1(x_0, v1_0, v2_0, l, g, condition["h"], max_steps, condition["eps"], max_periods,
                          border_eps)
            df = pd.DataFrame(results, columns=["x", "v1", "v2", "v1_1/2", "v2_1/2", "|v1-v1_1/2|", "|v2-v2_1/2|", "h", "Норма ОЛП", "div", "doub"])

            period = 2 * math.pi * (l / g) ** 0.5
            truncated_df = truncate_data_for_period(df, period)

            axs[0].plot(truncated_df["x"], truncated_df["v1"], label=condition["label"], color=condition["color"])
            axs[1].plot(truncated_df["x"], truncated_df["v2"], label=condition["label"], color=condition["color"])
            axs[2].plot(truncated_df["v1"], truncated_df["v2"], label=condition["label"], color=condition["color"])

        axs[0].set_title("Угол от времени для разных h и eps")
        axs[0].set_xlabel("Время (с)")
        axs[0].set_ylabel("Угол (рад)")
        axs[0].legend(loc='lower right')

        axs[1].set_title("Угловая скорость от времени для разных h и eps")
        axs[1].set_xlabel("Время (с)")
        axs[1].set_ylabel("Угловая скорость (рад/с)")
        axs[1].legend(loc='lower right')

        axs[2].set_title("Фазовый портрет для разных h и eps")
        axs[2].set_xlabel("Угол (рад)")
        axs[2].set_ylabel("Угловая скорость (рад/с)")
        axs[2].legend(loc='lower right')

        plt.show()

    plot_graphs()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Численные методы задание 11 вариант 1")

    frame = ttk.Frame(root)
    frame.pack(padx=10, pady=10)

    root.geometry("1100x300")
    root.minsize(1100, 300)

    window_width = 1100
    window_height = 300

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    maxError_label = ttk.Label(frame, text="Eps. контроля:") # параметр контроля лп
    maxError_label.grid(row=0, column=0)
    maxError_entry = ttk.Entry(frame)
    maxError_entry.grid(row=0, column=1)
    maxError_entry.insert(0, "1e-13")

    max_periods_label = ttk.Label(frame, text="Макс. период:")
    max_periods_label.grid(row=2, column=0)
    max_periods_entry = ttk.Entry(frame)
    max_periods_entry.grid(row=2, column=1)
    max_periods_entry.insert(0, "1")

    max_steps_label = ttk.Label(frame, text="max_steps:")
    max_steps_label.grid(row=3, column=0)
    max_steps_entry = ttk.Entry(frame)
    max_steps_entry.grid(row=3, column=1)
    max_steps_entry.insert(0, "1000")

    border_eps_label = ttk.Label(frame, text="Eps. граничный:")
    border_eps_label.grid(row=4, column=0)
    border_eps_entry = ttk.Entry(frame)
    border_eps_entry.grid(row=4, column=1)
    border_eps_entry.insert(0, "0.0001")

    h0_label = ttk.Label(frame, text="h0:")
    h0_label.grid(row=0, column=2)
    h0_entry = ttk.Entry(frame)
    h0_entry.grid(row=0, column=3)
    h0_entry.insert(0, "0.0001")

    v10_label = ttk.Label(frame, text="v1_0:") # изначальный угол
    v10_label.grid(row=1, column=2)
    v10_entry = ttk.Entry(frame)
    v10_entry.grid(row=1, column=3)
    v10_entry.insert(0, "0.314")

    v20_label = ttk.Label(frame, text="v2_0:") # изначальная угловая скорость
    v20_label.grid(row=2, column=2)
    v20_entry = ttk.Entry(frame)
    v20_entry.grid(row=2, column=3)
    v20_entry.insert(0, "0.0")

    x0_label = ttk.Label(frame, text="x_0:")
    x0_label.grid(row=3, column=2)
    x0_entry = ttk.Entry(frame)
    x0_entry.grid(row=3, column=3)
    x0_entry.insert(0, "0")

    l_label = ttk.Label(frame, text="l:")
    l_label.grid(row=4, column=2)
    l_entry = ttk.Entry(frame)
    l_entry.grid(row=4, column=3)
    l_entry.insert(0, "0.1")

    task_var = tk.StringVar()

    plot_param_button = ttk.Button(frame, text="Численное решение", command=show_graph1)
    plot_param_button.grid(row=0, column=30, columnspan=2)

    plot_param_button = ttk.Button(frame, text="Для разных н.у. и параметров", command=show_graph2)
    plot_param_button.grid(row=1, column=30, columnspan=2)

    plot_param_button = ttk.Button(frame, text="Сравнение со вторым вариантом", command=show_graph3)
    plot_param_button.grid(row=2, column=30, columnspan=2)

    plot_param_button = ttk.Button(frame, text="Для разных h и eps", command=show_graph4)
    plot_param_button.grid(row=3, column=30, columnspan=2)

    plot_param_button = ttk.Button(frame, text="Справка и таблица", command=show_table)
    plot_param_button.grid(row=4, column=30, columnspan=2)

    root.mainloop()