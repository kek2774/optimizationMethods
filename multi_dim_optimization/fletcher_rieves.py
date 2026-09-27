import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from one_dimensional_optimization import dihotomy

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
np.random.seed(42)

max_iterations: int = 2000  # максимальное число итераций
min_eps: float = (
    1e-5  # минимальная разница между точками для продолжения работы алгоритма
)
round_acc: int = 3  # до скольки цифр округлять


def get_my_func(x: np.ndarray) -> float:
    """
    Вычисляет минус значение целевой функции в точке x

    Args:
        x (np.ndarray): точка в 3х мерном пространстве

    Returns:
        float: значение функции в точке x
    """
    x1: float = x[0]
    x2: float = x[1]
    x3: float = x[2]
    return -1 * (
        -6 * np.square(x1)
        - np.square(x2)
        - 2 * np.square(x3)
        + x1 * x2
        - 2 * x1 * x3
        + 4 * x1
        - 5 * x3
    )


def get_my_func_grad(x: np.ndarray) -> np.ndarray:
    """
    Вычисляет минус значение градиента целевой функции в точке x

    Args:
        x (np.ndarray): точка в 3х мерном пространстве

    Returns:
        np.ndarray: вектор градиентов
    """

    # производная по x1
    def x1_der(x: np.ndarray) -> float:
        return -1 * (-12 * x[0] + x[1] - 2 * x[2] + 4)

    # производная по x2
    def x2_der(x: np.ndarray) -> float:
        return -1 * (-2 * x[1] + x[0])

    # производная по x3
    def x3_der(x: np.ndarray) -> float:
        return -1 * (-4 * x[2] - 2 * x[0] - 5)

    return np.array([x1_der(x), x2_der(x), x3_der(x)])


x: np.ndarray = np.random.uniform(-100, 100, 3)  # генерация рандомной точки

# инициализация вспомогательных списков для хранения данных итераций
iterations: list = []
x_k: list = []
x_k_plus_1: list = []
t_arr: list = []
minus_f_x_k_plus_1_arr: list = []
x_diff_arr: list = []

print(f"Initial point x = {x}")

start = time.perf_counter()


for i in range(max_iterations):

    if i == 0:
        grad: np.ndarray = get_my_func_grad(x)  # градиент в точке
        p: np.ndarray = grad
        t: float = dihotomy(
            0, 10, lambda t: get_my_func(x - t * p)
        )  # коэффициент шага градиентного спуска
        prev_x: np.ndarray = x.copy()  # точка до изменения

        # обновление точки
        x -= t * p
    else:
        new_grad: np.ndarray = get_my_func_grad(x)
        beta: float = np.square(np.linalg.norm(new_grad)) / np.square(np.linalg.norm(grad))
        p = new_grad + beta * p
        grad = new_grad

        t: float = dihotomy(
            0, 10, lambda t: get_my_func(x - t * p)
        )  # коэффициент шага градиентного спуска
        prev_x: np.ndarray = x.copy()  # точка до изменения

        # обновление точки
        x -= t * p
        

    x_diff = x - prev_x
    

    iterations.append(i)
    t_arr.append(round(t, round_acc))
    x_k.append(np.round(prev_x, round_acc))
    x_k_plus_1.append(np.round(x, round_acc))
    minus_f_x_k_plus_1_arr.append(np.round(get_my_func(x), round_acc))
    x_diff_arr.append(np.round(x - prev_x, round_acc))

    # условие досрочной остановки
    if np.sum(np.abs(x_diff)) < min_eps:
        break
end = time.perf_counter()

# таблица итераций
df = pd.DataFrame(
    {
        "iteration": iterations,
        "t": t_arr,
        "x_k": x_k,
        "x_k+1": x_k_plus_1,
        "x_diff": x_diff_arr,
        "f(x_k+1)": [-val for val in minus_f_x_k_plus_1_arr],
    }
)

print(df.head())
print(df.tail())

# найденный экстремум
extr: np.ndarray = x_k_plus_1[-1]
print(f"Extr = {extr}")

print(f"Execution time: {end - start:.6f}")
print(f"Iterations: {len(iterations)}")

df.to_csv("result_fletcher_rieves.txt", index=False)


plt.plot(iterations, [-val for val in minus_f_x_k_plus_1_arr])
plt.xlabel("Итерация")
plt.ylabel("Значение исходной функции")
plt.title("Сходимость метода Флетчера-Ривза")
plt.grid()
plt.show()
