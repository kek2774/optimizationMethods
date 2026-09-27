import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def phi(x: np.ndarray, i: int, t: float):
    x_temp = x.copy()
    x_temp[i] = t
    return get_my_func(x_temp)


x: np.ndarray = np.random.uniform(-100, 100, 3)  # генерация рандомной точки


print(f"Initial point x = {x}")

iterations: list = []
coords_iterations: list = []

start = time.perf_counter()

for i in range(max_iterations):
    prev_x = x.copy()

    for j in range(len(x)):
        x[j] = dihotomy(-1e7, 1e7, lambda t: phi(x, j, t))

    x_diff: np.ndarray = x - prev_x

    iterations.append(i)
    coords_iterations.append(x.copy())

    # условие досрочной остановки
    if np.sum(np.abs(x_diff)) < min_eps:
        break


end = time.perf_counter()

# таблица итераций
df = pd.DataFrame({"iteration": iterations, "x": coords_iterations})

print(df)

# найденный экстремум
print(f"Extr = {x}")

print(f"Execution time: {end - start:.6f}")
print(f"Iterations: {len(iterations)}")

plt.plot(iterations, [-val for val in [get_my_func(el) for el in coords_iterations]])
plt.xlabel("Итерация")
plt.ylabel("Значение исходной функции")
plt.title("Сходимость метода Гаусса-Зейделя")
plt.grid()
plt.show()

df.to_csv("result_gauss_zeidel.txt", index=False)
