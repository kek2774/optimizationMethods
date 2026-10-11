import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def gauss_zeidel_method(
    history: list[dict] | None = None,
    minimization_method: Callable = dihotomy,
    min_eps: float = 1e-5,
    max_iterations: int = 2000,
) -> np.ndarray:

    x: np.ndarray = np.random.uniform(-100, 100, 3)  # генерация рандомной точки

    for i in range(max_iterations):
        prev_x = x.copy()

        for j in range(len(x)):
            x[j] = minimization_method(-1e7, 1e7, lambda t: phi(x, j, t))

        x_diff: np.ndarray = x - prev_x

        if history is not None:
            history.append(
                {
                    "iteration": i + 1,
                    "x": x.copy(),
                    "f(x)": -get_my_func(x),
                }
            )

        # условие досрочной остановки
        if np.sum(np.abs(x_diff)) < min_eps:
            break

    return x


history: list[dict] = []

start = time.perf_counter()

x = gauss_zeidel_method(
    history=history,
    min_eps=min_eps,
    max_iterations=max_iterations,
)

end = time.perf_counter()

# таблица итераций
df = pd.DataFrame(history)

print(df)

# найденный экстремум
print(f"Extr = {x}")

print(f"Execution time: {end - start:.6f}")
print(f"Iterations: {len(history)}")

# график сходимости
plt.plot(df["iteration"], df["f(x)"])
plt.xlabel("Итерация")
plt.ylabel("Значение исходной функции")
plt.title("Сходимость метода Гаусса-Зейделя")
plt.grid()
plt.show()

df.to_csv("result_gauss_zeidel.txt", index=False)
