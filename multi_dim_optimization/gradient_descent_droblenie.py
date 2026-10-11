import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        float: минус значение функции в точке x
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


def gradient_descent_droblenie(
    func: Callable,
    func_grad: Callable,
    history: list[dict] | None = None,
    min_eps: float = 1e-5,
    max_iterations: int = 2000,
    c: float = 0.9,
) -> np.ndarray:

    x: np.ndarray = np.random.uniform(-100, 100, 3)

    for i in range(max_iterations):
        t: float = 1.0
        prev_x: np.ndarray = x.copy()
        grad: np.ndarray = func_grad(x)

        my_func_curr: float = func(x)
        my_func_next: float = func(x - t * grad)

        # дробление шага
        while my_func_next > my_func_curr:
            t *= c
            my_func_next = func(x - t * grad)

        # обновление точки
        x -= t * grad
        x_diff = x - prev_x

        if history is not None:
            history.append(
                {
                    "iteration": i + 1,
                    "t": t,
                    "c": c,
                    "x_k": prev_x.copy(),
                    "x_k+1": x.copy(),
                    "x_diff": x_diff.copy(),
                    "f(x_k+1)": -func(x),
                }
            )

        # условие досрочной остановки
        if np.sum(np.abs(x_diff)) < min_eps:
            break

    return x


history: list[dict] = []

start = time.perf_counter()
extr = gradient_descent_droblenie(
    func=get_my_func,
    func_grad=get_my_func_grad,
    history=history,
    min_eps=min_eps,
    max_iterations=max_iterations,
)
end = time.perf_counter()


# таблица итераций
df = pd.DataFrame(history)

print(df.head())
print(df.tail())

# найденный экстремум
print(f"Extr = {extr}")

print(f"Execution time: {end - start:.6f}")
print(f"Iterations: {len(history)}")


df.to_csv("result_gradient_descent_droblenie.txt", index=False)


plt.plot(df["iteration"], df["f(x_k+1)"])
plt.xlabel("Итерация")
plt.ylabel("Значение исходной функции")
plt.title("Сходимость градиентного спуска (дробление)")
plt.grid()
plt.show()
