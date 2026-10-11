from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd

dim: int = 4
max_iter: int = 100


def f(x: np.ndarray) -> float:
    return x[0] + x[1] - 2 * x[2] + x[3]


def simplex_method(
    B: list,
    c: dict,
    gamma: dict,
    beta: dict,
    dim: int,
    history: list[dict] | None = None,
    max_iter: int = 100,
    finding: Literal["max", "min"] = "min",
) -> np.ndarray:  # type: ignore
    """
    Вычисляет заданное в finding значение (min/max) с заданными ограничениями

    Args:
        B (list): вектор с индексами базисных переменных
        c (dict): коэффициенты в исходной функции
        gamma (dict): коэффициенты в ограничениях
        beta (dict): свободные члены
        dim (int): размерность пространства
        history(list[dict] | None = None): таблица итераций. Defaults to None
        max_iter (int, optional): максимальное количество итераций. Defaults to 100.
        finding (Literal["max", "min"], optional): какую точку находим (минимума или максимума). Defaults to "min".

    Returns:
        np.ndarray: точка минимума или максимума
    """

    B = deepcopy(B)
    c = deepcopy(c)
    gamma = deepcopy(gamma)
    beta = deepcopy(beta)

    x: np.ndarray = np.zeros(dim)
    delta: dict = {}
    NB: list = [i for i in range(1, dim + 1) if i not in B]  # свободные индексы

    for iteration in range(max_iter):
        x = np.zeros(dim)

        # базисное решение
        for i in B:
            x[i - 1] = beta[i]

        for j in NB:
            i_sum: float = 0.0
            for i in B:
                i_sum += c[i] * gamma[i][j]
            delta[j] = c[j] - i_sum

        if history is not None:
            history.append(
                {
                    "Итерация": iteration,
                    "Базис": B.copy(),
                    "Точка": x.copy(),
                    "f(x)": f(x),
                    "delta": delta.copy(),
                    "Входит": None,
                    "Выходит": None,
                    "rho": None,
                }
            )

        enters: int = 0
        leaves: int = 0

        min_delta_idx: int = np.inf  # type:ignore

        stop_condition: bool = (
            all([delta[i] >= 0 for i in NB])
            if finding == "min"
            else all([delta[i] <= 0 for i in NB])
        )
        if stop_condition:
            return x
        else:
            for j in NB:
                if delta[j] < delta.get(min_delta_idx, np.inf):
                    min_delta_idx = j

            enters = min_delta_idx

            rho: float = np.inf

            for i in beta:
                b: float = gamma[i][enters]
                if b > 0:
                    t = beta[i] / b
                    if t < rho:
                        rho = t
                        leaves = i

            if history is not None:
                history[-1].update({"Входит": enters, "Выходит": leaves, "rho": rho})

            # выражаем входящую переменную

            p = gamma[leaves][enters]

            new_beta = {}
            new_gamma = {}

            new_beta[enters] = beta[leaves] / p
            new_gamma[enters] = {leaves: 1 / p}

            # пересчитываем остальные уравнения

            for j in NB:
                if j != enters:
                    new_gamma[enters][j] = gamma[leaves][j] / p

            for i in B:
                if i == leaves:
                    continue

                a = gamma[i][enters]

                new_beta[i] = beta[i] - a * new_beta[enters]
                new_gamma[i] = {}

                for j in new_gamma[enters]:
                    new_gamma[i][j] = gamma[i].get(j, 0) - a * new_gamma[enters][j]

            B = [enters if i == leaves else i for i in B]
            NB = [leaves if j == enters else j for j in NB]

            beta = new_beta
            gamma = new_gamma


B: list = [1, 2]  # базисные индексы
c: dict = {1: 1, 2: 1, 3: -2, 4: 1}  # коэффициенты исходной функции
gamma: dict = {
    1: {3: 0.5, 4: 1.5},
    2: {3: 0.5, 4: 0.5},
}  # коэффициенты перед переменными
beta: dict = {1: 16, 2: 7}  # свободные члены
min_history: list[dict] = []
max_history: list[dict] = []

x_max: np.ndarray = simplex_method(B, c, gamma, beta, dim, max_history, finding="max")
x_min: np.ndarray = simplex_method(B, c, gamma, beta, dim, min_history, finding="min")

df_min = pd.DataFrame(min_history)
df_max = pd.DataFrame(max_history)
print(f"x_max = {x_max}, f_max = {f(x_max)}")
print(df_max)
print(f"x_min = {x_min}, f_min = {f(x_min)}")
print(df_min)
