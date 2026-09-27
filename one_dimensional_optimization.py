from typing import Callable
from numpy import sqrt

golden_section_number = (sqrt(5) + 1) / 2 # число золотого сечения


def dihotomy(
    a0: float, b0: float, func: Callable, delta: float = 1e-5, eps: float = 1e-3
) -> float:
    """
    Метод дихотомии для поиска минимума унимодальной функции одной переменной на отрезке [a, b]

    Args:
        a0 (float): левый конец отрезка
        b0 (float): правый конец отрезка
        func (Callable): унимодальная функция
        delta (float, optional): параметр метода дихотомии, расстояние от центра отрезка. Defaults to 1e-5.
        eps (float, optional): требуемая точность ответа. Defaults to 1e-3.

    Returns:
        float: минимум функции
    """
    a: float = a0
    b: float = b0

    while abs(a - b) > eps:
        x1: float = (a + b) / 2 - delta
        x2: float = (a + b) / 2 + delta

        f1: float = func(x1)
        f2: float = func(x2)

        if f1 < f2:
            b = x2

        else:
            a = x1

    return (a + b) / 2


def golden_section(a0: float, b0: float, func: Callable, eps: float = 1e-3) -> float:
    """
    Метод золотого сечения для поиска минимума унимодальной функции одной переменной на отрезке [a, b]

    Args:
        a0 (float): левый конец отрезка
        b0 (float): правый конец отрезка
        func (Callable): унимодальная функция
        eps (float, optional): требуемая точность ответа. Defaults to 1e-3.

    Returns:
        float: минимум функции
    """

    a: float = a0
    b: float = b0
    while abs(a - b) < eps:
        x1 = b - (b - a) / golden_section_number
        x2 = a + (b - a) / golden_section_number

        f1: float = func(x1)
        f2: float = func(x2)

        if f1 < f2:
            b = x2

        else:
            a = x1

    return (a + b) / 2

def calculate_fibonacci_numb(num: int) -> int:
    """
    Функция для вычисления num-того числа в последовательности Фибоначчи

    Args:
        num (int): номер числа в последовательности Фибоначчи

    Returns:
        int: число из последовательности Фибоначчи
    """

    prev: int = 0
    current: int = 1

    for _ in range(num):
        temp: int = current
        current += prev
        prev = temp

    return prev

def fibonacci(a0: float, b0: float, func: Callable, steps: int) -> float:
    """
    Метод Фибоначчи для поиска минимума унимодальной функции одной переменной на отрезке [a, b] с заданным числом шагов

    Args:
        a0 (float): левая граница отрезка
        b0 (float): правая граница отрезка
        func (Callable): унимодальная функция
        steps (int): число шагов

    Returns:
        float: минимум
    """

    a: float = a0
    b: float = b0
    for n in range(steps - 3):
        m: int = steps - n

        x1 = a + calculate_fibonacci_numb(m - 2) / calculate_fibonacci_numb(m) * (b - a)
        x2 = a + calculate_fibonacci_numb(m - 1) / calculate_fibonacci_numb(m) * (b - a)

        f1: float = func(x1)
        f2: float = func(x2)

        if f1 < f2:
            b = x2

        else:
            a = x1

    return (a + b) / 2