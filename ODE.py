# -*- coding: utf-8 -*-

"""
Python 3
01 / 08 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime
import os

import tools

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(f"{os.getlogin()}| {datetime.now()}")

# ----------------------------- #### --------------------------
from typing import Callable


# ####################################################################
def ODE_euler(
    *,
    a: float,
    b: float,
    f: Callable[[float, float], float],
    y_t0: float,
    N: int,
) -> tuple[list[float], list[float], float]:
    """Solves (numerically) an ODE of the form
        dy/dt = f(t, y)
            y(t_0) = y_t0, a <= t_0 <= b
    using the Euler method for the N+1 points in the time range [a, b].

    It generates N+1 mesh points with:
        t_i = a + i*h, h = (a - b) / N,
    where h is the step size.


    ## Parameters
    ``a``: initial time
    ``b``: final time
    ``f``: function of two variables ``t`` and ``y``
    ``y_t0``: initial condition
    ``N``: number of mesh points

    ## Return
    ``ys``: a list of the N+1 approximated values of y
    ``ts``: a list of the N+1 mesh points
    ``h``: the step size h

    """
    h = (b - a) / N
    t = a
    ts = [t]
    ys = [y_t0]

    for _ in range(N):
        y = ys[-1]
        y += h * f(t, y)
        ys.append(y)

        t += h
        ts.append(t)
    return ys, ts, h


# ####################################################################
from math import factorial


def ODE_euler_nth(
    *,
    a: float,
    b: float,
    f: Callable[[float, float], float],
    f_derivatives: list[Callable[[float, float], float]],
    y_t0: float,
    N: int,
) -> tuple[list[float], list[float], float]:
    """Solves (numerically) an ODE of the form
        dy/dt = f(t, y)
            y(t_0) = y_t0, a <= t_0 <= b
    using the Taylor method with (m - 1)th derivatives for the N+1 points in the time range [a, b].

    It generates N+1 mesh points with:
        t_i = a + i*h, h = (a - b) / N,
    where h is the step size.


    ## Parameters
    ``a``: initial time
    ``b``: final time
    ``f``: function of two variables ``t`` and ``y``
    ``f_derivatives``: list of (m - 1)th derivatives of f
    ``y_t0``: initial condition
    ``N``: number of mesh points

    ## Return
    ``ys``: a list of the N+1 approximated values of y
    ``ts``: a list of the N+1 mesh points
    ``h``: the step size h

    """
    h = (b - a) / N
    t = a
    ts = [t]
    ys = [y_t0]

    for _ in range(N):
        y = ys[-1]
        T = f(t, y)
        ders = [
            h / factorial(m + 2) * mth_derivative(t, y)
            for m, mth_derivative in enumerate(f_derivatives)
        ]
        T += sum(ders)
        y += h * T
        ys.append(y)

        t += h
        ts.append(t)
    return ys, ts, h


# ####################################################################
import numpy as np
import matplotlib.pyplot as plt
def f(t, y):
    return -5*y + 5*t**2 + 2*t
a, b = 0, 1
y_t0 = 1/3
N = 10
h = (b - a) / N
ts = np.linspace(a, b, N+1)
ys = np.zeros(N+1)
ys[0] = y_t0
for i in range(N):
    ys[i+1] = ys[i] + h * f(ts[i], ys[i])


print('ts',ts)
print('ys',ys)
def real(t):
    return t**2 + 1/3* np.exp(-5*t)
plt.plot(ts,real(ts),'-s',color='red')
plt.title('solucion real')
plt.plot(ts, ys, marker='o', linestyle='-', label="Euler Aproximado")
plt.xlabel("t")
plt.ylabel("y")
plt.title("Solución usando el Método de Euler")
plt.legend()
plt.grid(True)
plt.show()

def calcular_error_relativo_promedio(N):
    h = (b - a) / N
    ts = np.linspace(a, b, N+1)
    ys = np.zeros(N+1)
    ys[0] = y_t0

    for i in range(N):
        ys[i+1] = ys[i] + h * f(ts[i], ys[i])
    errores_relativos = np.abs((ys - real(ts)) / real(ts))
    return np.mean(errores_relativos)
Ns = [10, 5, 20]
errores_promedios = {N: calcular_error_relativo_promedio(N) for N in Ns}
print(errores_promedios)





