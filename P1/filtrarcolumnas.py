# Filtrar columnas en la tabla cuyos registros solo tienen un valor o ninguno

import pandas as pd
import p1
import matplotlib.pyplot as plt

LABELS=0
VALUES=1

# Load the database and compute frequencies

data = p1.DataTable()
freqs = data.frequencies()
freqs2 = data.frequencies()

# Duplicamos freqs en freqs2 para poder borrar las columnas durante la iteración
# si no da error "RuntimeError: dictionary changed size during iteration"

for key in freqs2:

    if len(freqs2[key][LABELS]) <= 1:
        print("Borramos las columnas ",key)
        del (freqs[key])


for key in freqs:
    print("Quedan las columnas ",key)

