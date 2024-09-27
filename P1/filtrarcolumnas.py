# Depuración de datos 
# Eliminación de columnas sin información significativa (registros vacios o con un único valor)

import pandas as pd

# Load the database and compute frequencies

LABELS=0
freqs = data.frequencies()

freqs2 == freqs

# Duplicamos freqs en freqs2 para poder eliminar columnas mientras estamos dentro del bucle,
# evitando el error "RuntimeError: dictionary changed size during iteration"

for key in freqs2:

    if  len(freqs[key][LABELS]) <= 1:

        print ("Borramos la columna ", key)

        del (freqs[key])
 
       
