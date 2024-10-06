import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV, ignore empty columns

df = pd.read_csv(
    "datos_practica.txt",
    sep="|", header=None, skipinitialspace=True
)

df.columns = [
    "FECHA TRANSACCION", "HORA TRANSACCION", "CLIENTE ID", "PERFIL CLIENTE", "SEGMENTO", "IP",
    "MODO ACCESO", "ID SESION", "IMPORTE", "TIPO MENSAJE", "CANAL", "FECHA SESION", "HORA SESION",
    "MEDIO AUTENTIFICACION", "TIPO TRANSACCION", "ENTIDAD", "OFICINA ORIGEN", "CUENTA ORIGEN",
    "ENTIDAD DESTINO", "OFICINA DESTINO", "CUENTA DESTINO", "TIPO FIRMA", "TIPO CUENTA ORIGEN",
    "PAIS DESTINO", "FECHA ALTA CANAL", "FECHA ACTIVACION CANAL", "FECHA NAC TITU CTA CARGO",
    "FECHA ALTA CTA CARGO", "PAIS IP", "LATITUD", "LONGITUD", "BROWSER", "BROWSER VERSION", "OS",
    "OS VERSION", "PROFESION CLIENTE", "SECTOR CLIENTE", "SEGMENTO CLIENTE", "INDICADOR FRAUDE"
]


# Seleccionamos el par de columnas SECTOR CLI y SEGMENTO CLI, que junto con INDICADOR FRAUDE
# servirá para hallar el vector de características.

# Extraemos los valores de cada columna y los convertimos en una lista de números enteros
# Definimos fraude como variable booleana (True = Fraude / False = No Fraude)


df.sec = ["SECTOR CLIENTE"]
df.seg = ["SEGMENTO CLIENTE"]
df.ind = ["INDICADOR FRAUDE"]

x = df.sec[0]
y = df.seg[0]
fraude = df.ind[0]


for filas in df.sec:

    x = np.array(df[filas])

    x = [0 if element == 'M   ' else element for element in x]
    x = [1 if element == 'F   ' else element for element in x]
    x = [2 if element == 'G   ' else element for element in x]
    x = [3 if element == '' else element for element in x]

for filas in df.seg:

    y = np.array(df[filas])
    y = y.astype(int)
    y = y.tolist()

# Definimos simbolo y color de los puntos según sea fraude o no y dibujamos el gráfico

    color = "black"
    simbolo ="D"
    fraude is True


for filas in df.ind:

    fraude = (df[filas])
    fraude = fraude.astype(bool)
    fraude = fraude.tolist()
    
        if  fraude is True:
        color = "red"
        simbolo = "+"

        if  fraude is False:
        color = "green"   
        simbolo = "o"

    plt.figure(figsize =(10, 7))
    plt.xticks([0,1,2], labels = (["SECTOR M","SECTOR F", "SECTOR G"]))
    plt.yticks([11,13,23,31,32,33,43], labels = (["SEG 11","SEG 13","SEG 23","SEG 31","SEG 32","SEG 33","SEG 43"]))
    plt.xlabel("SECTOR CLIENTE", labelpad=15)
    plt.ylabel("SEGMENTO CLIENTE",labelpad=15)
    plt.title(label="FRAUDE POR SECTOR Y SEGMENTO DE CLIENTE",fontsize=15, color="black")
    plt.scatter(x, y, c = color, marker = simbolo,s = 100, alpha=0.15)      

plt.show()
