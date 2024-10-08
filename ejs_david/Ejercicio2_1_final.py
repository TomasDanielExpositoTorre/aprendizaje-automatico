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

# Convertimos SECTOR y SEGMENTO CLIENTE en tablas categoricas de tipo int 

sec = pd.Categorical(df["SECTOR CLIENTE"]).codes
seg = pd.Categorical(df["SEGMENTO CLIENTE"]).codes


# Creamos nuevas tablas para separar y almacenar las
# transacciones fraudulentas (segfr / secfr) y las legítimas (segok / secok)

segfr = seg[df['INDICADOR FRAUDE'] == 1]
secfr = sec[df['INDICADOR FRAUDE'] == 1]

segok = seg[df['INDICADOR FRAUDE'] == 0]
secok = sec[df['INDICADOR FRAUDE'] == 0]

# Dibujamos el gráfico
# La intensidad del color de los puntos indica mayor densidad de transacciones

plt.figure(figsize =(10, 7))
plt.xticks([-1,0,1,2], labels = ([" DESC","SECTOR M","SECTOR F","SECTOR G"]))
plt.yticks([-1,0,1,2,3,4,5,6],labels = (["DESC","SEG 11","SEG 13","SEG 23","SEG 31","SEG 32","SEG 33","SEG43"]))
plt.xlabel("SECTOR CLIENTE", labelpad=15)
plt.ylabel("SEGMENTO CLIENTE",labelpad=15)
plt.title(label="FRAUDE POR SECTOR Y SEGMENTO DE CLIENTE",fontsize=15, color="black")    

plt.scatter(secok, segok, c= "green", marker = "+", s=200, alpha=0.13)
plt.scatter(secfr, segfr, c= "red", marker = "x", s=200, alpha=0.13)

plt.show()

