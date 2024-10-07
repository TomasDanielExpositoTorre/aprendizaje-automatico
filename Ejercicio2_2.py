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

# Convertimos MEDIO AUTENTIFICACION y TIPO TRANSACCION en tablas categoricas de tipo int 

medaut = pd.Categorical(df["MEDIO AUTENTIFICACION"]).codes
tipotr = pd.Categorical(df["TIPO TRANSACCION"]).codes


# Creamos nuevas tablas para separar y almacenar las
# transacciones fraudulentas (medautfr / tipotrfr) y las legítimas (medautok / tipotrok)

medautfr = medaut[df['INDICADOR FRAUDE'] == 1]
tipotrfr = tipotr[df['INDICADOR FRAUDE'] == 1]

medautok = medaut[df['INDICADOR FRAUDE'] == 0]
tipotrok = tipotr[df['INDICADOR FRAUDE'] == 0]

# Dibujamos el gráfico
# La intensidad del color de los puntos indica mayor densidad de transacciones

plt.figure(figsize =(12, 8))


plt.xlabel("MEDIO AUTENTIFICACION", labelpad=5,fontsize=12)
plt.ylabel("TIPO TRANSACCION",labelpad=5,fontsize=12)
plt.title(label="FRAUDE POR MEDIO DE AUTENTIFICACION y TIPO DE TRANSACCION",fontsize=15, color="black") 

plt.xticks([-1,0,1,2,3,4,5], labels = (['DESC','D3','S1','S2','T1','T3','T6']))
plt.yticks([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32], labels = (["DESC","H910","H920","H925","H940","H945","H950","H955","H980","H98C","H990","I8S1","I8S7","J7C0","J7E0","L381","L6C0","L6E0","L6E1","L6LB","LE04","LE08","LNLI","LNLO","LS10","NC32","NC33","NC34","NC35","NC39","NC3A","NC3B","P390"]))


plt.scatter(medautok, tipotrok, c= "green", marker = "+", s=200, alpha=0.13)
plt.scatter(medautfr, tipotrfr, c= "red", marker = "x", s=200, alpha=0.13)

plt.show()

