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

# Convertimos PERFIL CLIENTE y TIPO TRANSACCION en tablas categoricas de tipo int 

percli = pd.Categorical(df["PERFIL CLIENTE"]).codes
tipotr = pd.Categorical(df["TIPO TRANSACCION"]).codes


# Creamos nuevas tablas para separar y almacenar las transacciones fraudulentas y las legítimas

percliok= percli[df['INDICADOR FRAUDE'] == 0]
tipotrok= tipotr[df['INDICADOR FRAUDE'] == 0]
perclifr= percli[df['INDICADOR FRAUDE'] == 1]
tipotrfr = tipotr[df['INDICADOR FRAUDE'] == 1]


percliok = percliok.astype(int)
tipotrok = tipotrok.astype(int)
perclifr = perclifr.astype(int)
tipotrfr = tipotrfr.astype(int)



# Dibujamos el gráfico
# La intensidad del color de los puntos indica mayor densidad de transacciones

plt.figure(figsize =(12, 7))

plt.xticks(np.arange(0,24,1),rotation=45, labels = (["HG02","HG03","HG04","HG05","HG06","HG07","HG09","HG14","HG15","HG18","HN01","HN02","HN04","HN05","HN08","HN09","HN12","HN13","HN14","HN15","HN16","NG01","NG02","NG03"]))
plt.yticks(np.arange(0,32,1), labels = (["H910","H920","H925","H940","H945","H950","H955","H980","H98C","H990","I8S1","I8S7","J7C0","J7E0","L381","L6C0","L6E0","L6E1","L6LB","LE04","LE08","LNLI","LNLO","LS10","NC32","NC33","NC34","NC35","NC39","NC3A","NC3B","P390"]))


plt.xlabel("PERFIL DE CLIENTE", labelpad=15)
plt.ylabel("TIPO DE TRANSACCION",labelpad=15)
plt.title(label="FRAUDE POR PERFIL DE CLIENTE Y TIPO DE TRANSACCION",fontsize=15, color="black")    

plt.scatter(percliok, tipotrok, c= "green", marker = "+", s=300, alpha=0.13)
plt.scatter(perclifr, tipotrfr, c= "red", marker = "o", s=300, alpha=0.13)

plt.show()

