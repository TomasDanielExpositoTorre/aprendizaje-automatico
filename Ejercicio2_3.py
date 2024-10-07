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

# Convertimos PERFIL CLIENTE en tabla categorica de tipo int 

percli = pd.Categorical(df["PERFIL CLIENTE"]).codes
importe = np.array(df["IMPORTE"], dtype=float)


# Creamos nuevas tablas para separar y almacenar las
# transacciones fraudulentas (perclifr / importefr) y las legítimas (percliok / importeok)

perclifr = percli[df['INDICADOR FRAUDE'] == 1]
importefr = (importe[df['INDICADOR FRAUDE'] == 1])

percliok = percli[df['INDICADOR FRAUDE'] == 0]
importeok = (importe[df['INDICADOR FRAUDE'] == 0])


# Dibujamos el gráfico
# La intensidad del color de los puntos indica mayor densidad de transacciones

plt.figure(figsize =(12, 8))
plt.title(label="FRAUDE POR PERFIL DE CLIENTE E IMPORTE",fontsize=15, color="black") 
plt.xlabel("PERFIL DE CLIENTE", labelpad=10,fontsize=12)
plt.ylabel("IMPORTE en miles de Euros", labelpad=10,fontsize=12)

plt.xticks([-1,0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24],rotation=45, labels = (['DESC','HG02','HG03','HG04','HG05','HG06','HG07','HG09','HG14','HG15','HG18','HN01','HN02','HN04','HN05','HN06','HN07','HN09','HN14','HN15','HN16','NG01','NG02','NG03']))
plt.yticks([-1,0,10000,20000,30000,400000,50000], labels=(['0','10k','20k','30k','40k','50k','60k o más']))



plt.scatter(percliok,importeok, c= "green", marker = "+", s=200)
plt.scatter(perclifr,importefr, c= "red", marker = "x", s=200)

plt.show()

