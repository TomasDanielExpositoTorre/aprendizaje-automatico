import pandas as pd


class DataTable:
    def __init__(self, filename: str = "datos_practica.txt", sep: str = '|'):
        self.table: pd.DataFrame = pd.read_csv(filename, sep=sep, header=None)
        self.table.columns = [
            "FECHA TRANSACCION", "HORA TRANSACCION", "CLIENTE ID", "PERFIL CLIENTE", "SEGMENTO", "IP",
            "MODO ACCESO", "ID SESION", "IMPORTE", "TIPO MENSAJE", "CANAL", "FECHA SESION", "HORA SESION",
            "MEDIO AUTENTIFICACION", "TIPO TRANSACCION", "ENTIDAD", "OFICINA ORIGEN", "CUENTA ORIGEN",
            "ENTIDAD DESTINO", "OFICINA DESTINO", "CUENTA DESTINO", "TIPO FIRMA", "TIPO CUENTA ORIGEN",
            "PAIS DESTINO", "FECHA ALTA CANAL", "FECHA ACTIVACION CANAL", "FECHA NAC TITU CTA CARGO",
            "FECHA ALTA CTA CARGO", "PAIS IP", "LATITUD", "LONGITUD", "BROWSER", "BROWSER VERSION", "OS",
            "OS VERSION", "PROFESION CLIENTE", "SECTOR CLIENTE", "SEGMENTO CLIENTE", "INDICADOR FRAUDE"
        ]
        self.ncols = len(self.table.columns)
        pass

    def col(self, i: int):
        if i < -(self.ncols) or i >= self.ncols:
            raise ValueError(f"Given index of {i} is out of bounds for a table of {self.ncols} columns")
        return self.table[self.table.columns[i]]

    def col(self, name: str = None):
        if name.upper() not in self.table.columns:
            raise ValueError(f"'{name.upper()}' is not a column in this table")
        return self.table[name.upper()]