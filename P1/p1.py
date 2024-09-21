import pandas as pd


class DataTable:
    def __init__(self, filename: str = "datos_practica.txt", sep: str = '|'):
        """
        Class constructor. Reads a database from the given filename and sanitizes
        fields which contain spaces.

        Args:
            filename (str, optional): filename. Defaults to "datos_practica.txt".
            sep (str, optional): column separator. Defaults to '|'.
        """
        self.table: pd.DataFrame = pd.read_csv(
            filename,
            sep=sep, header=None, skipinitialspace=True
        )
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

    def frequencies(self):
        """
        Computes the value frequency for each column in the table.

        Returns:
            A dictionary containing the number of values per column, and a list
            with the label and frequency for each value.
        """
        freqs: dict = {}
        
        for value in self.table.columns:
            ifreqs = self.table.value_counts(value)
            keys = list(ifreqs.keys())
            freqs[value] = [keys, [ifreqs[k] for k in keys]]

        return freqs
