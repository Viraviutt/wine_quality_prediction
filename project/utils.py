import pandas as pd

def save_name_columns(df: pd.DataFrame) -> list:
    return df.columns.tolist()[0].replace(';"', ';').replace('";', ';').strip('"').split(';')

def separate_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Separar las columnas usando ';' como delimitador
    df_separated = df.iloc[:, 0].str.split(';', expand=True)
    # Asignar nombres de columnas adecuados
    df_separated.columns = save_name_columns(df)
    return df_separated

def change_dtype(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def int_to_float(value: int) -> float:
    try:
        return float(value)
    except ValueError:
        return None