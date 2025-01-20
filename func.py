def find_first_nonzero_indices(df, exclude_columns=['date']):
    """
    Encuentra el índice del primer valor no-cero para cada columna numérica.

    Args:
        df: DataFrame de pandas
        exclude_columns: Lista de columnas a excluir del análisis

    Returns:
        Dict con el índice del primer valor no-cero para cada columna
    """
    indices = {}

    # Obtener columnas numéricas (excluyendo las columnas especificadas)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    columns_to_process = [col for col in numeric_columns if col not in exclude_columns]

    for column in columns_to_process:
        # Encontrar el primer valor no-cero
        first_non_zero = None
        for idx, value in enumerate(df[column]):
            if value != 0:
                first_non_zero = idx
                break

        indices[column] = first_non_zero if first_non_zero is not None else len(df)

    return indices