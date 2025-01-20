from func import find_first_nonzero_indices
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


def fit_trendlines(x, y):
    # Convertir fechas a números para los cálculos
    x_num = np.arange(len(x))

    # Función para ajuste exponencial
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Calcular diferentes ajustes
    fits = {}

    # Lineal
    z = np.polyfit(x_num, y, 1)
    p = np.poly1d(z)
    fits['Linear'] = {
        'y_pred': p(x_num),
        'r2': r2_score(y, p(x_num)),
        'func': lambda x: p(x)
    }

    # Polinomial (grado 2)
    z = np.polyfit(x_num, y, 2)
    p = np.poly1d(z)
    fits['Polynomial'] = {
        'y_pred': p(x_num),
        'r2': r2_score(y, p(x_num)),
        'func': lambda x: p(x)
    }

    # Logarítmico
    try:
        z = np.polyfit(np.log(x_num + 1), y, 1)
        fits['Logarithmic'] = {
            'y_pred': z[0] * np.log(x_num + 1) + z[1],
            'r2': r2_score(y, z[0] * np.log(x_num + 1) + z[1]),
            'func': lambda x: z[0] * np.log(x + 1) + z[1]
        }
    except:
        pass

    # Exponencial
    try:
        popt, _ = curve_fit(exp_func, x_num, y, p0=[1, 0.1, 1])
        y_pred = exp_func(x_num, *popt)
        fits['Exponential'] = {
            'y_pred': y_pred,
            'r2': r2_score(y, y_pred),
            'func': lambda x: exp_func(x, *popt)
        }
    except:
        pass

    # Encontrar el mejor ajuste
    best_fit = max(fits.items(), key=lambda x: x[1]['r2'])
    return best_fit


# Inicializar variables fuera del sidebar
df = None
options = []
indices_seleccionados = {}

# Sidebar para carga y selección
with st.sidebar:
    uploaded_files = st.file_uploader(
        "Choose a CSV file", type="csv"
    )

    if uploaded_files:
        df = pd.read_csv(uploaded_files)
        columns = df.columns.tolist()
        df['date'] = pd.to_datetime(df['date'])
        columna_a_eliminar = "date"
        if columna_a_eliminar in columns:
            columns.remove(columna_a_eliminar)

        # Sección 1: Visualización individual
        st.subheader('Visualización Individual')
        options = st.multiselect(
            "Seleccione las columnas que quiere graficar",
            columns,
        )

        # Sección 2: Comparación de variables
        st.subheader('Comparación de Variables')

        # Lista para almacenar los pares de variables seleccionados
        if 'num_comparisons' not in st.session_state:
            st.session_state.num_comparisons = 1

        comparisons = []

        # Crear campos de selección dinámicamente
        for i in range(st.session_state.num_comparisons):
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox(f"Variable 1 #{i + 1}", columns, key=f"var1_{i}")
            with col2:
                var2 = st.selectbox(f"Variable 2 #{i + 1}", columns, key=f"var2_{i}")
            comparisons.append((var1, var2))

        # Botón para agregar más comparaciones
        if st.button("Agregar otra comparación"):
            st.session_state.num_comparisons += 1
            # st.experimental_rerun()

        st.subheader('Herramienta de visualizacion')
        herramienta = st.radio(
            "Elige tu herramienta de visualizacion",
            ["Seaborn", "Matplotlib", "Plotly"],
            index=None,
        )

        if st.button("Graficar"):
            indices_seleccionados = {col: find_first_nonzero_indices(df)[col]
                                     for col in options if col in find_first_nonzero_indices(df)}


# Contenido principal (fuera del sidebar)
if df is not None:
    # Primero mostrar las gráficas individuales si hay selecciones
    if indices_seleccionados and herramienta:
        st.header("Gráficas Individuales")
        for column in options:
            if column in indices_seleccionados:
                st.subheader(f"Gráfica de {column}")
                # Crear un DataFrame filtrado específico para cada columna
                df_filtrado = df[['date', column]].iloc[indices_seleccionados[column]:].copy()
                df_filtrado = df_filtrado.dropna()
                df_filtrado[column] = df_filtrado[column].astype(float).astype(int)
                # Calcular la línea de tendencia específica para esta columna
                best_fit_type, best_fit_data = fit_trendlines(
                    np.arange(len(df_filtrado)),
                    df_filtrado[column].values.copy()  # Usar una copia de los valores
                )

                if herramienta == "Plotly":
                    # Crear una figura nueva para cada gráfica
                    fig = px.line(df_filtrado, x='date', y=column,
                                  title=f'{column} vs Time (Trendline: {best_fit_type}, R² = {best_fit_data["r2"]:.3f})')

                    # Agregar línea de tendencia específica para esta gráfica
                    fig.add_scatter(
                        x=df_filtrado['date'],
                        y=best_fit_data['y_pred'],
                        name=f'{best_fit_type} trendline',
                        line=dict(color='red')
                    )

                    fig.update_xaxes(title='Date')
                    fig.update_yaxes(title=column)
                    st.plotly_chart(fig, use_container_width=True)

                elif herramienta == "Seaborn" or herramienta == "Matplotlib":
                    # Crear una figura nueva para cada gráfica
                    plt.clf()  # Limpiar cualquier figura anterior
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if herramienta == "Seaborn":
                        sns.lineplot(data=df_filtrado, x='date', y=column, ax=ax)
                    else:
                        ax.plot(df_filtrado['date'], df_filtrado[column])

                    # Agregar línea de tendencia específica para esta gráfica
                    ax.plot(df_filtrado['date'],
                            best_fit_data['y_pred'],
                            'r-',
                            label=f'{best_fit_type} trendline')

                    plt.xticks(rotation=45)
                    plt.title(f'{column} vs Time\nTrendline: {best_fit_type}, R² = {best_fit_data["r2"]:.3f}')
                    plt.xlabel('Date')
                    plt.ylabel(column)
                    plt.legend()
                    st.pyplot(fig)
                    plt.close()

    # Luego mostrar las gráficas de comparación
    # En la parte de las gráficas comparativas:
    if comparisons and herramienta:
        st.header("Gráficas Comparativas")
        for i, (var1, var2) in enumerate(comparisons):
            st.subheader(f"Comparación {i + 1}: {var1} vs {var2}")

            # Obtener los índices de las variables seleccionadas
            indices = find_first_nonzero_indices(df)
            if var1 in indices and var2 in indices:
                # Encontrar el índice mínimo entre las dos variables
                min_index = min(indices[var1], indices[var2])
            else:
                min_index = 0
                # Crear DataFrame con ambas variables a partir del índice mínimo
            df_comp = df[['date', var1, var2]].iloc[min_index:].copy()

            # Calcular los límites del eje Y
            y_min = min(df_comp[var1].min(), df_comp[var2].min())
            y_max = max(df_comp[var1].max(), df_comp[var2].max())
            # Añadir un margen del 10%
            y_margin = (y_max - y_min) * 0.1
            y_min -= y_margin
            y_max += y_margin

            if herramienta == "Seaborn":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df_comp, x='date', y=var1, label=var1)
                sns.lineplot(data=df_comp, x='date', y=var2, label=var2)
                plt.xticks(rotation=45)
                plt.ylim(y_min, y_max)  # Establecer límites del eje Y
                plt.title(f'Comparación: {var1} vs {var2}\nDatos desde el índice {min_index}')
                plt.xlabel('Date')
                plt.legend()
                st.pyplot(fig)
                plt.close()

            elif herramienta == "Matplotlib":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_comp['date'], df_comp[var1], label=var1)
                ax.plot(df_comp['date'], df_comp[var2], label=var2)
                plt.xticks(rotation=45)
                plt.ylim(y_min, y_max)  # Establecer límites del eje Y
                plt.title(f'Comparación: {var1} vs {var2}\nDatos desde el índice {min_index}')
                plt.xlabel('Date')
                plt.legend()
                st.pyplot(fig)
                plt.close()

            elif herramienta == "Plotly":
                fig = px.line(df_comp, x='date', y=[var1, var2],
                              title=f'Comparación: {var1} vs {var2}\nDatos desde el índice {min_index}')
                fig.update_xaxes(title='Date')
                fig.update_yaxes(range=[y_min, y_max])  # Establecer límites del eje Y
                st.plotly_chart(fig, use_container_width=True)
