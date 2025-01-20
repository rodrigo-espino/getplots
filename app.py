from func import find_first_nonzero_indices
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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
            st.experimental_rerun()

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
                df_filtrado = df[['date', column]].iloc[indices_seleccionados[column]:]

                if herramienta == "Seaborn":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(data=df_filtrado, x='date', y=column, ax=ax)
                    plt.xticks(rotation=45)
                    plt.title(f'{column} vs Time')
                    plt.xlabel('Date')
                    plt.ylabel(column)
                    st.pyplot(fig)
                    plt.close()

                elif herramienta == "Matplotlib":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(df_filtrado['date'], df_filtrado[column])
                    plt.xticks(rotation=45)
                    plt.title(f'{column} vs Time')
                    plt.xlabel('Date')
                    plt.ylabel(column)
                    st.pyplot(fig)
                    plt.close()

                elif herramienta == "Plotly":
                    fig = px.line(df_filtrado, x='date', y=column,
                                  title=f'{column} vs Time')
                    fig.update_xaxes(title='Date')
                    fig.update_yaxes(title=column)
                    st.plotly_chart(fig, use_container_width=True)

    # Luego mostrar las gráficas de comparación
    if comparisons and herramienta:
        st.header("Gráficas Comparativas")
        for i, (var1, var2) in enumerate(comparisons):
            st.subheader(f"Comparación {i + 1}: {var1} vs {var2}")

            # Crear DataFrame con ambas variables
            df_comp = df[['date', var1, var2]].copy()

            if herramienta == "Seaborn":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df_comp, x='date', y=var1, label=var1)
                sns.lineplot(data=df_comp, x='date', y=var2, label=var2)
                plt.xticks(rotation=45)
                plt.title(f'Comparación: {var1} vs {var2}')
                plt.xlabel('Date')
                plt.legend()
                st.pyplot(fig)
                plt.close()

            elif herramienta == "Matplotlib":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_comp['date'], df_comp[var1], label=var1)
                ax.plot(df_comp['date'], df_comp[var2], label=var2)
                plt.xticks(rotation=45)
                plt.title(f'Comparación: {var1} vs {var2}')
                plt.xlabel('Date')
                plt.legend()
                st.pyplot(fig)
                plt.close()

            elif herramienta == "Plotly":
                fig = px.line(df_comp, x='date', y=[var1, var2],
                              title=f'Comparación: {var1} vs {var2}')
                fig.update_xaxes(title='Date')
                st.plotly_chart(fig, use_container_width=True)