import streamlit as st

st.title("🐦 Dashboard Avifauna & Variables Climáticas")


import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn import metrics

st.set_page_config(
    page_title="Avifauna & Clima — Dashboard",
    page_icon="🕊️",
    layout="wide",
)

st.title("🐦 Dashboard Avifauna & Variables Climáticas")
st.caption("Explora avistamientos por especie y su relación con variables climáticas. Filtra, compara y prepara insumos para tu modelo predictivo.")
st.info('Modelo multivariante para predecir abundancia y diversidad de aves según variables climáticas en el campus de la ESPOL ')

@st.cache_data(show_spinner=True)
def load_dataset(zip_path: Path) -> pd.DataFrame:
    """Carga el conjunto de datos desde el archivo ZIP.

    Args:
        zip_path: Ruta al archivo ZIP que contiene los datos.

    Returns:
        DataFrame con las columnas definidas en la descripción.

    Esta función busca el primer archivo CSV dentro del ZIP y lo lee con
    ``pandas.read_csv``. Las fechas en ``YEAR_MONTH`` se convierten en
    datetime, y el campo ``MONTH`` se asegura como entero.
    """
    if not zip_path.exists():
        st.error(f"El archivo {zip_path} no existe en el repositorio.")
        return pd.DataFrame()

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Obtiene el nombre del primer archivo CSV dentro del ZIP
        csv_files = [name for name in zf.namelist() if name.lower().endswith('.csv')]
        if not csv_files:
            st.error("No se encontró ningún archivo CSV dentro del ZIP.")
            return pd.DataFrame()
        data_name = csv_files[0]
        with zf.open(data_name) as f:
            df = pd.read_csv(f)

    # Conversión de tipos
    if 'YEAR_MONTH' in df.columns:
        try:
            df['YEAR_MONTH'] = pd.to_datetime(df['YEAR_MONTH'], errors='coerce')
        except Exception:
            pass
    if 'MONTH' in df.columns:
        df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce').astype('Int64')

    return df


@st.cache_data(show_spinner=True)
def load_models(model_paths: Dict[str, Path]) -> Dict[str, object]:
    """Carga los modelos de regresión logística desde archivos pickle.

    Args:
        model_paths: Mapeo entre nombre de especie y ruta al archivo .pkl

    Returns:
        Diccionario que asocia cada especie con su modelo (o None si no se
        encuentra el archivo).
    """
    models = {}
    for species, path in model_paths.items():
        if path.exists():
            try:
                models[species] = joblib.load(path)
            except Exception as exc:
                st.warning(f"No se pudo cargar el modelo de {species}: {exc}")
                models[species] = None
        else:
            models[species] = None
    return models


@st.cache_data(show_spinner=True)
def load_logistic_model(model_path: Path):
    """Carga un único modelo de regresión logística que aplica a todas las especies.

    Args:
        model_path: Ruta al archivo .pkl del modelo logístico.

    Returns:
        El modelo cargado o ``None`` si no se encuentra el archivo.
    """
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as exc:
            st.warning(f"No se pudo cargar el modelo logístico: {exc}")
            return None
    else:
        return None


def get_species_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Obtiene un DataFrame con pares únicos de nombre común y científico.

    Esto garantiza que cada combinación sea única y evita combinaciones
    incorrectas.

    Args:
        df: Conjunto de datos completo.

    Returns:
        DataFrame con columnas ``COMMON NAME`` y ``SCIENTIFIC NAME`` sin
        duplicados.
    """
    species_df = df[['COMMON NAME', 'SCIENTIFIC NAME']].drop_duplicates().copy()
    species_df = species_df.sort_values('COMMON NAME').reset_index(drop=True)
    return species_df


def filter_by_species(df: pd.DataFrame, common_name: str) -> pd.DataFrame:
    """Filtra el DataFrame por especie utilizando el nombre común.

    Args:
        df: Conjunto de datos completo.
        common_name: Nombre común seleccionado.

    Returns:
        Subconjunto del DataFrame con solo las filas correspondientes a la
        especie.
    """
    return df[df['COMMON NAME'] == common_name].copy()


def plot_boxplot(df: pd.DataFrame, variable: str) -> None:
----
def plot_boxplot(df: pd.DataFrame, variable: str) -> None:
    """Genera un diagrama de cajas (boxplot) de una variable por mes.

    Esta función permite visualizar la distribución de la variable
    seleccionada (climática o ``LOG_AVISTAMIENTOS``) agrupada por
    ``MONTH``. Se crea un boxplot por cada mes, mostrando los puntos
    individuales para facilitar la exploración de la variabilidad.

    Args:
        df: DataFrame ya filtrado por especie.
        variable: Nombre de la columna a graficar. Debe existir en ``df``.

    Si la variable o la columna ``MONTH`` no existen en ``df``, se
    mostrará un mensaje de advertencia.
    """
    if 'LOG_AVISTAMIENTOS' not in df.columns or 'MONTH' not in df.columns:
        st.warning("No se encuentran las columnas LOG_AVISTAMIENTOS o MONTH en los datos.")
        return

    try:
        # Crear el boxplot con Plotly, agrupando por mes
        fig = px.box(
            df,
            x='MONTH',
            y='LOG_AVISTAMIENTOS',
            points="all",
            title=f"Distribución mensual de avistamientos por mes"
        )
        fig.update_layout(
            xaxis_title="Mes",
            yaxis_title='Log10(Avistamientos + 1)'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("No se pudo generar el diagrama de cajas debido a un error con los datos.")
        ---


def plot_variable_importance(model, feature_names: List[str]) -> None:
    """Grafica la importancia de las variables de un modelo logístico.

    Args:
        model: Modelo entrenado (debe tener atributo ``coef_``).
        feature_names: Lista de nombres de las variables en el mismo orden
            en que fueron utilizadas para entrenar el modelo.

    La importancia se calcula a partir de los coeficientes de la regresión
    logística. Se utiliza el valor absoluto para ordenar las variables.
    """
    if model is None:
        st.info("No se ha cargado un modelo para esta especie.")
        return

    try:
        coefs = model.coef_.ravel()
    except Exception as exc:
        st.warning(f"El modelo no contiene coeficientes accesibles: {exc}")
        return

    importance_df = pd.DataFrame({
        'Variable': feature_names,
        'Coeficiente': coefs,
        'Importancia': np.abs(coefs)
    }).sort_values('Importancia', ascending=False)

    fig = px.bar(importance_df, x='Variable', y='Importancia', color='Coeficiente',
                 color_continuous_scale='RdBu', title="Importancia de variables")
    fig.update_layout(xaxis_title="Variable", yaxis_title="|Coeficiente|")
    st.plotly_chart(fig, use_container_width=True)


def plot_time_series(df: pd.DataFrame, variables: List[str]) -> None:
    """Grafica series de tiempo para las variables seleccionadas.

    Args:
        df: DataFrame filtrado por especie.
        variables: Lista de variables climáticas a mostrar.

    Se agrupan los datos por ``YEAR_MONTH`` y se calcula la media de cada
    variable. Cada variable se grafica en la misma figura para facilitar la
    comparación temporal.
    """
    if not variables:
        st.info("Seleccione al menos una variable climática para visualizar la serie de tiempo.")
        return

    # Asegurarse de que ``YEAR_MONTH`` sea de tipo datetime
    if df['YEAR_MONTH'].dtype != 'datetime64[ns]':
        try:
            df['YEAR_MONTH'] = pd.to_datetime(df['YEAR_MONTH'], errors='coerce')
        except Exception:
            st.warning("No se pudo convertir YEAR_MONTH a formato fecha.")

    grouped = df.groupby('YEAR_MONTH')[variables].mean().reset_index()
    fig = go.Figure()
    for var in variables:
        fig.add_trace(go.Scatter(x=grouped['YEAR_MONTH'], y=grouped[var],
                                 mode='lines', name=var))
    fig.update_layout(title="Series de tiempo de variables climáticas",
                      xaxis_title="Fecha",
                      yaxis_title="Valor")
    st.plotly_chart(fig, use_container_width=True)


def plot_top_n_birds(df: pd.DataFrame, n: int) -> None:
    """Muestra un gráfico con las N especies más avistadas.

    Args:
        df: Conjunto de datos completo.
        n: Número de especies a mostrar.

    Ordena las especies por su total de avistamientos y grafica las primeras
    ``n``. Se utilizan los campos ``COMMON NAME`` y ``AVISTAMIENTOS``.
    """
    if 'AVISTAMIENTOS' not in df.columns:
        st.warning("No se encuentra la columna AVISTAMIENTOS en los datos.")
        return
    agg_df = df.groupby('COMMON NAME')['AVISTAMIENTOS'].sum().sort_values(ascending=False).head(n).reset_index()
    fig = px.bar(agg_df, x='AVISTAMIENTOS', y='COMMON NAME', orientation='h',
                 title=f"Top {n} especies por número de avistamientos")
    fig.update_layout(xaxis_title="Total de avistamientos", yaxis_title="Especie")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Dashboard de avistamientos de aves", layout="wide")
    st.title("Dashboard interactivo de avistamientos de aves")

    # Carga del dataset
    data_path = Path('output.zip')
    df = load_dataset(data_path)
    if df.empty:
        st.stop()

    # Obtener mapeo entre nombre común y científico
    species_mapping = get_species_mapping(df)

    # Sidebar para seleccionar especie y variables
    st.sidebar.header("Filtros")
    selected_common_name = st.sidebar.selectbox(
        "Seleccione el nombre común de la especie:",
        options=species_mapping['COMMON NAME'],
        index=0
    )
    # Obtener nombre científico correspondiente
    selected_scientific_name = species_mapping.loc[
        species_mapping['COMMON NAME'] == selected_common_name, 'SCIENTIFIC NAME'
    ].iloc[0]
    st.sidebar.markdown(f"**Nombre científico:** {selected_scientific_name}")

    # Filtrar datos por especie
    species_df = filter_by_species(df, selected_common_name)

    # Variables climáticas candidatas (columna excepto identificadores y variables de respuesta)
    climate_vars = [
        'PRECTOTCORR', 'PS', 'QV2M', 'RH2M', 'T2M', 'T2MDEW', 'T2MWET',
        'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WD10M', 'WD2M', 'WS10M',
        'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS2M', 'WS2M_MAX',
        'WS2M_MIN', 'WS2M_RANGE'
    ]
    available_vars = [v for v in climate_vars if v in df.columns]

    # Widget para seleccionar variable de boxplot
    selected_var_box = st.sidebar.selectbox(
        "Variable climática para boxplot:", available_vars, index=0
    )

    # Widget para seleccionar variables para la serie de tiempo
    selected_vars_time = st.sidebar.multiselect(
        "Variables climáticas para series de tiempo:", available_vars, default=[]
    )

    # Top N de avistamientos (fuera del filtro de especie)
    st.sidebar.markdown("---")
    max_n = min(20, species_mapping.shape[0])  # limite de especies a mostrar
    n_top = st.sidebar.slider("Número de especies para Top N", 5, max_n, 5)

    # Carga de modelos predictivos
    st.sidebar.markdown("---")
    st.sidebar.header("Modelos predictivos")
    # Modelos para predecir cantidades (uno por especie)
    model_paths = {
        'Especie 1': Path('model_especie1.pkl'),
        'Especie 2': Path('model_especie2.pkl'),
        'Especie 3': Path('model_especie3.pkl')
    }
    models = load_models(model_paths)
    # Modelo logístico general para presencia/ausencia
    logistic_model_path = Path('model_logistic.pkl')
    logistic_model = load_logistic_model(logistic_model_path)

    # Contenido principal
    st.subheader(f"Especie seleccionada: {selected_common_name}")

    # Boxplot
    st.markdown("### Distribución de la variable climática")
    plot_boxplot(species_df, selected_var_box)

    # Resultados del modelo logístico para la especie seleccionada
    st.markdown("### Modelo logístico general (importancia de variables y métricas)")
    if logistic_model is not None:
        # Importancia de variables (coeficientes)
        plot_variable_importance(logistic_model, available_vars)
        # Métricas para la especie seleccionada: se debe haber creado la columna PRESENCIA
        if 'PRESENCIA' in df.columns:
            X_spec = species_df[available_vars].dropna()
            if not X_spec.empty:
                y_spec = species_df.loc[X_spec.index, 'PRESENCIA']
                acc_spec, auc_spec = compute_model_metrics(logistic_model, X_spec, y_spec)
                st.write(f"Exactitud para la especie seleccionada: {acc_spec:.2f}")
                st.write(f"AUC para la especie seleccionada: {auc_spec:.2f}")
            else:
                st.info("No hay suficientes datos para evaluar el modelo logístico en esta especie.")
        else:
            st.info("No se ha podido calcular la métrica porque la variable PRESENCIA no está disponible.")
    else:
        st.info("No se ha cargado un modelo logístico general. Asegúrese de incluir el archivo 'model_logistic.pkl'.")

    # Series de tiempo de variables climáticas
    st.markdown("### Series de tiempo de variables climáticas")
    plot_time_series(species_df, selected_vars_time)

    # Top N especies más avistadas
    st.markdown("### Top N especies por avistamientos (en todo el conjunto de datos)")
    plot_top_n_birds(df, n_top)

    # Comparación de modelos
    st.markdown("## Comparación de modelos por especie")
    with st.expander("Ver comparación de modelos"):
        # Preparar variable objetivo (presencia/no presencia) a partir de AVISTAMIENTOS
        if 'AVISTAMIENTOS' in df.columns:
            df['PRESENCIA'] = (df['AVISTAMIENTOS'] > 0).astype(int)
        else:
            st.warning("No se encuentra la columna AVISTAMIENTOS para construir la variable objetivo.")
            return
        # Métricas y curvas para cada modelo cargado
        for key, model in models.items():
            st.markdown(f"### {key}")
            if model is None:
                st.info("Modelo no disponible.")
                continue
            # Filtrar datos para la especie del modelo
            species_name = key
            # Se busca la fila correspondiente en species_mapping
            try:
                common_name_model = species_mapping.loc[
                    species_mapping['COMMON NAME'].str.contains(species_name.split()[-1], case=False), 'COMMON NAME'
                ].iloc[0]
            except Exception:
                # Si no se encuentra coincidencia, se usan todos los datos
                common_name_model = None
            if common_name_model:
                df_model = filter_by_species(df, common_name_model)
            else:
                df_model = df
            
    st.markdown("---")
    st.caption("Aplicación desarrollada para visualizar avistamientos y variables climáticas de aves.")


if __name__ == '__main__':
    main()
