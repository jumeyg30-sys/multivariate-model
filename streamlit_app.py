import streamlit as st
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
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
import calendar  # Para convertir números de mes a nombres
import streamlit as st
from PIL import Image
import streamlit as st
import base64

# Ruta de tu imagen de fondo (ajusta el nombre y la ubicación según sea necesario)
fondo_path = "espol_fondo.png"

# Leer y codificar la imagen en base64
with open(fondo_path, "rb") as file:
    encoded = base64.b64encode(file.read()).decode()

  
# Inyectar el CSS para el fondo con capa morada atenuada
st.markdown(
    f"""
    <style>
    .stApp {{
        /* Aplica primero la capa morada semitransparente y luego la imagen de fondo */
        background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(128, 0, 128, 0.3)),
                          url(data:image/jpeg;base64,{encoded});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Nombres de los archivos de imágenes que se utilizarán en el dashboard.
# Asegúrate de colocar estas imágenes en el mismo directorio que este archivo o actualiza las rutas.


MODEL_IMAGE_FILES = ["saffron2.png", "ecuadorian2.png", "blue2.png"]  # Imágenes de los modelos predictivos
MODEL_IMAGE_DESCRIPTIONS = [
    "Se espera que el número de avistamientos sea similar o ligeramente mayor que en años anteriores. En resumen, podríamos ver alrededor de 100 avistamientos de Saffron Finch en 2025, con un pequeño aumento en comparación con los años previos.",
    "El modelo predice que en 2025 los avistamientos del Ecuadorian Ground Dove tendrán ligera tendencia a bajar en comparación con los años anteriores. Se indica que la cantidad de avistamientos podría estabilizarse en un nivel bajo.",
    "El modelo predice que se espera que los avistamientos de la especie Blue Grays podrían estabilizarse en niveles bajos, con fluctuaciones a lo largo del año. No se anticipa un aumento significativo."
]

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


def compute_model_metrics(model, X, y):
    """Calcula las métricas del modelo, como la exactitud (Accuracy) y AUC."""
    # Realiza las predicciones con el modelo
    y_pred = model.predict(X)

    # Calcular Exactitud (Accuracy)
    acc = accuracy_score(y, y_pred)

    # Calcular AUC (Área bajo la curva ROC)
    auc = roc_auc_score(y, y_pred)

    return acc, auc


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


def plot_time_series(df_climate: pd.DataFrame, variables: List[str]) -> None:
    """Grafica series de tiempo para las variables seleccionadas con líneas de tendencia.

    Args:
        df_climate: DataFrame de datos climáticos completos (sin filtrar por especie).
        variables: Lista de variables climáticas a mostrar.

    Se agrupan los datos por ``YEAR`` y se calcula la media de cada
    variable climática. Cada variable se grafica en la misma figura para facilitar la
    comparación temporal, y se añade una línea de tendencia.
    """
    if not variables:
        st.info("Seleccione al menos una variable climática para visualizar la serie de tiempo.")
        return

    # Asegurarse de que 'YEAR' en df_climate esté presente como tipo entero
    if df_climate['YEAR'].dtype != 'int':
        try:
            df_climate['YEAR'] = df_climate['YEAR'].astype(int)
        except Exception:
            st.warning("No se pudo convertir la columna 'YEAR' a tipo entero.")
            return

    # Agrupar los datos climáticos por 'YEAR' y calcular la media de cada variable climática
    grouped_climate = df_climate.groupby('YEAR')[variables].mean().reset_index()

    # Crear el gráfico de las variables climáticas con sus líneas de tendencia
    fig = go.Figure()

    # Graficar las variables climáticas y sus líneas de tendencia
    for var in variables:
        # Obtener los datos de la variable climática
        x = grouped_climate['YEAR']
        y = grouped_climate[var]

        # Calcular la línea de tendencia utilizando regresión lineal
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception:
            slope, intercept = 0, y.mean()
        trendline = slope * x + intercept

        # Graficar la serie de tiempo de la variable climática
        fig.add_trace(go.Scatter(x=grouped_climate['YEAR'], y=grouped_climate[var],
                                 mode='lines', name=f"{var} - Serie de Tiempo"))

        # Graficar la línea de tendencia
        fig.add_trace(go.Scatter(x=grouped_climate['YEAR'], y=trendline,
                                 mode='lines', name=f"{var} - Tendencia", line=dict(dash='dash')))

    # Ajustar el diseño del gráfico
    fig.update_layout(title="",
                      xaxis_title="Año",
                      yaxis_title="Valor",
                      template="plotly_dark")

    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)


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


def plot_variable_importance(model, feature_names: List[str], key: str) -> None:
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
    st.plotly_chart(fig, use_container_width=True, key=key)


def main() -> None:
    st.set_page_config(page_title="Dashboard de avistamientos de aves", layout="wide")
    st.markdown(
        '<h1 class ="main-header"> Observa, Conoce y Protege: La Avifauna de ESPOL.</h1>',
        unsafe_allow_html=True
    )
    
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

    st.markdown("Estadísticas importantes que te gustaría saber.")

        # Sección: Aves más probables por mes de avistamiento
    st.markdown("### Aves más probables por mes")
    month_map = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
                 7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
    
    month_options = ['Todos'] + [month_map[m] for m in sorted(df['MONTH'].dropna().unique())]
    selected_month_name = st.selectbox("Seleccione un mes para ver las especies más probables",
                                       options=month_options)
    
    if selected_month_name != 'Todos':
        # Convertir nombre de mes a número
        month_name_to_number = {v: k for k, v in month_map.items()}
        selected_month = month_name_to_number.get(selected_month_name)
        df_month_filter = df[df['MONTH'] == selected_month]
    
        if not df_month_filter.empty:
            month_counts = df_month_filter.groupby('COMMON NAME')['AVISTAMIENTOS'].sum().reset_index()
            month_counts = month_counts.sort_values('AVISTAMIENTOS', ascending=False).head(10)
            fig_month = px.bar(
                month_counts, x='COMMON NAME', y='AVISTAMIENTOS',
                title=f"Aves más probables en {selected_month_name}",
                labels={'COMMON NAME':'Especie','AVISTAMIENTOS':'Avistamientos'}
            )
            fig_month.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.markdown(
                "<div style=\"background-color:#6a1b9a; color:white; padding:15px; border-radius:8px;\">"
                "No hay datos de avistamientos para el mes seleccionado."
                "</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<div style=\"background-color:#6a1b9a; color:white; padding:15px; border-radius:8px;\">"
            "Seleccione un mes para ver las especies más probables en ese periodo."
            "</div>",
            unsafe_allow_html=True
        )

    # Definimos las columnas
    col1, col2, col3, col4 = st.columns(4)

    # Definimos una altura fija para las cajas
    height = 200  # Ajusta la altura según sea necesario

    # Caja 1: Total de Aves
    with col1:
        aves_totales = df['ALL SPECIES REPORTED'].sum().astype(int)
        kpi_html = f"""
        <div style="background-color:#800080; padding: 20px; border-radius: 10px; color:white; text-align:center; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); height: {height}px; display: flex; flex-direction: column; 
                    justify-content: center; align-items: center;">
            <h4 style="margin:0; font-size: 22px;">🦜 Total de Aves</h4>
            <h3 style="margin:5px 0; font-size: 24px;">{aves_totales}</h3>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

    # Caja 2: Total de la especie seleccionada
    with col2:
        aves_totales_especie = df[df['COMMON NAME'] == selected_common_name]['ALL SPECIES REPORTED'].sum().astype(int)
        kpi_html = f"""
        <div style="background-color:#800080; padding: 20px; border-radius: 10px; color:white; text-align:center; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); height: {height}px; display: flex; flex-direction: column; 
                    justify-content: center; align-items: center;">
            <h4 style="margin:0; font-size:22px;">🐤 Total de {selected_common_name}</h4>
            <h3 style="margin:5px 0; font-size: 24px;">{aves_totales_especie}</h3>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

    # Caja 3: Categoría UICN
    with col3:
        categoria = df[df['COMMON NAME'] == selected_common_name]['CATEGORIA']
        kpi_html = f"""
        <div style="background-color:#800080; padding: 20px; border-radius: 10px; color:white; text-align:center; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); height: {height}px; display: flex; flex-direction: column; 
                    justify-content: center; align-items: center;">
            <h4 style="margin:0; font-size:22px;">Riesgo de Extinción</h4>
            <h3 style="margin:5px 0; font-size: 24px;">{categoria.iloc[0]}</h3>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

    # Caja 4: Distribución Geográfica
    with col4:
        endemica = df[df['COMMON NAME'] == selected_common_name]['ENDEMICO']
        endemica_texto = endemica.apply(lambda x: 'Especie endémica' if x else 'Especie no endémica')
        kpi_html = f"""
        <div style="background-color:#800080; padding: 20px; border-radius: 10px; color:white; text-align:center; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); height: {height}px; display: flex; flex-direction: column; 
                    justify-content: center; align-items: center;">
            <h4 style="margin:0; font-size:22px;">Distribución Geográfica</h4>
            <h3 style="margin:5px 0; font-size: 24px;">{endemica_texto.iloc[0]}</h3>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

    # Filtrar datos por especie
    species_df = filter_by_species(df, selected_common_name)

    # Calcular el mes con mayor número de avistamientos para la especie seleccionada
    month_map = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    mes_mas_frecuente = None
    month_counts_species = pd.Series(dtype=float)
    if not species_df.empty and 'MONTH' in species_df.columns:
        # Usar AVISTAMIENTOS si está disponible, de lo contrario usar ALL SPECIES REPORTED
        if 'AVISTAMIENTOS' in species_df.columns and not species_df['AVISTAMIENTOS'].isnull().all():
            month_counts_species = species_df.groupby('MONTH')['AVISTAMIENTOS'].sum()
        elif 'ALL SPECIES REPORTED' in species_df.columns:
            month_counts_species = species_df.groupby('MONTH')['ALL SPECIES REPORTED'].sum()
        if not month_counts_species.empty:
            top_month = month_counts_species.idxmax()
            try:
                mes_mas_frecuente = month_map.get(int(top_month), str(top_month))
            except Exception:
                mes_mas_frecuente = str(top_month)
                
        # Mostrar el mes con más avistamientos en un apartado destacado
    if mes_mas_frecuente:
        st.markdown("### 📅 Mes de mayor avistamiento")
        # Tarjeta morada para el mensaje de mayor avistamiento
        st.markdown(
            f"""
            <div style="background-color:#6a1b9a;
                        color:white;
                        padding:15px;
                        border-radius:8px;
                        margin:10px 0;">
                Según los registros, el mes con mayor número de avistamientos de <strong>{selected_common_name}</strong>
                es <strong>{mes_mas_frecuente}</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )
        # Mostrar un gráfico de barras de avistamientos por mes para la especie
        fig_month = px.bar(
            month_counts_species.reset_index(),
            x='MONTH',
            y=month_counts_species.name,
            title=f"Avistamientos mensuales de {selected_common_name}"
        )
        fig_month.update_layout(
            xaxis_title="Mes",
            yaxis_title="Total de avistamientos"
        )
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        # Tarjeta morada para indicar que no hay datos suficientes
        st.markdown(
            """
            <div style="background-color:#6a1b9a;
                        color:white;
                        padding:15px;
                        border-radius:8px;
                        margin:10px 0;">
                No hay datos suficientes para determinar un mes de mayor avistamiento para esta especie.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Sección: Variables climáticas que más afectan a la especie
    st.markdown("### Variables climáticas que más afectan a la especie")
    
    if not species_df.empty:
        climate_vars_list = list(climate_variable_names.keys())
        correlation_dict = {}
        for var in climate_vars_list:
            if var in species_df.columns and 'LOG_AVISTAMIENTOS' in species_df.columns:
                corr_value = species_df[var].corr(species_df['LOG_AVISTAMIENTOS'])
                if not pd.isna(corr_value):
                    correlation_dict[var] = corr_value
    
        if correlation_dict:
            corr_df = pd.DataFrame.from_dict(correlation_dict, orient='index', columns=['Correlation'])
            corr_df['AbsCorr'] = corr_df['Correlation'].abs()
            corr_df = corr_df.sort_values('AbsCorr', ascending=False).reset_index()
            corr_df['Variable'] = corr_df['index']
    
            fig_corr = px.bar(
                corr_df, x='Variable', y='AbsCorr',
                title=f"Impacto de variables climáticas en {selected_common_name}",
                labels={'AbsCorr':'|Correlación|'}
            )
            fig_corr.update_layout(xaxis_title="Variable", yaxis_title="|Correlación|")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.markdown(
                "<div style=\"background-color:#6a1b9a; color:white; padding:15px; border-radius:8px;\">"
                "No se pudieron calcular correlaciones para las variables climáticas de esta especie."
                "</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<div style=\"background-color:#6a1b9a; color:white; padding:15px; border-radius:8px;\">"
            "No se encontraron datos de esta especie para calcular correlaciones de variables climáticas."
            "</div>",
            unsafe_allow_html=True
        )

    
    # Variables climáticas candidatas (columna excepto identificadores y variables de respuesta)
    climate_variable_names = {
        'PRECTOTCORR': 'Precipitación total corregida',
        'PS': 'Presión en superficie',
        'RH2M': 'Humedad relativa',
        'T2M': 'Temperatura a 2 metros',
        'T2MWET': 'Temperatura media a 2 metros',
        'T2M_MAX': 'Temperatura máxima a 2 metros',
        'TS': 'Temperatura superficial',
        'WS10M': 'Velocidad del viento a 10 metros'
    }

    # Variables climáticas candidatas (siglas)
    climate_vars = list(climate_variable_names.keys())  # Las siglas de las variables climáticas

    # Widget para seleccionar variables para la serie de tiempo (mostrar nombres completos)
    selected_vars_time = st.sidebar.multiselect(
        "Variables climáticas para series de tiempo:",
        options=[climate_variable_names[var] for var in climate_vars],  # Mostrar nombres completos
        default=[]
    )

    # Convertir las selecciones del usuario de vuelta a las siglas
    selected_vars_time_siglas = [
        var for var in climate_vars if climate_variable_names[var] in selected_vars_time
    ]

    # Top N de avistamientos (fuera del filtro de especie)
    st.sidebar.markdown("---")
    max_n = min(10, species_mapping.shape[0])  # limite de especies a mostrar
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

        # Mostrar instrucciones de lectura del gráfico en una tarjeta morada personalizada
    st.markdown(
        """
        <div style="background-color:#6a1b9a;
                    color:white;
                    padding:15px;
                    border-radius:8px;
                    margin:10px 0;">
            <strong>Instrucciones para leer el gráfico:</strong>
            <ul style="margin-top:8px;">
                <li>El gráfico muestra la <strong>serie de tiempo</strong> de la variable climática seleccionada.</li>
                <li><strong>Líneas continuas</strong>: Muestran la evolución de la variable a lo largo del tiempo.</li>
                <li><strong>Líneas discontinuas</strong>: Representan la tendencia general (línea de regresión).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
    # Series de tiempo de variables climáticas
    st.markdown("### Series de tiempo de variables climáticas")
    plot_time_series(df, selected_vars_time_siglas)

    # Top N especies más avistadas
    st.markdown("### Top N especies por avistamientos (en todo el conjunto de datos)")
    plot_top_n_birds(df, n_top)

    st.markdown("## Comparación de modelos por especie")
    with st.expander("Ver comparación de modelos"):
        for img_file, desc in zip(MODEL_IMAGE_FILES, MODEL_IMAGE_DESCRIPTIONS):
            # Utilizar columnas para poner cada imagen con su tarjeta informativa
            col_img, col_info = st.columns([3, 2])
            with col_img:
                if Path(img_file).exists():
                    st.image(img_file, use_container_width=True)
                else:
                    st.warning(
                        f"No se encontró la imagen '{img_file}'. "
                        "Asegúrate de colocar el archivo en la misma carpeta del script o actualiza la ruta en MODEL_IMAGE_FILES."
                    )
            with col_info:
                # Tarjeta personalizada de color morado
                st.markdown(
                    f"""
                    <div style="background-color:#6a1b9a;
                                color:white;
                                padding:15px;
                                border-radius:8px;
                                margin:5px 0;">
                        {desc}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        
    st.markdown("---")
    st.caption("Aplicación desarrollada para visualizar avistamientos y variables climáticas de aves.")


if __name__ == '__main__':
    main()
