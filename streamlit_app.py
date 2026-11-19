from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Segmentación de clientes | Taller R",
    page_icon="📊",
    layout="wide",
)

CARPETA_DATOS = Path(__file__).resolve().parent / "datos_practica"
ARCHIVOS = {
    "Ventas mensuales": "01_ventas_mensuales.csv",
    "Clientes por ciudad y canal": "02_clientes_por_ciudad_canal.csv",
    "Clientes": "03_clientes.csv",
    "Entregas": "04_entregas.csv",
    "Productos": "05_productos.csv",
}
VARIABLES = ["compras", "gasto_medio", "edad"]
ETIQUETAS = {
    "compras": "Compras por cliente",
    "gasto_medio": "Gasto medio por compra",
    "edad": "Edad",
}


@st.cache_data
def cargar_bases():
    bases = {}
    faltantes = []
    for nombre, archivo in ARCHIVOS.items():
        ruta = CARPETA_DATOS / archivo
        if not ruta.exists():
            faltantes.append(str(ruta))
        else:
            bases[nombre] = pd.read_csv(ruta, encoding="utf-8-sig")
    if faltantes:
        raise FileNotFoundError("No se encontraron estos archivos:\n" + "\n".join(faltantes))
    return bases


try:
    bases = cargar_bases()
except Exception as error:
    st.error(f"No fue posible cargar los datos: {error}")
    st.stop()

clientes = bases["Clientes"].copy()
columnas_requeridas = {"cliente", "segmento", *VARIABLES}
if not columnas_requeridas.issubset(clientes.columns):
    st.error("03_clientes.csv debe tener estas columnas: " + ", ".join(sorted(columnas_requeridas)))
    st.stop()

clientes[VARIABLES] = clientes[VARIABLES].apply(pd.to_numeric, errors="coerce")
clientes = clientes.dropna(subset=VARIABLES).reset_index(drop=True)
if len(clientes) < 3:
    st.error("Se requieren al menos tres clientes con datos completos para ejecutar K-means.")
    st.stop()

st.title("Segmentación de clientes")
st.caption("Taller de visualización · Datos ficticios para fines académicos")
st.write(
    "Explora los perfiles de clientes y compara la segmentación original con los grupos "
    "calculados mediante K-means. Ajusta los controles de la barra lateral."
)

# Controles en la barra lateral, siguiendo la maqueta de Streamlit.
st.sidebar.header("Controles")
segmentos = ["Todos", *sorted(clientes["segmento"].astype(str).unique())]
segmento_seleccionado = st.sidebar.selectbox("Segmento de referencia", segmentos)

k_max = min(8, len(clientes) - 1)
k = st.sidebar.slider("Número de grupos (k)", min_value=2, max_value=k_max,
                      value=min(3, k_max), step=1)
eje_x = st.sidebar.selectbox("Variable del eje horizontal", VARIABLES, index=0,
                             format_func=lambda x: ETIQUETAS[x])
eje_y = st.sidebar.selectbox("Variable del eje vertical", VARIABLES, index=1,
                             format_func=lambda x: ETIQUETAS[x])
estandarizar = st.sidebar.checkbox("Estandarizar variables", value=True)
semilla = st.sidebar.number_input("Semilla aleatoria", min_value=1, max_value=999999,
                                  value=42, step=1)
gasto_minimo = st.sidebar.slider(
    "Gasto medio mínimo", min_value=float(clientes["gasto_medio"].min()),
    max_value=float(clientes["gasto_medio"].max()),
    value=float(clientes["gasto_medio"].min()), step=1.0,
)

# K-means se calcula con todos los clientes completos; los filtros solo cambian la vista.
x = clientes[VARIABLES].to_numpy(dtype=float)
x_modelo = StandardScaler().fit_transform(x) if estandarizar else x
modelo = KMeans(n_clusters=int(k), n_init=20, random_state=int(semilla))
clientes["grupo_kmeans"] = modelo.fit_predict(x_modelo).astype(str)

vista = clientes.copy()
if segmento_seleccionado != "Todos":
    vista = vista.loc[vista["segmento"].astype(str) == segmento_seleccionado]
vista = vista.loc[vista["gasto_medio"] >= gasto_minimo]
st.write(f"**Clientes visibles:** {len(vista)} de {len(clientes)}")

# Preparar la curva del codo con los mismos datos y la misma opción de escala.
ks = list(range(1, k_max + 1))
inercia = []
for numero_grupos in ks:
    ajuste = KMeans(n_clusters=numero_grupos, n_init=10, random_state=int(semilla))
    ajuste.fit(x_modelo)
    inercia.append(ajuste.inertia_)
tabla_codo = pd.DataFrame({"Número de grupos (k)": ks, "Suma intra-grupo": inercia})

# Pestañas principales de la página.
tab_resumen, tab_segmentos, tab_codo, tab_bases = st.tabs(
    ["Resumen", "Segmentación", "Método del codo", "Bases de práctica"]
)

with tab_resumen:
    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes analizados", f"{len(clientes):,}")
    c2.metric("Compras promedio", f"{clientes['compras'].mean():.1f}")
    c3.metric("Gasto medio promedio", f"{clientes['gasto_medio'].mean():.2f}")
    st.subheader("Tamaño de los grupos calculados")
    resumen = (
        clientes.groupby("grupo_kmeans", as_index=False)
        .agg(
            Clientes=("cliente", "count"),
            Compras_promedio=("compras", "mean"),
            Gasto_medio_promedio=("gasto_medio", "mean"),
            Edad_promedio=("edad", "mean"),
        )
        .sort_values("grupo_kmeans")
    )
    st.dataframe(resumen.round(2), hide_index=True, use_container_width=True)
    st.info("Los identificadores de grupo son etiquetas arbitrarias; un número menor no indica un grupo mejor.")

with tab_segmentos:
    st.subheader("Grupos de clientes según las variables seleccionadas")
    if vista.empty:
        st.warning("Ningún cliente cumple los filtros. Reduce el gasto mínimo o cambia el segmento.")
    else:
        st.scatter_chart(
            vista,
            x=eje_x,
            y=eje_y,
            color="grupo_kmeans",
            x_label=ETIQUETAS[eje_x],
            y_label=ETIQUETAS[eje_y],
            height=480,
        )
        st.caption(
            f"K-means utiliza compras, gasto medio y edad. "
            f"El gráfico muestra {ETIQUETAS[eje_x].lower()} frente a {ETIQUETAS[eje_y].lower()}."
        )
    st.subheader("Registros filtrados")
    st.dataframe(vista, hide_index=True, use_container_width=True)

with tab_codo:
    st.subheader("Método del codo")
    st.line_chart(tabla_codo, x="Número de grupos (k)", y="Suma intra-grupo")
    st.write(f"Valor de k seleccionado: **{k}**.")
    st.caption("El codo orienta la selección de k, pero debe interpretarse junto con el propósito y los perfiles de los grupos.")

with tab_bases:
    nombre_base = st.selectbox("Selecciona una base", list(bases.keys()))
    base = bases[nombre_base]
    st.write(f"Registros: **{len(base):,}** · Variables: **{len(base.columns)}**")
    st.dataframe(base, hide_index=True, use_container_width=True)
    calidad = pd.DataFrame({
        "Indicador": ["Filas", "Columnas", "Celdas faltantes", "Filas duplicadas"],
        "Resultado": [len(base), len(base.columns), int(base.isna().sum().sum()), int(base.duplicated().sum())],
    })
    st.subheader("Revisión rápida")
    st.dataframe(calidad, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Alcance: datos simulados y análisis descriptivo. La asociación observada no demuestra causalidad; "
    "los grupos dependen de k, de la escala y de las variables incluidas."
)
