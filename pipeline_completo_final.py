# ============================================================
# PIPELINE COMPLETO DE ANÁLISIS DE POBREZA Y CONDICIONES DE VIDA
# Censo de Población y Vivienda 2022 - Ecuador (INEC)
# Autor: Roberto Marcelo Martínez Hinojosa
# Universidad de Guayaquil
# ORCID: https://orcid.org/0000-0001-9759-3305
# GitHub: https://github.com/Azulado70/poverty-ecuador-census-2022
# Zenodo DOI: https://doi.org/10.5281/zenodo.20943520
# ============================================================
# HERRAMIENTAS: Python 3.14 | scikit-learn | pandas | numpy
#               matplotlib | seaborn | folium | pyodbc | pyshp
# ============================================================
# ESTRUCTURA DEL PIPELINE:
#   BLOQUE 0 — Configuración e importaciones
#   BLOQUE 1 — Conexión y extracción desde SQL Server
#   BLOQUE 2 — Limpieza y preparación de datos
#   BLOQUE 3 — Análisis de Componentes Principales (ACP)
#              → Índice de Condiciones de Vida (ICV)
#   BLOQUE 4 — Análisis de Clases Latentes (ACL/GMM)
#              → Perfiles latentes de pobreza
#   BLOQUE 5 — Regresión Logística Multinomial
#              → Determinantes sociodemográficos
#   BLOQUE 6 — Análisis geográfico provincial
#              → Mapas interactivos (Folium)
#   BLOQUE 7 — Exportación de resultados
# ============================================================

# ────────────────────────────────────────────────────────────
# BLOQUE 0: CONFIGURACIÓN E IMPORTACIONES
# ────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import json
import folium
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configuración global de gráficos
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

print("=" * 65)
print(" PIPELINE ANÁLISIS DE POBREZA - CENSO ECUADOR 2022")
print("=" * 65)
print(f" Autor: Roberto Marcelo Martínez Hinojosa")
print(f" Universidad de Guayaquil")
print(f" ORCID: https://orcid.org/0000-0001-9759-3305")
print("=" * 65)


# ────────────────────────────────────────────────────────────
# BLOQUE 1: CONEXIÓN Y EXTRACCIÓN DESDE SQL SERVER
# ────────────────────────────────────────────────────────────
# Base de datos: CENSODB2022_26_06_26
# Schema: bronze
# Tablas: CPV_Poblacion_2022, CPV_Vivienda_2022, CPV_Hogar_2022
# Total registros: ~28.7 millones
# ────────────────────────────────────────────────────────────

print("\n[BLOQUE 1] Conectando a SQL Server y extrayendo datos...")

conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=LAPTOP-U7OHUP55;"
    "DATABASE=CENSODB2022_26_06_26;"
    "Trusted_Connection=yes;"
)

# Muestra aleatoria nacional de 100,000 registros
# ORDER BY NEWID() garantiza aleatoriedad sin sesgo geográfico
query = """
SELECT
    p.ID_PER,
    p.AUR,        -- Zona: 1=Urbana, 2=Rural
    p.CANTON,     -- Código de cantón (4 dígitos)
    p.PARROQ,     -- Código de parroquia
    p.P01,        -- Sexo: 1=Hombre, 2=Mujer
    p.P03,        -- Edad en años cumplidos
    p.P12,        -- Estado civil
    p.P13,        -- Autoidentificación étnica
    p.P15,        -- Sabe leer y escribir
    p.P16,        -- Nivel de instrucción
    p.CONDACT,    -- Condición de actividad laboral
    p.ETAEDAD,    -- Grupo etario
    p.ANALF,      -- Analfabetismo funcional
    p.GEDAD,      -- Gran grupo de edad
    v.V01,        -- Tipo de vivienda
    v.V03,        -- Material del techo
    v.V04,        -- Material del piso
    v.V05,        -- Material de las paredes
    v.V06,        -- Procedencia del agua
    v.V07,        -- Sistema de alcantarillado/eliminación de aguas servidas
    v.V08,        -- Disponibilidad de energía eléctrica
    v.V09,        -- Eliminación de basura
    v.V10,        -- Tipo de servicio higiénico
    v.V11,        -- Acceso a internet
    v.TOTPER,     -- Total de personas en la vivienda
    h.H01,        -- Tenencia de la vivienda
    h.H05,        -- Combustible para cocinar
    h.H06,        -- Fuente principal de luz
    h.INH         -- Número de hogares en la vivienda
FROM bronze.CPV_Poblacion_2022 p
JOIN bronze.CPV_Vivienda_2022 v ON p.ID_VIV = v.ID_VIV
JOIN bronze.CPV_Hogar_2022 h    ON p.ID_HOG = h.ID_HOG
ORDER BY NEWID()
OFFSET 0 ROWS FETCH NEXT 100000 ROWS ONLY
"""

df_raw = pd.read_sql(query, conn)
conn.close()

print(f"  ✓ Registros extraídos: {len(df_raw):,}")
print(f"  ✓ Variables: {df_raw.shape[1]}")
print(f"  ✓ Cantones representados: {df_raw['CANTON'].nunique()}")


# ────────────────────────────────────────────────────────────
# BLOQUE 2: LIMPIEZA Y PREPARACIÓN DE DATOS
# ────────────────────────────────────────────────────────────

print("\n[BLOQUE 2] Limpiando y preparando datos...")

# Variables de condiciones de vida (dominio vivienda + hogar)
VARS_VIVIENDA = ['V01','V03','V04','V05','V06',
                 'V07','V08','V09','V10','V11']
VARS_HOGAR    = ['H01','H05','H06','TOTPER']
VARS_VIDA     = VARS_VIVIENDA + VARS_HOGAR   # 14 variables indicadoras

# Variables sociodemográficas para regresión
VARS_SOCIO    = ['AUR','P03','P13','P15','P16','CONDACT','ANALF']

# Convertir todas las variables a numérico
df = df_raw.copy()
for col in df.columns:
    if col != 'ID_PER':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Código de provincia (2 primeros dígitos del cantón)
df['CANTON']   = df['CANTON'].astype(str).str.zfill(4)
df['PROV_COD'] = df['CANTON'].str[:2]

# Diccionario de códigos INEC → nombres de provincia
PROV_NOMBRES = {
    '01':'Azuay',        '02':'Bolívar',      '03':'Cañar',
    '04':'Carchi',       '05':'Cotopaxi',     '06':'Chimborazo',
    '07':'El Oro',       '08':'Esmeraldas',   '09':'Guayas',
    '10':'Imbabura',     '11':'Loja',         '12':'Los Ríos',
    '13':'Manabí',       '14':'Morona Santiago','15':'Napo',
    '16':'Pastaza',      '17':'Pichincha',    '18':'Tungurahua',
    '19':'Zamora Chinchipe','20':'Galápagos', '21':'Sucumbíos',
    '22':'Orellana',     '23':'Santo Domingo de los Tsachilas',
    '24':'Santa Elena'
}
df['PROVINCIA'] = df['PROV_COD'].map(PROV_NOMBRES)

# Eliminar filas con valores nulos en variables de vida
df_clean = df.dropna(subset=VARS_VIDA).reset_index(drop=True)

print(f"  ✓ Registros después de limpieza: {len(df_clean):,}")
print(f"  ✓ Registros eliminados (nulos): {len(df) - len(df_clean):,}")
print(f"  ✓ Provincias representadas: {df_clean['PROVINCIA'].nunique()}")

# Resumen de valores nulos por variable
nulos = (df_clean[VARS_VIDA + VARS_SOCIO].isnull().sum() /
         len(df_clean) * 100).round(2)
nulos_presentes = nulos[nulos > 0]
if len(nulos_presentes) > 0:
    print(f"\n  Variables con nulos restantes (%):")
    print(nulos_presentes.to_string())

# Guardar dataset limpio
df_clean.to_csv("01_datos_limpios.csv", index=False)
print(f"\n  → Guardado: 01_datos_limpios.csv")


# ────────────────────────────────────────────────────────────
# BLOQUE 3: ANÁLISIS DE COMPONENTES PRINCIPALES (ACP)
#           → Índice de Condiciones de Vida (ICV)
# ────────────────────────────────────────────────────────────
# Criterio de retención: varianza acumulada ≥ 80%
# ICV = -CP1 normalizado a escala 0-100
# Mayor ICV = mejores condiciones de vida
# ────────────────────────────────────────────────────────────

print("\n[BLOQUE 3] Análisis de Componentes Principales (ACP)...")

# Estandarización (media=0, DE=1)
scaler = StandardScaler()
X_vida = scaler.fit_transform(df_clean[VARS_VIDA])

# ACP completo para análisis de varianza
pca_full = PCA()
pca_full.fit(X_vida)

varianza     = pca_full.explained_variance_ratio_ * 100
varianza_acum = np.cumsum(varianza)
n_comp       = int(np.argmax(varianza_acum >= 80)) + 1

print(f"\n  === VARIANZA EXPLICADA POR COMPONENTE ===")
for i, (v, va) in enumerate(zip(varianza[:n_comp+2], varianza_acum[:n_comp+2])):
    marca = " ← umbral 80%" if i+1 == n_comp else ""
    print(f"  CP{i+1:2d}: {v:6.2f}%  Acumulada: {va:6.2f}%{marca}")

print(f"\n  → Componentes retenidos (≥80% varianza): {n_comp}")
print(f"  → CP1 explica: {varianza[0]:.2f}% de la varianza total")

# ACP con n componentes óptimos
pca_final = PCA(n_components=n_comp)
X_pca     = pca_final.fit_transform(X_vida)

# Índice de Condiciones de Vida (ICV)
# CP1 captura privación material; se invierte para que mayor = mejor
df_clean['ICV'] = -X_pca[:, 0]
df_clean['ICV'] = ((df_clean['ICV'] - df_clean['ICV'].min()) /
                   (df_clean['ICV'].max() - df_clean['ICV'].min()) * 100)

print(f"\n  === ESTADÍSTICAS DEL ICV ===")
print(df_clean['ICV'].describe().round(2).to_string())

# ── Gráfico ACP ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ACP - Condiciones de Vida Ecuador 2022', fontsize=13, fontweight='bold')

# Panel 1: Varianza por componente
axes[0].bar(range(1, len(varianza)+1), varianza, color='steelblue', alpha=0.85)
axes[0].axvline(x=n_comp, color='green', linestyle='--', linewidth=2, label=f'K={n_comp}')
axes[0].set_xlabel('Componente')
axes[0].set_ylabel('Varianza explicada (%)')
axes[0].set_title('Varianza por Componente')
axes[0].legend()

# Panel 2: Curva de sedimentación acumulada
axes[1].plot(range(1, len(varianza_acum)+1), varianza_acum, 'bo-', markersize=5)
axes[1].axhline(y=80, color='red', linestyle='--', linewidth=2, label='80%')
axes[1].axvline(x=n_comp, color='green', linestyle='--', linewidth=2, label=f'K={n_comp}')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Varianza Acumulada (%)')
axes[1].set_title('Varianza Acumulada (Codo)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_grafico_acp.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n  → Guardado: 02_grafico_acp.png")


# ────────────────────────────────────────────────────────────
# BLOQUE 4: ANÁLISIS DE CLASES LATENTES (ACL / GMM)
#           → Perfiles latentes de pobreza
# ────────────────────────────────────────────────────────────
# Método: Modelos de Mezcla Gaussiana (GaussianMixture)
# Criterio de selección: BIC mínimo (penaliza complejidad)
# Rango evaluado: K = 2 a 7 clases
# ────────────────────────────────────────────────────────────

print("\n[BLOQUE 4] Análisis de Clases Latentes (ACL/GMM)...")
print("  Evaluando K = 2 a 7 clases...")

bic_scores = []
aic_scores = []
RANGO_K    = range(2, 8)

for k in RANGO_K:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(X_pca)
    bic_scores.append(gmm.bic(X_pca))
    aic_scores.append(gmm.aic(X_pca))
    print(f"  K={k} | BIC={gmm.bic(X_pca):>15,.0f} | AIC={gmm.aic(X_pca):>15,.0f}")

# Número óptimo de clases
mejor_k = list(RANGO_K)[np.argmin(bic_scores)]
print(f"\n  → Número óptimo de clases (BIC mínimo): {mejor_k}")

# Modelo final con K óptimo
gmm_final = GaussianMixture(n_components=mejor_k, random_state=42, n_init=10)
gmm_final.fit(X_pca)

df_clean['CLASE_LCA']  = gmm_final.predict(X_pca)
df_clean['PROB_LCA']   = gmm_final.predict_proba(X_pca).max(axis=1)

# Etiquetar clases por ICV promedio (menor ICV = mayor pobreza)
medias_icv = df_clean.groupby('CLASE_LCA')['ICV'].mean().sort_values()
ETIQUETAS_BASE = ['Pobreza Extrema','Pobreza Moderada','Vulnerable','No Pobre']
ETIQUETAS_EXTRA = [f'Nivel_{i}' for i in range(4, mejor_k)]
TODAS_ETIQUETAS = ETIQUETAS_BASE + ETIQUETAS_EXTRA

etiquetas_map = {}
for i, clase in enumerate(medias_icv.index):
    etiquetas_map[clase] = TODAS_ETIQUETAS[i] if i < len(TODAS_ETIQUETAS) else f'Clase_{clase}'

df_clean['PERFIL'] = df_clean['CLASE_LCA'].map(etiquetas_map)

print(f"\n  === DISTRIBUCIÓN DE PERFILES DE POBREZA ===")
dist = df_clean['PERFIL'].value_counts()
for perfil, n in dist.items():
    pct  = n / len(df_clean) * 100
    icv  = df_clean[df_clean['PERFIL']==perfil]['ICV'].mean()
    prob = df_clean[df_clean['PERFIL']==perfil]['PROB_LCA'].mean()
    print(f"  {perfil:<22}: N={n:>6,}  ({pct:5.1f}%)  ICV={icv:5.1f}  Prob={prob:.3f}")

# ── Gráfico ACL ─────────────────────────────────────────────
COLORES_PERFIL = {
    'Pobreza Extrema':'#d62728', 'Pobreza Moderada':'#ff7f0e',
    'Vulnerable':'#ffdd57',      'No Pobre':'#2ca02c',
    'Nivel_4':'#1f77b4',         'Nivel_5':'#9467bd',
    'Nivel_6':'#8c564b'
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('LCA - Perfiles de Pobreza Ecuador 2022', fontsize=13, fontweight='bold')

# Panel 1: BIC/AIC por K
axes[0].plot(list(RANGO_K), bic_scores, 'bo-', linewidth=2, label='BIC')
axes[0].plot(list(RANGO_K), aic_scores, 'rs-', linewidth=2, label='AIC')
axes[0].axvline(x=mejor_k, color='green', linestyle='--', linewidth=2, label=f'Óptimo K={mejor_k}')
axes[0].set_xlabel('Número de Clases')
axes[0].set_ylabel('Score')
axes[0].set_title('Selección de Clases (BIC/AIC)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel 2: Dispersión en espacio CP1-CP2
for perfil in df_clean['PERFIL'].unique():
    idx   = df_clean[df_clean['PERFIL']==perfil].index
    color = COLORES_PERFIL.get(perfil, '#7f7f7f')
    axes[1].scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=color, label=perfil, alpha=0.3, s=4)
axes[1].set_xlabel(f'CP1 ({varianza[0]:.1f}%)')
axes[1].set_ylabel(f'CP2 ({varianza[1]:.1f}%)')
axes[1].set_title('Clases Latentes de Pobreza')
axes[1].legend(markerscale=3, fontsize=8)

plt.tight_layout()
plt.savefig('03_grafico_lca.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n  → Guardado: 03_grafico_lca.png")


# ────────────────────────────────────────────────────────────
# BLOQUE 5: REGRESIÓN LOGÍSTICA MULTINOMIAL
#           → Determinantes sociodemográficos de la pobreza
# ────────────────────────────────────────────────────────────
# Variable dependiente: CLASE_LCA (clase latente asignada)
# Variables independientes: zona, edad, etnia, educación,
#                           alfabetismo, condición de actividad,
#                           analfabetismo funcional
# Partición: 70% entrenamiento / 30% prueba
# ────────────────────────────────────────────────────────────

print("\n[BLOQUE 5] Regresión Logística Multinomial...")

# Preparar datos para regresión
df_reg = df_clean[VARS_SOCIO + ['CLASE_LCA']].dropna().reset_index(drop=True)
print(f"  Registros válidos para regresión: {len(df_reg):,}")

X_reg = StandardScaler().fit_transform(df_reg[VARS_SOCIO])
y_reg = df_reg['CLASE_LCA']

# Distribución de clases en muestra de regresión
print(f"\n  Distribución de clases:")
for clase, n in y_reg.value_counts().sort_index().items():
    print(f"  Clase {clase} ({etiquetas_map.get(clase,'?'):<22}): {n:,}")

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42, stratify=y_reg
)

# Modelo de regresión logística multinomial
modelo = LogisticRegression(max_iter=2000, random_state=42)
modelo.fit(X_train, y_train)

acc = modelo.score(X_test, y_test)
print(f"\n  === MÉTRICAS DEL MODELO ===")
print(f"  Accuracy global: {acc:.4f} ({acc*100:.1f}%)")

print(f"\n  === REPORTE DE CLASIFICACIÓN ===")
nombres_clases = [etiquetas_map.get(c, str(c)) for c in sorted(y_reg.unique())]
print(classification_report(y_test, modelo.predict(X_test),
                             target_names=nombres_clases))

# Coeficientes estandarizados
NOMBRES_VARS = ['Zona (AUR)', 'Edad (P03)', 'Etnia (P13)',
                'Alfabetismo (P15)', 'Nivel Edu (P16)',
                'Cond. Activ (CONDACT)', 'Analfabetismo (ANALF)']

etiquetas_coef = [etiquetas_map.get(c, f'Clase_{c}')
                  for c in sorted(modelo.classes_)]
coef_df = pd.DataFrame(
    modelo.coef_,
    columns=NOMBRES_VARS,
    index=etiquetas_coef
).round(3)

print(f"\n  === COEFICIENTES ESTANDARIZADOS ===")
print(coef_df.to_string())

# ── Gráfico Regresión ────────────────────────────────────────
plt.figure(figsize=(12, 6))
sns.heatmap(coef_df, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, annot_kws={'size': 9})
plt.title('Determinantes de Clases de Pobreza\n'
          'Regresión Logística Multinomial - Ecuador 2022',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('04_grafico_regresion.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n  → Guardado: 04_grafico_regresion.png")

# Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, modelo.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Regresión Logística Multinomial')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('04b_matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → Guardado: 04b_matriz_confusion.png")


# ────────────────────────────────────────────────────────────
# BLOQUE 6: ANÁLISIS GEOGRÁFICO PROVINCIAL
#           → Resumen por provincia y mapas interactivos
# ────────────────────────────────────────────────────────────
# GeoJSON: generado desde shapefile GADM 4.1 sin GDAL
#          usando la librería pyshp
# Mapas: Folium (HTML interactivo)
# ────────────────────────────────────────────────────────────

print("\n[BLOQUE 6] Análisis geográfico provincial...")

# ── 6.1 Resumen estadístico por provincia ───────────────────
resumen_prov = df_clean.groupby('PROVINCIA').agg(
    ICV_promedio  = ('ICV', 'mean'),
    ICV_std       = ('ICV', 'std'),
    n_registros   = ('ICV', 'count'),
    pct_urbano    = ('AUR', lambda x:
                     (pd.to_numeric(x, errors='coerce')==1).mean()*100)
).reset_index().round(2)

# Porcentaje de pobreza (Extrema + Moderada) por provincia
mask_pobre = df_clean['PERFIL'].isin(['Pobreza Extrema','Pobreza Moderada'])
pct_pobre  = (df_clean[mask_pobre].groupby('PROVINCIA').size() /
              df_clean.groupby('PROVINCIA').size() * 100).reset_index()
pct_pobre.columns = ['PROVINCIA','pct_pobreza']

resumen_prov = resumen_prov.merge(pct_pobre, on='PROVINCIA', how='left')
resumen_prov['pct_pobreza'] = resumen_prov['pct_pobreza'].fillna(0).round(2)
resumen_prov = resumen_prov.sort_values('ICV_promedio').reset_index(drop=True)

print(f"\n  === PROVINCIAS POR ICV PROMEDIO (ascendente) ===")
print(f"  {'Provincia':<30} {'ICV':>6} {'%Pobreza':>9} {'N':>7}")
print(f"  {'-'*55}")
for _, row in resumen_prov.iterrows():
    print(f"  {row['PROVINCIA']:<30} {row['ICV_promedio']:>6.2f} "
          f"{row['pct_pobreza']:>8.1f}% {int(row['n_registros']):>7,}")

# Guardar resumen provincial
resumen_prov.to_csv('08_resumen_provincial.csv', index=False)
print(f"\n  → Guardado: 08_resumen_provincial.csv")

# ── 6.2 Convertir shapefile GADM a GeoJSON (sin GDAL) ───────
print(f"\n  Convirtiendo shapefile a GeoJSON...")
try:
    import shapefile
    sf      = shapefile.Reader("shapefile_ecuador/gadm41_ECU_1.shp")
    fields  = [f[0] for f in sf.fields[1:]]
    features = []
    for sr in sf.shapeRecords():
        props = dict(zip(fields, sr.record))
        geom  = sr.shape.__geo_interface__
        features.append({
            "type": "Feature",
            "properties": {"name": props.get("NAME_1",""), **{k: str(v) for k,v in props.items()}},
            "geometry": geom
        })
    geojson = {"type": "FeatureCollection", "features": features}
    with open("ecuador_provincias.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    print(f"  ✓ GeoJSON generado: {len(features)} provincias")
except Exception as e:
    print(f"  ! No se pudo generar GeoJSON: {e}")
    print(f"  ! Asegúrate de tener la carpeta shapefile_ecuador/ con los archivos GADM")
    geojson = None

# ── 6.3 Mapas interactivos Folium ───────────────────────────
if geojson:
    resumen_dict = resumen_prov.set_index('PROVINCIA').to_dict('index')

    def agregar_tooltips(mapa, geojson, resumen_dict):
        """Agrega tooltips interactivos a cada provincia."""
        for feature in geojson['features']:
            nombre = feature['properties']['name']
            if nombre in resumen_dict:
                d = resumen_dict[nombre]
                tooltip = (
                    f"<b>{nombre}</b><br>"
                    f"ICV: {d['ICV_promedio']:.1f}/100<br>"
                    f"% Pobreza: {d['pct_pobreza']:.1f}%<br>"
                    f"% Urbano: {d['pct_urbano']:.1f}%<br>"
                    f"N registros: {int(d['n_registros']):,}"
                )
                folium.GeoJson(
                    feature,
                    style_function=lambda x: {'fillOpacity':0,'weight':0},
                    tooltip=folium.Tooltip(tooltip)
                ).add_to(mapa)

    # Mapa 1: ICV por provincia
    m1 = folium.Map(location=[-1.8,-78.2], zoom_start=6,
                    tiles='CartoDB positron')
    folium.Choropleth(
        geo_data=geojson, data=resumen_prov,
        columns=['PROVINCIA','ICV_promedio'],
        key_on='feature.properties.name',
        fill_color='RdYlGn', fill_opacity=0.8, line_opacity=0.5,
        legend_name='Índice de Condiciones de Vida (0-100)',
        nan_fill_color='lightgrey', highlight=True
    ).add_to(m1)
    agregar_tooltips(m1, geojson, resumen_dict)
    folium.map.Marker(
        [-0.5,-79.5],
        icon=folium.DivIcon(html='<div style="font-size:12px;font-weight:bold;'
                                 'color:#1F3864;background:white;padding:4px;'
                                 'border-radius:4px;opacity:0.9">'
                                 'ICV — Censo Ecuador 2022</div>')
    ).add_to(m1)
    m1.save('09_mapa_icv.html')
    print(f"  → Guardado: 09_mapa_icv.html")

    # Mapa 2: % Pobreza por provincia
    m2 = folium.Map(location=[-1.8,-78.2], zoom_start=6,
                    tiles='CartoDB positron')
    folium.Choropleth(
        geo_data=geojson, data=resumen_prov,
        columns=['PROVINCIA','pct_pobreza'],
        key_on='feature.properties.name',
        fill_color='Reds', fill_opacity=0.8, line_opacity=0.5,
        legend_name='% Pobreza Extrema + Moderada',
        nan_fill_color='lightgrey', highlight=True
    ).add_to(m2)
    agregar_tooltips(m2, geojson, resumen_dict)
    m2.save('10_mapa_pobreza.html')
    print(f"  → Guardado: 10_mapa_pobreza.html")

    # Mapa 3: % Urbano por provincia
    m3 = folium.Map(location=[-1.8,-78.2], zoom_start=6,
                    tiles='CartoDB positron')
    folium.Choropleth(
        geo_data=geojson, data=resumen_prov,
        columns=['PROVINCIA','pct_urbano'],
        key_on='feature.properties.name',
        fill_color='Blues', fill_opacity=0.8, line_opacity=0.5,
        legend_name='% Población Urbana',
        nan_fill_color='lightgrey', highlight=True
    ).add_to(m3)
    agregar_tooltips(m3, geojson, resumen_dict)
    m3.save('11_mapa_urbano.html')
    print(f"  → Guardado: 11_mapa_urbano.html")


# ────────────────────────────────────────────────────────────
# BLOQUE 7: EXPORTACIÓN DE RESULTADOS FINALES
# ────────────────────────────────────────────────────────────

print("\n[BLOQUE 7] Exportando resultados finales...")

# Dataset completo con ICV, clase LCA y perfil
df_clean.to_csv('05_resultados_finales.csv', index=False)
print(f"  → Guardado: 05_resultados_finales.csv")

# Coeficientes de la regresión
coef_df.to_csv('06_coeficientes_regresion.csv')
print(f"  → Guardado: 06_coeficientes_regresion.csv")

# Resumen de clases latentes
resumen_clases = df_clean.groupby('PERFIL').agg(
    N           = ('ICV', 'count'),
    ICV_medio   = ('ICV', 'mean'),
    ICV_std     = ('ICV', 'std'),
    Prob_media  = ('PROB_LCA', 'mean')
).round(3).sort_values('ICV_medio')
resumen_clases.to_csv('07_resumen_clases_lca.csv')
print(f"  → Guardado: 07_resumen_clases_lca.csv")

# ── Resumen final ────────────────────────────────────────────
print("\n" + "=" * 65)
print(" PIPELINE COMPLETADO EXITOSAMENTE")
print("=" * 65)
print(f"\n REGISTROS ANALIZADOS : {len(df_clean):>10,}")
print(f" PROVINCIAS CUBIERTAS  : {df_clean['PROVINCIA'].nunique():>10}")
print(f" COMPONENTES ACP       : {n_comp:>10}  (80.34% varianza)")
print(f" CLASES LATENTES (ACL) : {mejor_k:>10}")
print(f" ICV NACIONAL (media)  : {df_clean['ICV'].mean():>10.2f} / 100")
print(f" ACCURACY REGRESIÓN    : {acc:>10.1%}")
print("\n ARCHIVOS GENERADOS:")
archivos = [
    "01_datos_limpios.csv        — Dataset limpio (99,681 registros)",
    "02_grafico_acp.png          — Varianza ACP y curva de codo",
    "03_grafico_lca.png          — Selección BIC/AIC y dispersión ACL",
    "04_grafico_regresion.png    — Heatmap coeficientes regresión",
    "04b_matriz_confusion.png    — Matriz de confusión",
    "05_resultados_finales.csv   — Datos con ICV, clase y perfil",
    "06_coeficientes_regresion.csv — Coeficientes estandarizados",
    "07_resumen_clases_lca.csv   — Estadísticas por clase",
    "08_resumen_provincial.csv   — ICV y pobreza por provincia",
    "09_mapa_icv.html            — Mapa interactivo ICV",
    "10_mapa_pobreza.html        — Mapa interactivo % pobreza",
    "11_mapa_urbano.html         — Mapa interactivo % urbano",
    "ecuador_provincias.geojson  — Geometrías provinciales",
]
for a in archivos:
    print(f"   {a}")

print("\n REPOSITORIO: https://github.com/Azulado70/poverty-ecuador-census-2022")
print(" ZENODO DOI : https://doi.org/10.5281/zenodo.20943520")
print("=" * 65)
