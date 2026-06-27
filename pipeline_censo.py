# ============================================================
# PIPELINE COMPLETO - POBREZA Y CONDICIONES DE VIDA ECUADOR
# Censo de Población y Vivienda 2022 - INEC
# Universidad de Guayaquil
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PIPELINE: POBREZA Y CONDICIONES DE VIDA - ECUADOR 2022")
print("=" * 60)

# ============================================================
# PASO 1: EXTRACCIÓN DE DATOS DESDE SQL SERVER
# ============================================================
print("\n[1/5] Extrayendo datos desde SQL Server...")

conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=LAPTOP-U7OHUP55;"
    "DATABASE=CENSODB2022_26_06_26;"
    "Trusted_Connection=yes;"
)

query = """
SELECT
    p.ID_PER, p.AUR, p.CANTON, p.PARROQ,
    p.P01, p.P03, p.P12, p.P13,
    p.P15, p.P16, p.CONDACT, p.ETAEDAD,
    p.ANALF, p.GEDAD,
    v.V01, v.V03, v.V04, v.V05, v.V06,
    v.V07, v.V08, v.V09, v.V10, v.V11,
    v.TOTPER,
    h.H01, h.H05, h.H06, h.INH
FROM bronze.CPV_Poblacion_2022 p
JOIN bronze.CPV_Vivienda_2022 v ON p.ID_VIV = v.ID_VIV
JOIN bronze.CPV_Hogar_2022 h    ON p.ID_HOG = h.ID_HOG
ORDER BY NEWID()
OFFSET 0 ROWS FETCH NEXT 100000 ROWS ONLY
"""

df = pd.read_sql(query, conn)
conn.close()

# Convertir todo a numérico
for col in df.columns:
    if col != 'ID_PER':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.reset_index(drop=True)
print(f"  -> Filas extraídas: {len(df)}")
print(f"  -> Columnas: {len(df.columns)}")

# ============================================================
# PASO 2: LIMPIEZA Y PREPARACIÓN
# ============================================================
print("\n[2/5] Limpiando datos...")

vars_vivienda = ['V01','V03','V04','V05','V06',
                 'V07','V08','V09','V10','V11']
vars_hogar    = ['H01','H05','H06','TOTPER']
vars_persona  = ['AUR','P01','P03','P12','P13',
                 'P15','P16','CONDACT','ANALF']
vars_vida     = vars_vivienda + vars_hogar

# Limpiar nulos en variables de condiciones de vida
df_clean = df.dropna(subset=vars_vida).reset_index(drop=True)
print(f"  -> Registros limpios: {len(df_clean)}")
print(f"  -> Nulos eliminados: {len(df) - len(df_clean)}")

# Guardar dataset limpio
df_clean.to_csv("01_datos_limpios.csv", index=False)

# ============================================================
# PASO 3: ANÁLISIS DE COMPONENTES PRINCIPALES (ACP)
# ============================================================
print("\n[3/5] Ejecutando ACP...")

scaler = StandardScaler()
X_vida = scaler.fit_transform(df_clean[vars_vida])

pca = PCA()
pca.fit(X_vida)

varianza = pca.explained_variance_ratio_ * 100
varianza_acum = np.cumsum(varianza)
n_comp = np.argmax(varianza_acum >= 80) + 1
print(f"  -> Componentes para 80% varianza: {n_comp}")
print(f"  -> CP1 explica: {varianza[0]:.2f}%")

# ACP con n componentes óptimos
pca_final = PCA(n_components=n_comp)
X_pca = pca_final.fit_transform(X_vida)

# Índice de Condiciones de Vida (ICV)
df_clean['ICV'] = -X_pca[:, 0]
df_clean['ICV'] = ((df_clean['ICV'] - df_clean['ICV'].min()) /
                   (df_clean['ICV'].max() - df_clean['ICV'].min()) * 100)

print(f"  -> ICV promedio: {df_clean['ICV'].mean():.2f}")
print(f"  -> ICV std: {df_clean['ICV'].std():.2f}")

# Gráfico varianza ACP
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(varianza)+1), varianza, color='steelblue')
axes[0].set_xlabel('Componente')
axes[0].set_ylabel('Varianza explicada (%)')
axes[0].set_title('Varianza por Componente')
axes[1].plot(range(1, len(varianza_acum)+1), varianza_acum, 'bo-')
axes[1].axhline(y=80, color='r', linestyle='--', label='80%')
axes[1].axvline(x=n_comp, color='g', linestyle='--', label=f'K={n_comp}')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Varianza Acumulada (%)')
axes[1].set_title('Varianza Acumulada (Codo)')
axes[1].legend()
plt.suptitle('ACP - Condiciones de Vida Ecuador 2022', fontsize=13)
plt.tight_layout()
plt.savefig('02_grafico_acp.png', dpi=150)
plt.close()
print("  -> Gráfico: 02_grafico_acp.png")

# ============================================================
# PASO 4: ANÁLISIS DE CLASES LATENTES (LCA/GMM)
# ============================================================
print("\n[4/5] Ejecutando LCA (selección de clases)...")

bic_scores, aic_scores = [], []
rango = range(2, 8)

for k in rango:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(X_pca)
    bic_scores.append(gmm.bic(X_pca))
    aic_scores.append(gmm.aic(X_pca))
    print(f"  K={k} | BIC={gmm.bic(X_pca):,.0f} | AIC={gmm.aic(X_pca):,.0f}")

mejor_k = rango[np.argmin(bic_scores)]
print(f"  -> Clases óptimas (BIC): {mejor_k}")

# Modelo final LCA
gmm_final = GaussianMixture(n_components=mejor_k, random_state=42, n_init=10)
gmm_final.fit(X_pca)
df_clean['Clase_LCA'] = gmm_final.predict(X_pca)
df_clean['Prob_LCA']  = gmm_final.predict_proba(X_pca).max(axis=1)

# Etiquetar clases por ICV
medias_icv = df_clean.groupby('Clase_LCA')['ICV'].mean().sort_values()
etiquetas = {}
niveles = ['Pobreza Extrema','Pobreza Moderada','Vulnerable',
           'No Pobre'] + [f'Nivel_{i}' for i in range(4, mejor_k)]
for i, clase in enumerate(medias_icv.index):
    etiquetas[clase] = niveles[i] if i < len(niveles) else f'Clase_{clase}'
df_clean['Perfil'] = df_clean['Clase_LCA'].map(etiquetas)

print("\n  === DISTRIBUCIÓN DE PERFILES ===")
dist = df_clean['Perfil'].value_counts()
for perfil, n in dist.items():
    pct = n/len(df_clean)*100
    print(f"  {perfil:<20}: {n:>6,} ({pct:.1f}%)")

print("\n  === ICV PROMEDIO POR PERFIL ===")
print(df_clean.groupby('Perfil')['ICV'].mean().round(2).to_string())

print(f"\n  === PROB. ASIGNACIÓN PROMEDIO ===")
print(df_clean.groupby('Perfil')['Prob_LCA'].mean().round(3).to_string())

# Gráficos LCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(rango), bic_scores, 'bo-', label='BIC')
axes[0].plot(list(rango), aic_scores, 'rs-', label='AIC')
axes[0].axvline(x=mejor_k, color='g', linestyle='--', label=f'Óptimo K={mejor_k}')
axes[0].set_xlabel('Número de Clases')
axes[0].set_ylabel('Score')
axes[0].set_title('Selección de Clases (BIC/AIC)')
axes[0].legend()

colores = {'Pobreza Extrema':'#d62728','Pobreza Moderada':'#ff7f0e',
           'Vulnerable':'#ffdd57','No Pobre':'#2ca02c',
           'Nivel_4':'#1f77b4','Nivel_5':'#9467bd'}
for perfil in df_clean['Perfil'].unique():
    idx = df_clean[df_clean['Perfil']==perfil].index
    color = colores.get(perfil, '#7f7f7f')
    axes[1].scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=color, label=perfil, alpha=0.3, s=3)
axes[1].set_xlabel(f'CP1 ({varianza[0]:.1f}%)')
axes[1].set_ylabel(f'CP2 ({varianza[1]:.1f}%)')
axes[1].set_title('Clases Latentes de Pobreza')
axes[1].legend(markerscale=3)
plt.suptitle('LCA - Perfiles de Pobreza Ecuador 2022', fontsize=13)
plt.tight_layout()
plt.savefig('03_grafico_lca.png', dpi=150)
plt.close()
print("  -> Gráfico: 03_grafico_lca.png")

# ============================================================
# PASO 5: REGRESIÓN LOGÍSTICA MULTINOMIAL
# ============================================================
print("\n[5/5] Ejecutando Regresión Logística Multinomial...")

vars_reg = ['AUR','P03','P13','P15','P16','CONDACT','ANALF']
df_reg = df_clean[vars_reg + ['Clase_LCA']].dropna().reset_index(drop=True)
print(f"  -> Registros para regresión: {len(df_reg)}")

X_reg = StandardScaler().fit_transform(df_reg[vars_reg])
y_reg = df_reg['Clase_LCA']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42, stratify=y_reg)

modelo = LogisticRegression(max_iter=2000, random_state=42)
modelo.fit(X_train, y_train)
acc = modelo.score(X_test, y_test)
print(f"  -> Accuracy: {acc:.4f}")
print("\n  === REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_test, modelo.predict(X_test)))

nombres_vars = ['Zona','Edad','Etnia','Alfabetismo',
                'NivelEdu','CondActiv','Analfabetismo']
coef_df = pd.DataFrame(modelo.coef_, columns=nombres_vars,
    index=[etiquetas.get(i, f'Clase_{i}')
           for i in range(modelo.coef_.shape[0])])

plt.figure(figsize=(12, 6))
sns.heatmap(coef_df, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, linewidths=0.5)
plt.title('Determinantes de Clases de Pobreza\nRegresión Logística Multinomial - Ecuador 2022')
plt.tight_layout()
plt.savefig('04_grafico_regresion.png', dpi=150)
plt.close()
print("  -> Gráfico: 04_grafico_regresion.png")

# ============================================================
# GUARDAR RESULTADOS FINALES
# ============================================================
df_clean.to_csv("05_resultados_finales.csv", index=False)
coef_df.to_csv("06_coeficientes_regresion.csv")

print("\n" + "=" * 60)
print("PIPELINE COMPLETADO")
print("=" * 60)
print("\nArchivos generados:")
print("  01_datos_limpios.csv")
print("  02_grafico_acp.png")
print("  03_grafico_lca.png")
print("  04_grafico_regresion.png")
print("  05_resultados_finales.csv")
print("  06_coeficientes_regresion.csv")
print(f"\nTotal registros analizados: {len(df_clean):,}")