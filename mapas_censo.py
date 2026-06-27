import pandas as pd
import json
import folium
import warnings
warnings.filterwarnings('ignore')

print("=== GENERANDO MAPAS DE POBREZA - ECUADOR 2022 ===")

# ── Cargar GeoJSON local ────────────────────────────────
print("\n[1/4] Cargando GeoJSON...")
with open("ecuador_provincias.geojson", "r", encoding="utf-8") as f:
    geojson = json.load(f)
print(f"  Provincias: {len(geojson['features'])}")

# ── Cargar y procesar datos del censo ──────────────────
print("\n[2/4] Procesando datos...")
df = pd.read_csv("05_resultados_finales.csv")
df['CANTON'] = df['CANTON'].astype(str).str.zfill(4)
df['provincia_cod'] = df['CANTON'].str[:2]

codigos = {
    '01':'Azuay','02':'Bolivar','03':'Cañar','04':'Carchi',
    '05':'Cotopaxi','06':'Chimborazo','07':'El Oro','08':'Esmeraldas',
    '09':'Guayas','10':'Imbabura','11':'Loja','12':'Los Rios',
    '13':'Manabi','14':'Morona Santiago','15':'Napo','16':'Pastaza',
    '17':'Pichincha','18':'Tungurahua','19':'Zamora Chinchipe',
    '20':'Galapagos','21':'Sucumbios','22':'Orellana',
    '23':'Santo Domingo de los Tsachilas','24':'Santa Elena'
}
df['provincia'] = df['provincia_cod'].map(codigos)

resumen = df.groupby('provincia').agg(
    ICV_promedio=('ICV','mean'),
    n_registros=('ICV','count'),
    pct_urbano=('AUR', lambda x: (pd.to_numeric(x,errors='coerce')==1).mean()*100)
).reset_index().round(2)

pobreza = df[df['Perfil'].isin(['Pobreza Extrema','Pobreza Moderada'])]
pct_pobre = (pobreza.groupby('provincia').size() /
             df.groupby('provincia').size() * 100).reset_index()
pct_pobre.columns = ['provincia','pct_pobreza']
resumen = resumen.merge(pct_pobre, on='provincia', how='left').fillna(0).round(2)

print(resumen.sort_values('ICV_promedio')[['provincia','ICV_promedio','pct_pobreza','n_registros']].to_string(index=False))
resumen.to_csv("08_resumen_provincial.csv", index=False)

# ── MAPA 1: ICV ─────────────────────────────────────────
print("\n[3/4] Generando mapa ICV...")
m1 = folium.Map(location=[-1.8,-78.2], zoom_start=6, tiles='CartoDB positron')

cp1 = folium.Choropleth(
    geo_data=geojson,
    data=resumen,
    columns=['provincia','ICV_promedio'],
    key_on='feature.properties.name',
    fill_color='RdYlGn',
    fill_opacity=0.8,
    line_opacity=0.5,
    legend_name='Índice de Condiciones de Vida (0-100)',
    nan_fill_color='lightgrey',
    highlight=True
).add_to(m1)

# Tooltip interactivo
resumen_dict = resumen.set_index('provincia').to_dict('index')
for feature in geojson['features']:
    nombre = feature['properties']['name']
    if nombre in resumen_dict:
        d = resumen_dict[nombre]
        tooltip = f"""
        <b>{nombre}</b><br>
        ICV: {d['ICV_promedio']:.1f}/100<br>
        % Pobreza: {d['pct_pobreza']:.1f}%<br>
        % Urbano: {d['pct_urbano']:.1f}%<br>
        Registros: {int(d['n_registros']):,}
        """
        folium.GeoJson(
            feature,
            style_function=lambda x: {'fillOpacity':0,'weight':0},
            tooltip=folium.Tooltip(tooltip)
        ).add_to(m1)

folium.map.Marker(
    [-0.2, -78.5],
    icon=folium.DivIcon(html='<div style="font-size:14px;font-weight:bold;color:#333">Índice de Condiciones de Vida<br>Ecuador — Censo 2022</div>')
).add_to(m1)

m1.save("09_mapa_icv.html")
print("  -> 09_mapa_icv.html")

# ── MAPA 2: % Pobreza ───────────────────────────────────
print("\n[4/4] Generando mapa pobreza...")
m2 = folium.Map(location=[-1.8,-78.2], zoom_start=6, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=geojson,
    data=resumen,
    columns=['provincia','pct_pobreza'],
    key_on='feature.properties.name',
    fill_color='Reds',
    fill_opacity=0.8,
    line_opacity=0.5,
    legend_name='% Pobreza Extrema + Moderada',
    nan_fill_color='lightgrey',
    highlight=True
).add_to(m2)

for feature in geojson['features']:
    nombre = feature['properties']['name']
    if nombre in resumen_dict:
        d = resumen_dict[nombre]
        tooltip = f"<b>{nombre}</b><br>% Pobreza: {d['pct_pobreza']:.1f}%<br>ICV: {d['ICV_promedio']:.1f}"
        folium.GeoJson(
            feature,
            style_function=lambda x: {'fillOpacity':0,'weight':0},
            tooltip=folium.Tooltip(tooltip)
        ).add_to(m2)

m2.save("10_mapa_pobreza.html")
print("  -> 10_mapa_pobreza.html")

print("\n=== MAPAS COMPLETADOS ===")
print("Abre los archivos .html en tu navegador para ver los mapas interactivos")