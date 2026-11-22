from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from dash import no_update, callback_context
from variables import NUMERICAS, CATEGORICAS, COLUMNAS, agrupar_raras, agregar_derivadas, validar
import os
import requests

# Campos 
CAMPOS = list(CATEGORICAS) + list(NUMERICAS)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/")

def predecir(df):
    estudiante = df[CAMPOS].to_dict(orient="records")[0]
    respuesta = requests.post(
        f"{API_URL}/predict",
        json=estudiante,
        timeout=30,
    )
    respuesta.raise_for_status()
    return [respuesta.json()["probabilidad_desercion"]]

def predecir_lote_api(df):
    estudiantes = df[CAMPOS].to_dict(orient="records")

    try:
        respuesta = requests.post(
            f"{API_URL}/predict-batch",
            json=estudiantes,
            timeout=30,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "No se pudo enviar el lote a la API. Comprueba que Uvicorn esté corriendo."
        ) from error

    if respuesta.status_code == 422:
        detalle = respuesta.json()["detail"]
        raise ValueError(f"La API rechazó datos del archivo. Detalle: {detalle}")

    respuesta.raise_for_status()
    return np.asarray(respuesta.json()["probabilidades_desercion"])

UMBRAL = 0.5
def hay_desercion(p):
    return p >= UMBRAL

# Generacion de Campos
def crear_campo(col):
    if col in CATEGORICAS:
        etiqueta, opciones = CATEGORICAS[col]
        control = dcc.Dropdown(id=col, options=[{"label": t, "value": k} for k, t in opciones.items()])
    else:
        etiqueta, minimo, maximo = NUMERICAS[col]
        control = dcc.Input(id=col, type="number", min=minimo, max=maximo, debounce=True, style={"width": "100%"})
    return dbc.Col([
        html.Label(etiqueta + ":", style={"color": "white", "fontWeight": "bold", "fontSize": "20px"}),
        control
    ], width=4)

## Generacion de columnas 
filas_formulario = [
    dbc.Row([crear_campo(c) for c in CAMPOS[i:i + 2]], className="mb-2", style={"justifyContent": "center"})
    for i in range(0, len(CAMPOS), 2)
]


# Inicializar app
app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR], suppress_callback_exceptions=True)
server = app.server  # lo usa gunicorn en el contenedor


# Layout principal
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Predicción de Deserción Estudiantil", className="text-center text-light mb-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Tabs(id="tabs", value="tab1", children=[
                dcc.Tab(label="Entrada de Datos-Caso Individual", value="tab1", style={"color": "Black"}, selected_style={"color": "black", "fontWeight": "bold", "backgroundColor": "#ffc107"}),
                dcc.Tab(label="Resultados-Caso Individual", value="tab2", style={"color": "Black"}, selected_style={"color": "black", "fontWeight": "bold", "backgroundColor": "#ffc107"}),
                dcc.Tab(label="Revisión Excel- Caso Multiple", value="tab3", style={"color": "Black"}, selected_style={"color": "black", "fontWeight": "bold", "backgroundColor": "#ffc107"})
            ]),
            width=12
        )
    ]),

    html.Div(id="tab1-content", children=[
        html.Br(),
        dbc.Alert(
            html.Span([
                "Por favor, rellena todos los campos con la información del estudiante para obtener una predicción precisa.",
                html.Br(),
                "Posteriormente diríjase a la pestaña de resultados."
            ]),
            color="info",
            dismissable=False,
            className="text-center"
        ),

        html.Div(
            dbc.Button("🔄 Reiniciar", id="boton-reiniciar-individual-tab1", color="secondary", className="mt-4"),
            className="d-flex justify-content-end"
        ),

        html.Br(),

    ] + filas_formulario),

    html.Div(id="tab2-content", children=[
        html.Br(),
        dbc.Row([
            dbc.Col(html.H3("Resultado de la Predicción:", className="text-light mb-4 text-center"), width=12)
        ]),

        dbc.Row([
            dbc.Col(
                html.Div([
                    dbc.Button("🔍 Predecir", id="boton-predecir", color="warning", size="lg", className="px-4 mb-2"),
                ], className="text-center"),
                width="auto",
                className="mx-auto"
            )
        ]),

        dbc.Row([
            dbc.Col(
                html.Div(
                    dbc.Button("🔄 Reiniciar", id="boton-reiniciar-individual", color="secondary"),
                    className="d-flex justify-content-end"
                ),
                width=12
            )
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(
                html.Div(id="resultado-prediccion", children=[
                    dbc.Alert("ℹ️ Aún no se ha realizado ninguna predicción. Completa todos los campos en la pestaña de entrada y haz clic en 'Predecir'.",
                              id="mensaje-inicial", color="info", className="text-center")
                ]),
                width=12
            )
        ]),
    ], style={"display": "none"}),

    html.Div(id="tab3-content", children=[
        html.Br(),

        html.Div([
            dbc.Alert(
                "Por favor, adjunte los datos de los estudiantes a predecir en un archivo .xlsx o .csv con la primera fila con el nombre de la variable.",
                id="mensaje-carga",
                color="info",
                className="text-center",
                is_open=True
            ),
            dbc.Alert(
                "✅ Archivo cargado correctamente.",
                id="mensaje-exito",
                color="success",
                className="text-center",
                is_open=False,
                dismissable=False
            ),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Label("Número de estudiantes a priorizar", style={"color": "white", "fontWeight": "bold", "fontSize": "20px"}),
                        dcc.Input(id="n-casos", type="number", min=1, step=1, debounce=True, style={"width": "100%"})
                    ], width=4),
                    dbc.Col(
                        dbc.Button("🔄 Reiniciar", id="boton-reiniciar-masiva", color="secondary"),
                        className="d-flex justify-content-end align-items-end"
                    )
                ], className="mt-2"),

                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Arrastra o haz click para cargar el archivo (.xlsx o .csv)'], style={"color": "#ffffff", "fontWeight": "bold"}),

                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px', 'backgroundColor': '#154360'
                    },
                    multiple=False
                ),

                html.Div(id='output-table'),
                dcc.Download(id="descarga")
            ])
        ])
    ], style={"display": "none"})
], fluid=True)


# Para alterar entre pestañas
@app.callback(
    [Output("tab1-content", "style"),
     Output("tab2-content", "style"),
     Output("tab3-content", "style")],
    Input("tabs", "value")
)
def alternar_pestanas(tab):
    return [
        {"display": "block"} if tab == "tab1" else {"display": "none"},
        {"display": "block"} if tab == "tab2" else {"display": "none"},
        {"display": "block"} if tab == "tab3" else {"display": "none"}
    ]


# Callback para reiniciar 
@app.callback(
    [Output(c, "value") for c in CAMPOS],
    Input("boton-reiniciar-individual-tab1", "n_clicks"),
    prevent_initial_call=True
)
def reiniciar_campos(n_clicks):
    return [None] * len(CAMPOS)
 
 
# Callback para ejecutar la predicción individual 
@app.callback(
    Output("resultado-prediccion", "children"),
    [Input("boton-predecir", "n_clicks"),
     Input("boton-reiniciar-individual", "n_clicks")],
    [State(c, "value") for c in CAMPOS],
)
def manejar_prediccion(n_clicks_predecir, n_clicks_reiniciar, *inputs):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
 
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
 
    if trigger_id == "boton-reiniciar-individual":
        return dbc.Alert(
            ["ℹ️ Aún no se ha realizado ninguna predicción.",
             html.Br(),
             "Completa todos los campos en la pestaña de entrada y haz clic en 'Predecir'."],
            color="info",
            className="text-center"
        )
 
    datos_dict = dict(zip(CAMPOS, inputs))
 
    # Validar que no haya campos vacíos
    if any(valor is None or valor == '' for valor in datos_dict.values()):
        return dbc.Alert("⚠️ Por favor completa todos los campos.", color="warning")
 
    # Validar rangos y coherencia (aprobadas <= inscritas)
    errores = []
    for col, (etiqueta, minimo, maximo) in NUMERICAS.items():
        if not minimo <= datos_dict[col] <= maximo:
            errores.append(f"{etiqueta}: debe estar entre {minimo} y {maximo}.")
    if not errores:
        errores = validar(datos_dict)
    if errores:
        return dbc.Alert([html.Div("⚠️ " + e) for e in errores], color="warning")
 
    # Realizar predicción
    df_input = pd.DataFrame([datos_dict])
    probabilidad = round(float(predecir(df_input)[0]), 3)
 
    # Mostrar resultado
    if hay_desercion(probabilidad):
        color_card = "danger"
        icono = "❌"
        titulo = "Riesgo de Deserción Detectado"
        mensaje = "Se espera que el estudiante abandone.\nRecomendamos un seguimiento prioritario."
    else:
        color_card = "success"
        icono = "✅"
        titulo = "Sin Riesgo de Deserción"
        mensaje = "No se espera que el estudiante abandone."
 
    # Gráfico tipo 
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad * 100,
        title={'text': "Probabilidad de Deserción (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, UMBRAL * 100], 'color': '#d4edda'},
                {'range': [UMBRAL * 100, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probabilidad * 100
            }
        }
    ))
 
    return dbc.Card(
        dbc.CardBody([
            html.H1(icono, className="text-center", style={"fontSize": "90px"}),
            html.H2(titulo, className="text-center"),
            html.P(mensaje, className="text-center", style={"whiteSpace": "pre-line"}),
            html.H4(f"Probabilidad estimada de deserción: {probabilidad * 100:.1f}%", className="text-center mt-4"),
            dcc.Graph(figure=gauge_fig)
        ]),
        color=color_card,
        inverse=True,
        className="mt-4"
    )
 
 
# Callback para cargar archivo Excel 
@app.callback(
    [Output('output-table', 'children'),
     Output('mensaje-carga', 'is_open'),
     Output('mensaje-exito', 'is_open')],
    [Input('upload-data', 'contents'),
     Input('boton-reiniciar-masiva', 'n_clicks'),
     Input('n-casos', 'value')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def manejar_carga_y_reinicio(contents, n_clicks_reiniciar, n_casos, filename):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
 
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
 
    if trigger == 'boton-reiniciar-masiva':
        return None, True, False
 
    if contents is None:
        return None, True, False
 
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(decoded), sep=None, engine="python", encoding="utf-8-sig")
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return dbc.Alert("Formato no soportado. Sube un archivo .xlsx o .csv.", color="warning"), False, False
    except Exception as e:
        return dbc.Alert(f"No se pudo leer el archivo: {e}", color="danger"), False, False

    if df.empty:
        return dbc.Alert("El archivo está vacío o no tiene columnas.", color="warning"), False, False
 
    try:
        df.columns = df.columns.astype(str).str.strip()
 
        faltan = [c for c in CAMPOS if c not in df.columns]
        if faltan:
            return dbc.Alert(f"Faltan columnas en el archivo: {', '.join(faltan)}", color="danger"), False, False
 
        df = df[CAMPOS]
        if df.isna().any().any():
            return dbc.Alert("El archivo tiene celdas vacías. Completa todos los datos.", color="warning"), False, False
 
        for i, fila in df.iterrows():
            errores = validar(fila.to_dict())
            if errores:
                return dbc.Alert(f"Fila {i + 2} del Excel: {errores[0]}", color="warning"), False, False
 
        probs = predecir_lote_api(df)
 
        df_resultado = df.copy()
        df_resultado['Probabilidad'] = np.round(probs, 3)
        df_resultado['Resultado'] = np.where(probs >= UMBRAL, "Deserción", "No deserción")

        # Ordenar de mayor a menor probabilidad
        df_resultado = df_resultado.sort_values('Probabilidad', ascending=False, kind="stable").reset_index(drop=True)
        try:
            n = min(max(int(n_casos), 0), len(df_resultado))
        except (TypeError, ValueError):
            n = 0
        df_resultado.insert(0, 'Prioridad', [str(i + 1) if i < n else "—" for i in range(len(df_resultado))])

        # Tabla 
        header = [html.Th(col) for col in df_resultado.columns]
        rows = []
        for i in range(len(df_resultado)):
            fila = []
            for col in df_resultado.columns:
                valor = df_resultado.iloc[i][col]
                if col == "Prioridad":
                    estilo = {"backgroundColor": "#ffc107", "color": "black", "fontWeight": "bold", "textAlign": "center", "verticalAlign": "middle"} if i < n else {"textAlign": "center", "verticalAlign": "middle"}
                    fila.append(html.Td(valor, style=estilo))
                elif col == "Resultado":
                    es_desercion = valor == "Deserción"
                    icono = "❌" if es_desercion else "✅"
                    color = "#f8d7da" if es_desercion else "#d4edda"
                    fila.append(html.Td(
                        html.Div([
                            html.Div(icono, style={"fontSize": "26px", "fontWeight": "bold"}),
                            html.Div(valor, style={"fontSize": "15px", "color": "black"})
                        ], style={"textAlign": "center"}),
                        style={
                            "backgroundColor": color,
                            "verticalAlign": "middle"
                        }
                    ))
                else:
                    estilo = {"backgroundColor": "rgba(255, 193, 7, 0.25)"} if i < n else {}
                    fila.append(html.Td(str(valor), style=estilo))
            rows.append(html.Tr(fila))

        tabla = dbc.Table(
            [html.Thead(html.Tr(header))] + [html.Tbody(rows)],
            striped=True, bordered=True, hover=True, responsive=True
        )

        if n == 0:
            texto_resumen = (f"Se muestran los {len(df_resultado)} estudiantes de mayor a menor probabilidad de deserción. "
                             "Escribe cuántos casos puedes revisar para resaltar los prioritarios.")
        else:
            texto_resumen = (f"Se muestran los {len(df_resultado)} estudiantes de mayor a menor probabilidad de deserción. "
                             f"Los {n} primeros (resaltados) son los que conviene revisar primero.")
        resumen = html.P(texto_resumen, className="text-light mt-3")

        # Botón para descargar los N 
        boton_descarga = html.Div(
            dbc.Button(f"⬇️ Descargar los {n} prioritarios (.xlsx)", id="boton-descargar", color="warning"),
            className="mb-3"
        ) if n > 0 else None

        tabla_scroll = html.Div(tabla, style={"maxHeight": "450px", "overflowY": "auto"})

        return html.Div([resumen, boton_descarga, tabla_scroll, construir_eda(df_resultado, n)]), False, True
 
    except Exception as e:
        return dbc.Alert(f"Error procesando el archivo: {e}", color="danger"), False, False
 
 
## Graficas para EDA de los cargados
COLOR_PRIORIDAD = "#ffc107"
COLOR_RESTO = "#2aa198"
ESTADOS = {"normal": "Normal", "no_inscrito": "No inscrito",
           "inscrito_sin_eval": "Inscrito sin evaluación", "evaluado_sin_aprobar": "Evaluado sin aprobar"}


def grafica(fig, titulo):
    """Da el mismo formato a todas las gráficas y las pone en media fila."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=titulo, height=340, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}), md=6)


def construir_eda(resultado, n):
    """resultado: estudiantes ya ordenados de mayor a menor probabilidad, con las columnas del formulario y 'Probabilidad'."""
    d = agregar_derivadas(agrupar_raras(resultado[CAMPOS]))
    d["Probabilidad"] = resultado["Probabilidad"].to_numpy()
    total = len(d)

    if n > 0:
        nombres = ("Prioritarios", "Resto")
        d["grupo"] = np.where(np.arange(total) < n, nombres[0], nombres[1])
    else:
        nombres = ("Con riesgo", "Sin riesgo")
        d["grupo"] = np.where(d["Probabilidad"] >= UMBRAL, nombres[0], nombres[1])
    colores = {nombres[0]: COLOR_PRIORIDAD, nombres[1]: COLOR_RESTO}
    grupos = [(g, d[d["grupo"] == g]) for g in nombres if (d["grupo"] == g).any()]

    # Tarjetas de resumen
    con_riesgo = int((d["Probabilidad"] >= UMBRAL).sum())

    def tarjeta(titulo, valor):
        return dbc.Col(dbc.Card(dbc.CardBody([html.Div(titulo), html.H3(valor)]), className="text-center"),
                       md=3, className="mb-3")

    kpis = dbc.Row([
        tarjeta("Estudiantes cargados", f"{total}"),
        tarjeta("Prioritarios", f"{n}" if n > 0 else "—"),
        tarjeta("Probabilidad promedio", f"{d['Probabilidad'].mean() * 100:.1f}%"),
        tarjeta(f"Con riesgo (≥ {UMBRAL * 100:.0f}%)", f"{con_riesgo} ({con_riesgo / total * 100:.0f}%)"),
    ])

    # 1. Distribución de la probabilidad
    f1 = go.Figure(go.Histogram(x=d["Probabilidad"] * 100, xbins=dict(start=0, end=100, size=5), marker_color=COLOR_RESTO))
    f1.add_vline(x=UMBRAL * 100, line_dash="dash", line_color="red", annotation_text="Umbral")
    if 0 < n < total:
        f1.add_vline(x=d["Probabilidad"].iloc[n - 1] * 100, line_color=COLOR_PRIORIDAD,
                     annotation_text="Corte de los N", annotation_position="top left")
    f1.update_layout(xaxis_title="Probabilidad de deserción (%)", yaxis_title="Estudiantes")

    # 2. Perfil: % de estudiantes con cada característica
    indicadores = {
        "Mayor de 25 años": lambda x: x["es_adulto"] == 1,
        "Sin aprobadas 1.er sem": lambda x: x["Curricular units 1st sem (approved)"] == 0,
        "Vive lejos de su origen": lambda x: x["Displaced"] == 1,
        "Jornada nocturna": lambda x: x["Daytime/evening attendance"] == 0,
        "Hombre": lambda x: x["Gender"] == 1,
        "1.er sem. no normal": lambda x: x["estado_1st"] != "normal",
    }
    f2 = go.Figure([go.Bar(name=g, x=list(indicadores), y=[f(sub).mean() * 100 for f in indicadores.values()],
                           marker_color=colores[g]) for g, sub in grupos])
    f2.update_layout(barmode="group", yaxis_title="% de estudiantes", yaxis_range=[0, 100])

    # 3. Estudiantes por programa (los 8 más frecuentes)
    programa = d["Course"].map(lambda c: CATEGORICAS["Course"][1].get(c, str(c)))
    top = programa.value_counts().head(8).index
    f3 = go.Figure([go.Bar(name=g, y=top, x=[(programa[sub.index] == p).sum() for p in top], orientation="h",
                           marker_color=colores[g]) for g, sub in grupos])
    f3.update_layout(barmode="stack", xaxis_title="Estudiantes", yaxis=dict(autorange="reversed"),
                     xaxis_dtick=1 if programa.value_counts().max() <= 8 else None)   # sin medios estudiantes en el eje

    # 4. Estado académico del 1.er semestre
    f4 = go.Figure([go.Bar(name=g, x=list(ESTADOS.values()), y=[(sub["estado_1st"] == k).sum() for k in ESTADOS],
                           marker_color=colores[g]) for g, sub in grupos])
    f4.update_layout(barmode="group", yaxis_title="Estudiantes")

    # 5. Edad al ingresar
    f5 = go.Figure([go.Histogram(name=g, x=sub["Age at enrollment"], xbins=dict(size=2), marker_color=colores[g], opacity=0.75)
                    for g, sub in grupos])
    f5.update_layout(barmode="overlay", xaxis_title="Edad al ingresar", yaxis_title="Estudiantes")

    # 6. Unidades del 1.er semestre (promedio)
    f6 = go.Figure([go.Bar(name=g, x=["Inscritas", "Aprobadas"],
                           y=[sub["Curricular units 1st sem (enrolled)"].mean(), sub["Curricular units 1st sem (approved)"].mean()],
                           marker_color=colores[g]) for g, sub in grupos])
    f6.update_layout(barmode="group", yaxis_title="Unidades (promedio)")

    return html.Div([
        html.H4("Perfil de los casos cargados", className="text-light mt-4"),
        html.P(f"Descripción de los estudiantes del archivo, comparando {nombres[0].lower()} con {nombres[1].lower()}. "
               "Son descripciones de estos datos.", className="text-light"),
        kpis,
        dbc.Row([grafica(f1, "Distribución de la probabilidad de deserción"), grafica(f2, "Perfil: % de estudiantes con cada característica")]),
        dbc.Row([grafica(f3, "Estudiantes por programa (8 más frecuentes)"), grafica(f4, "Estado académico del 1.er semestre")]),
        dbc.Row([grafica(f5, "Edad al ingresar"), grafica(f6, "Unidades 1.er semestre (promedio)")]),
    ])


# Callback para descargar en Excel
@app.callback(
    Output("descarga", "data"),
    Input("boton-descargar", "n_clicks"),
    [State("upload-data", "contents"),
     State("upload-data", "filename"),
     State("n-casos", "value")],
    prevent_initial_call=True
)
def descargar_prioritarios(n_clicks, contents, filename, n_casos):
    if not n_clicks or contents is None or not n_casos:
        raise PreventUpdate

    decoded = base64.b64decode(contents.split(',')[1])
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(decoded), sep=None, engine="python", encoding="utf-8-sig")
    else:
        df = pd.read_excel(io.BytesIO(decoded))
    df.columns = df.columns.astype(str).str.strip()
    df = df[CAMPOS]

    probs = predecir_lote_api(df)
    resultado = df.copy()
    resultado['Probabilidad'] = np.round(probs, 3)
    resultado['Resultado'] = np.where(probs >= UMBRAL, "Deserción", "No deserción")
    resultado = resultado.sort_values('Probabilidad', ascending=False, kind="stable").reset_index(drop=True)

    n = min(max(int(n_casos), 1), len(resultado))
    prioritarios = resultado.head(n)
    prioritarios.insert(0, 'Prioridad', range(1, n + 1))

    return dcc.send_data_frame(prioritarios.to_excel, f"{n}_estudiantes_prioritarios.xlsx",
                               index=False, sheet_name="Prioritarios")


# Callback para reiniciar la carga de archivo
@app.callback(
    [Output('upload-data', 'contents'),
     Output('n-casos', 'value')],
    Input('boton-reiniciar-masiva', 'n_clicks'),
    prevent_initial_call=True
)
def limpiar_archivo(n_clicks):
    return None, None

# Correr app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)