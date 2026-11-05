from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import json
import base64
import io
from xgboost import XGBClassifier
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from dash import no_update, callback_context

from variables import NUMERICAS, CATEGORICAS, COLUMNAS, agrupar_raras, agregar_derivadas, validar


# Cargar modelo
modelo = XGBClassifier()
modelo.load_model("Dash/app/xgboost_dropout.json")      


columnas_modelo = modelo.get_booster().feature_names


# Campos del formulario: el id de cada componente es el nombre de la columna
CAMPOS = list(CATEGORICAS) + list(NUMERICAS)
# Columnas que se convirtieron a dummies al entrenar
COLS_DUMMY = list(CATEGORICAS) + ["estado_1st", "estado_2nd"]


# Preprocesamiento (igual que en el entrenamiento) y predicción
def preparar(df):
    df = agregar_derivadas(agrupar_raras(df))
    X = pd.get_dummies(df[COLUMNAS], columns=COLS_DUMMY, drop_first=True)
    return X.astype(float)


def predecir(df):
    return modelo.predict_proba(preparar(df))[:, 1]


def nivel_riesgo(p):
    if p < 0.33:
        return "Bajo"
    if p < 0.66:
        return "Medio"
    return "Alto"


COLOR_NIVEL = {"Bajo": "#d4edda", "Medio": "#fff3cd", "Alto": "#f8d7da"}


# Crea un campo del formulario (lista desplegable o número) a partir de variables.py
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


filas_formulario = [
    dbc.Row([crear_campo(c) for c in CAMPOS[i:i + 2]], className="mb-2", style={"justifyContent": "center"})
    for i in range(0, len(CAMPOS), 2)
]


# Inicializar app
app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR], suppress_callback_exceptions=True)


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

        # Campos generados desde variables.py
        *filas_formulario

    ]),

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
                "Por favor, adjunte los datos de los estudiantes a predecir en un archivo .xlsx con la primera fila con el nombre de la variable.",
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
                html.Div(
                    dbc.Button("🔄 Reiniciar", id="boton-reiniciar-masiva", color="secondary", className="mt-2"),
                    className="d-flex justify-content-end"
                ),
                html.Div(id='output-table'),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Arrastra o haz click para cargar el archivo .xlsx'], style={"color": "#ffffff", "fontWeight": "bold"}),

                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px', 'backgroundColor': '#154360'
                    },
                    multiple=False
                )
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


# Callback para reiniciar todos los campos de la pestaña de entrada individual
@app.callback(
    [Output(c, "value") for c in CAMPOS],
    Input("boton-reiniciar-individual-tab1", "n_clicks"),
    prevent_initial_call=True
)
def reiniciar_campos(n_clicks):
    return [None] * len(CAMPOS)


# Callback para ejecutar la predicción individual y mostrar mensaje de reinicio en la pestaña de resultados
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

    # Validar rangos y coherencia (aprobadas <= inscritas, nota 0 solo si no aprobó)
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
    nivel = nivel_riesgo(probabilidad)

    # Mostrar resultado
    if nivel == "Alto":
        color_card = "danger"
        icono = "❌"
        titulo = "Riesgo Alto de Deserción"
        mensaje = "Alta probabilidad de que el estudiante abandone.\nRecomendamos un seguimiento prioritario."
    elif nivel == "Medio":
        color_card = "warning"
        icono = "⚠️"
        titulo = "Riesgo Medio de Deserción"
        mensaje = "Hay señales de riesgo.\nRecomendamos hacer seguimiento."
    else:
        color_card = "success"
        icono = "✅"
        titulo = "Riesgo Bajo de Deserción"
        mensaje = "No se espera que el estudiante abandone.\nBajo riesgo de deserción."

    # Gráfico tipo gauge
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad * 100,
        title={'text': "Probabilidad de Deserción (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 33], 'color': '#d4edda'},
                {'range': [33, 66], 'color': '#fff3cd'},
                {'range': [66, 100], 'color': '#f8d7da'}
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


# Callback para cargar archivo Excel en la pestaña de revisión masiva, predecir y mostrar tabla de resultados
@app.callback(
    [Output('output-table', 'children'),
     Output('mensaje-carga', 'is_open'),
     Output('mensaje-exito', 'is_open')],
    [Input('upload-data', 'contents'),
     Input('boton-reiniciar-masiva', 'n_clicks')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def manejar_carga_y_reinicio(contents, n_clicks_reiniciar, filename):
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
    df = pd.read_excel(io.BytesIO(decoded))

    if df.empty:
        return dbc.Alert("El archivo está vacío o no tiene columnas.", color="warning"), False, False

    try:
        df.columns = df.columns.astype(str).str.strip()

        # El Excel debe traer las columnas del formulario
        faltan = [c for c in CAMPOS if c not in df.columns]
        if faltan:
            return dbc.Alert(f"Faltan columnas en el archivo: {', '.join(faltan)}", color="danger"), False, False

        df = df[CAMPOS]
        if df.isna().any().any():
            return dbc.Alert("El archivo tiene celdas vacías. Completa todos los datos.", color="warning"), False, False

        # Validar coherencia fila por fila
        for i, fila in df.iterrows():
            errores = validar(fila.to_dict())
            if errores:
                return dbc.Alert(f"Fila {i + 2} del Excel: {errores[0]}", color="warning"), False, False

        probs = predecir(df)

        df_resultado = df.copy()
        df_resultado['Probabilidad'] = np.round(probs, 3)
        df_resultado['Resultado'] = [nivel_riesgo(p) for p in probs]

        # Tabla estilizada
        header = [html.Th(col) for col in df_resultado.columns]
        rows = []
        for i in range(len(df_resultado)):
            fila = []
            for col in df_resultado.columns:
                valor = df_resultado.iloc[i][col]
                if col == "Resultado":
                    icono = {"Alto": "❌", "Medio": "⚠️", "Bajo": "✅"}[valor]
                    fila.append(html.Td(
                        html.Div([
                            html.Div(icono, style={"fontSize": "26px", "fontWeight": "bold"}),
                            html.Div(f"Riesgo {valor.lower()}", style={"fontSize": "15px", "color": "black"})
                        ], style={"textAlign": "center"}),
                        style={
                            "backgroundColor": COLOR_NIVEL[valor],
                            "verticalAlign": "middle"
                        }
                    ))
                else:
                    fila.append(html.Td(str(valor)))
            rows.append(html.Tr(fila))

        tabla = dbc.Table(
            [html.Thead(html.Tr(header))] + [html.Tbody(rows)],
            striped=True, bordered=True, hover=True, responsive=True
        )

        return tabla, False, True

    except Exception as e:
        return dbc.Alert(f"Error procesando el archivo: {e}", color="danger"), False, False


# Correr app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)