from pathlib import Path

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from variables import CATEGORICAS, NUMERICAS, agregar_derivadas, agrupar_raras, validar

app = FastAPI(title="API de riesgo de deserción estudiantil")

modelo = joblib.load(Path(__file__).with_name("model.pkl"))
columnas_modelo = list(modelo.feature_names_in_)
indice_desercion = list(modelo.classes_).index(1)
campos_esperados = set(CATEGORICAS) | set(NUMERICAS)

EJEMPLO_ESTUDIANTE = {
    "Gender": 0,
    "Marital status": 1,
    "Displaced": 0,
    "Mother's qualification": 1,
    "Father's qualification": 1,
    "Mother's occupation": 4,
    "Father's occupation": 4,
    "Course": 9500,
    "Application mode": 1,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Age at enrollment": 20,
    "Application order": 1,
    "Previous qualification (grade)": 130,
    "Admission grade": 140,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 6,
    "Curricular units 1st sem (approved)": 4,
    "Curricular units 1st sem (grade)": 12,
    "Curricular units 1st sem (without evaluations)": 0,
    "Unemployment rate": 10,
    "Inflation rate": 2,
    "GDP": 1,
}


def validar_estudiante(estudiante: dict[str, float]) -> None:
    faltantes = campos_esperados - estudiante.keys()
    adicionales = estudiante.keys() - campos_esperados

    if faltantes or adicionales:
        raise HTTPException(
            status_code=422,
            detail={
                "campos_faltantes": sorted(faltantes),
                "campos_adicionales": sorted(adicionales),
            },
        )

    errores = [
        f"{campo}: debe estar entre {minimo} y {maximo}."
        for campo, (_, minimo, maximo) in NUMERICAS.items()
        if not minimo <= estudiante[campo] <= maximo
    ]

    for campo, (_, opciones) in CATEGORICAS.items():
        codigo = estudiante[campo]
        if not codigo.is_integer() or (codigo not in opciones and -1 not in opciones):
            errores.append(f"{campo}: código de categoría inválido.")

    errores.extend(validar(estudiante))

    if errores:
        raise HTTPException(status_code=422, detail=errores)


def calcular_probabilidades(estudiantes: list[dict[str, float]]) -> list[float]:
    datos = pd.DataFrame(estudiantes)

    for campo in CATEGORICAS:
        datos[campo] = datos[campo].astype(int)

    preparados = agregar_derivadas(agrupar_raras(datos))
    probabilidades = modelo.predict_proba(preparados[columnas_modelo])

    return probabilidades[:, indice_desercion].tolist()


@app.post("/predict")
def predecir(
    estudiante: dict[str, float] = Body(
        openapi_examples={
            "estudiante": {
                "summary": "Ejemplo de estudiante",
                "value": EJEMPLO_ESTUDIANTE,
            }
        },
    ),
):
    validar_estudiante(estudiante)
    return {"probabilidad_desercion": calcular_probabilidades([estudiante])[0]}


@app.post("/predict-batch")
def predecir_lote(
    estudiantes: list[dict[str, float]] = Body(
        min_length=1,
        openapi_examples={
            "lote": {
                "summary": "Lote de un estudiante",
                "value": [EJEMPLO_ESTUDIANTE],
            }
        },
    ),
):
    for fila, estudiante in enumerate(estudiantes, start=1):
        try:
            validar_estudiante(estudiante)
        except HTTPException as error:
            raise HTTPException(
                status_code=422,
                detail={"fila": fila, "errores": error.detail},
            ) from error

    return {"probabilidades_desercion": calcular_probabilidades(estudiantes)}