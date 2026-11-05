"""
Variables.py:
Codigo realizado para crear diccionarios con los codigos y valor referencia del repositorio de UCI, para poder hacer la 
traduccion de los datos y poder ser utilizados en el modelo de prediccion de desercion estudiantil.

"""
import numpy as np

SI_NO = {0: "No", 1: "Sí"}

ESTUDIOS = {
    1: "Secundaria (12.º año) o equivalente", 2: "Educación superior · licenciatura",
    3: "Educación superior · grado", 4: "Educación superior · maestría",
    19: "Educación básica 3.er ciclo (9.º–11.º) o equivalente", 34: "Desconocida",
    37: "Educación básica 1.er ciclo (4.º–5.º) o equivalente",
    38: "Educación básica 2.º ciclo (6.º–8.º) o equivalente", -1: "Otro (agrupado)",
}

OCUPACION = {
    0: "Estudiante", 1: "Directivos y representantes del poder legislativo o ejecutivo",
    2: "Especialistas en actividades intelectuales y científicas",
    3: "Técnicos y profesiones de nivel intermedio", 4: "Personal administrativo",
    5: "Servicios personales, seguridad y ventas",
    6: "Agricultores y trabajadores agropecuarios cualificados",
    7: "Trabajadores cualificados de industria, construcción y oficios",
    8: "Operadores de instalaciones y máquinas, ensambladores",
    9: "Trabajadores no cualificados", 10: "Fuerzas armadas", 90: "Otra situación",
    -1: "Otro (agrupado)",
}

## Campos numericos, se ingresan a mano
NUMERICAS = {
    "Age at enrollment": ("Edad al ingresar", 16, 80),
    "Application order": ("Orden de preferencia de la carrera (0 = primera opción)", 0, 9),
    "Previous qualification (grade)": ("Nota de estudios previos (0–200)", 0, 200),
    "Admission grade": ("Nota de admisión (0–200)", 0, 200),

    "Curricular units 1st sem (credited)": ("1.er semestre · Unidades reconocidas", 0, 60),
    "Curricular units 1st sem (enrolled)": ("1.er semestre · Unidades inscritas", 0, 60),
    "Curricular units 1st sem (evaluations)": ("1.er semestre · Evaluaciones presentadas", 0, 100),
    "Curricular units 1st sem (approved)": ("1.er semestre · Unidades aprobadas", 0, 60),
    "Curricular units 1st sem (grade)": ("1.er semestre · Nota promedio (0–20)", 0, 20),
    "Curricular units 1st sem (without evaluations)": ("1.er semestre · Unidades sin evaluación", 0, 60),

    "Curricular units 2nd sem (credited)": ("2.º semestre · Unidades reconocidas", 0, 60),
    "Curricular units 2nd sem (enrolled)": ("2.º semestre · Unidades inscritas", 0, 60),
    "Curricular units 2nd sem (evaluations)": ("2.º semestre · Evaluaciones presentadas", 0, 100),
    "Curricular units 2nd sem (approved)": ("2.º semestre · Unidades aprobadas", 0, 60),
    "Curricular units 2nd sem (grade)": ("2.º semestre · Nota promedio (0–20)", 0, 20),
    "Curricular units 2nd sem (without evaluations)": ("2.º semestre · Unidades sin evaluación", 0, 60),

    "Unemployment rate": ("Tasa de desempleo (%)", 0, 30),
    "Inflation rate": ("Tasa de inflación (%)", -5, 20),
    "GDP": ("PIB", -15, 15),
}

# ---------- Campos que se ELIGEN: columna -> (etiqueta, {código: texto}) ----------
# El -1 es la categoría "Otro" con la que se agruparon las categorías raras.
CATEGORICAS = {
    "Gender": ("Género", {0: "Femenino", 1: "Masculino"}),
    "Marital status": ("Estado civil", {1: "Soltero(a)", 2: "Casado(a)", 4: "Divorciado(a)", -1: "Otro (agrupado)"}),
    "Displaced": ("Vive lejos de su lugar de origen", SI_NO),
    "Mother's qualification": ("Estudios de la madre", ESTUDIOS),
    "Father's qualification": ("Estudios del padre", {k: t for k, t in ESTUDIOS.items() if k != 4}), 
    "Mother's occupation": ("Ocupación de la madre", {k: t for k, t in OCUPACION.items() if k not in (8, 10)}),  
    "Father's occupation": ("Ocupación del padre", OCUPACION),
    "Debtor": ("Tiene deudas con la institución", SI_NO),
    "Tuition fees up to date": ("Matrícula al día", SI_NO),
    "Scholarship holder": ("Becario", SI_NO),
    "Course": ("Programa académico", {
        171: "Diseño de animación y multimedia", 8014: "Servicio social (nocturno)", 9003: "Agronomía",
        9070: "Diseño de comunicación", 9085: "Enfermería veterinaria", 9119: "Ingeniería informática",
        9130: "Equinocultura", 9147: "Gestión", 9238: "Servicio social", 9254: "Turismo",
        9500: "Enfermería", 9556: "Higiene oral", 9670: "Publicidad y marketing",
        9773: "Periodismo y comunicación", 9853: "Educación básica", 9991: "Gestión (nocturno)",
        -1: "Otro programa (agrupado)"}),
    "Application mode": ("Modo de ingreso", {
        1: "1.ª fase · contingente general", 7: "Titular de otro curso superior",
        17: "2.ª fase · contingente general", 18: "3.ª fase · contingente general",
        39: "Mayor de 23 años", 42: "Transferencia", 43: "Cambio de curso",
        44: "Diploma de especialización tecnológica", 51: "Cambio de institución o curso",
        -1: "Otro (agrupado)"}),
    "Daytime/evening attendance": ("Jornada", {1: "Diurna", 0: "Nocturna"}),
    "Previous qualification": ("Estudios previos", {
        1: "Educación secundaria", 3: "Educación superior (grado)", 12: "Otro · 11.º año de escolaridad",
        19: "Educación básica 3.er ciclo (9.º–11.º) o equivalente",
        39: "Curso de especialización tecnológica", -1: "Otro (agrupado)"}),
}


COLUMNAS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
    "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Debtor", "Tuition fees up to date", "Gender",
    "Scholarship holder", "Age at enrollment",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
    "estado_1st", "estado_2nd", "es_adulto", "tasa_aprob_1st", "tasa_aprob_2nd",
]


#### Preprocesamiento realizado en el notebook Limpieza de datos.ipynb 
def agrupar_raras(df):
    """Los códigos que no están en la lista de opciones pasan a -1 ("Otro")."""
    df = df.copy()
    for col, (_, opciones) in CATEGORICAS.items():
        if col in df.columns and -1 in opciones:
            df[col] = df[col].where(df[col].isin(opciones), -1)
    return df


def agregar_derivadas(df):
    """Crea estado_1st/2nd, tasa_aprob_1st/2nd y es_adulto."""
    df = df.copy()
    for s in ("1st", "2nd"):
        p = f"Curricular units {s} sem ("
        inscritas, evals, aprobadas = df[p + "enrolled)"], df[p + "evaluations)"], df[p + "approved)"]
        df[f"estado_{s}"] = np.select(
            [inscritas == 0, (inscritas > 0) & (evals == 0), (evals > 0) & (aprobadas == 0)],
            ["no_inscrito", "inscrito_sin_eval", "evaluado_sin_aprobar"],
            default="normal",
        )
        
        df[f"tasa_aprob_{s}"] = np.where(inscritas > 0, aprobadas / inscritas.replace(0, 1), 0)
    df["es_adulto"] = (df["Age at enrollment"] >= 25).astype(int)
    return df


def validar(d):
    """d = {columna: valor}. Devuelve una lista de errores (vacía = todo bien)."""
    errores = []
    for s, nombre in (("1st", "Primer"), ("2nd", "Segundo")):
        p = f"Curricular units {s} sem ("
        if d[p + "approved)"] > d[p + "enrolled)"]:
            errores.append(f"{nombre} semestre: las unidades aprobadas no pueden superar las inscritas.")
        if (d[p + "approved)"] == 0) != (d[p + "grade)"] == 0):
            errores.append(f"{nombre} semestre: la nota es 0 solo si no aprobó ninguna unidad.")
    return errores