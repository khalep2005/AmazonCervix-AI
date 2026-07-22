"""
==============================================================================
AmazonCervix-AI — API de Inferencia (Capa Gold)
==============================================================================
Autor      : Equipo MLOps / AmazonCervix-AI
Versión    : 1.0.0
Stack      : FastAPI + ONNX Runtime + Pillow
Descripción: Servidor REST local para clasificación de lesiones cervicouterinas
             según el Sistema Bethesda. Recibe una imagen por HTTP, la
             preprocesa a 224×224 px en formato RGB y devuelve la clase
             Bethesda predicha junto con el porcentaje de confianza.

Requisitos (instalar con pip):
    pip install fastapi uvicorn onnxruntime pillow numpy

Ejecución:
    uvicorn main:app --reload

Endpoints:
    GET  /          → Bienvenida y estado del servicio
    POST /predict   → Inferencia sobre una imagen enviada como form-data
==============================================================================
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

# Ruta al modelo ONNX
MODEL_PATH: Path = Path(__file__).parent / "models" / "resnet50_model.onnx"

# Dimensiones de entrada
IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224

# Clases Bethesda soportadas
BETHESDA_CLASSES: list[str] = [
    "NILM",   # Negativo para lesión intraepitelial o malignidad
    "ASC-US", # Células escamosas atípicas de significado indeterminado
    "ASC-H",  # Células escamosas atípicas, no se puede excluir HSIL
    "LSIL",   # Lesión intraepitelial escamosa de bajo grado
    "HSIL",   # Lesión intraepitelial escamosa de alto grado
    "SCC",    # Carcinoma de células escamosas
]

# Media y desviación estándar de ImageNet (estándar para modelos pre-entrenados
# con ResNet18, EfficientNet-B0 o MobileNetV3 como recomiendan los notebooks Gold)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float]  = [0.229, 0.224, 0.225]

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("amazoncervix_api")


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO GLOBAL DE LA SESIÓN ONNX
# Dict mutable para poder actualizarlo dentro del lifespan sin variables globales.
# ─────────────────────────────────────────────────────────────────────────────

app_state: dict = {
    "ort_session": None,    # Sesión de inferencia ONNX Runtime
    "input_name": None,     # Nombre del tensor de entrada del modelo
    "model_loaded": False,  # Bandera de disponibilidad del modelo
    "startup_time": None,   # Timestamp de arranque del servidor
}


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — Carga del modelo al iniciar / liberación al apagar el servidor
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Contexto de vida de la aplicación.

    - Al INICIAR: carga el modelo ONNX en memoria (una sola vez, modo Edge AI).
    - Al CERRAR:  libera la sesión de ONNX Runtime.

    Si el archivo model.onnx no existe aún, el servidor arranca en
    modo stub: /predict devuelve HTTP 503 hasta que se coloque el modelo.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    app_state["startup_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    logger.info("Iniciando AmazonCervix-AI API ...")

    if MODEL_PATH.exists():
        try:
            # Proveedores en orden de preferencia:
            # CUDAExecutionProvider → GPU (si disponible)
            # CPUExecutionProvider  → CPU (siempre disponible, modo Edge AI)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            session_options = ort.SessionOptions()
            # Activa todas las optimizaciones de grafo disponibles en ONNX Runtime
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            app_state["ort_session"] = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=session_options,
                providers=providers,
            )
            # Obtener el nombre del tensor de entrada desde los metadatos del modelo
            app_state["input_name"] = app_state["ort_session"].get_inputs()[0].name
            app_state["model_loaded"] = True

            logger.info(
                "Modelo cargado exitosamente desde '%s'. "
                "Tensor de entrada: '%s'.",
                MODEL_PATH,
                app_state["input_name"],
            )
        except Exception as exc:
            logger.error("Error al cargar el modelo ONNX: %s", exc)
    else:
        logger.warning(
            "Archivo '%s' no encontrado. "
            "El servidor arranca en modo stub. "
            "Coloca model.onnx en '%s' y reinicia el servidor.",
            MODEL_PATH.name,
            MODEL_PATH.parent,
        )

    yield  # ── El servidor está activo durante este bloque ──────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    if app_state["ort_session"] is not None:
        del app_state["ort_session"]
        logger.info("Sesión ONNX Runtime liberada correctamente.")
    logger.info("AmazonCervix-AI API apagada.")


# ─────────────────────────────────────────────────────────────────────────────
# INSTANCIA FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AmazonCervix-AI — Clasificación Bethesda",
    description=(
        "API de inferencia local para clasificación de lesiones cervicouterinas "
        "según el Sistema Bethesda. Utiliza un modelo exportado a ONNX para "
        "garantizar inferencia rápida en entornos Edge (Edge AI)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",   # Swagger UI interactiva
    redoc_url="/redoc", # ReDoc
)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: PREPROCESAMIENTO DE IMAGEN
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convierte los bytes raw de una imagen en un tensor NCHW listo para ONNX.

    Pipeline de preprocesamiento (Capa Gold — estándar del proyecto):
        1. Decodificación desde bytes (Pillow).
        2. Conversión a RGB: elimina canal alpha y garantiza exactamente 3 canales.
        3. Redimensionado a 224×224 px con LANCZOS (anti-aliasing de alta calidad).
        4. Normalización por canal:  pixel = (pixel / 255 − mean) / std
           donde mean y std son los valores de ImageNet.
        5. Transposición HWC → CHW y adición de dimensión de batch → NCHW.

    Args:
        image_bytes: Contenido binario de la imagen recibida por HTTP.

    Returns:
        Tensor numpy de shape (1, 3, 224, 224) y dtype float32.

    Raises:
        ValueError: Si los bytes no corresponden a una imagen válida.
    """
    # ── Decodificación ────────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"No se pudo decodificar la imagen: {exc}") from exc

    # ── Redimensionado a 224×224 (Capa Gold) ─────────────────────────────────
    image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)

    # ── Normalización ─────────────────────────────────────────────────────────
    # Pasar a float32 y escalar de [0, 255] a [0, 1]
    img_array: np.ndarray = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)

    # Aplicar media y desviación estándar de ImageNet canal a canal
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)  # shape (3,)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)  # shape (3,)
    img_array = (img_array - mean) / std               # broadcasting → (H, W, 3)

    # ── Formato NCHW para ONNX Runtime ───────────────────────────────────────
    img_array = img_array.transpose(2, 0, 1)    # HWC → CHW: (3, 224, 224)
    img_array = np.expand_dims(img_array, 0)    # CHW → NCHW: (1, 3, 224, 224)

    return img_array


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: SOFTMAX ESTABLE
# ─────────────────────────────────────────────────────────────────────────────

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Aplica softmax numéricamente estable sobre el vector de logits.

    Usa el truco de restar el máximo para evitar overflow en exponenciales.

    Args:
        logits: Array 1D de logits crudos devueltos por el modelo.

    Returns:
        Array 1D de probabilidades que suman exactamente 1.0.
    """
    shifted   = logits - np.max(logits)  # estabilidad numérica
    exp_vals  = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: GET / — Bienvenida y estado del servicio
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/",
    summary="Estado del servicio",
    response_description="Información de bienvenida y estado del modelo.",
    tags=["Health"],
)
async def root() -> JSONResponse:
    """
    Ruta principal de bienvenida.

    Retorna el nombre del servicio, versión, estado del modelo ONNX,
    clases Bethesda disponibles y el timestamp de arranque del servidor.
    """
    return JSONResponse(
        content={
            "servicio": "AmazonCervix-AI — API de Clasificación Bethesda",
            "version": "1.0.0",
            "descripcion": (
                "Clasificación de lesiones cervicouterinas según el Sistema "
                "Bethesda mediante Edge AI con ONNX Runtime."
            ),
            "modelo_cargado": app_state["model_loaded"],
            "clases_soportadas": BETHESDA_CLASSES,
            "startup_time": app_state["startup_time"],
            "docs_url": "/docs",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /predict — Inferencia principal
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    summary="Clasificar lesión cervicouterina",
    response_description="Clase Bethesda predicha con porcentaje de confianza.",
    tags=["Inferencia"],
)
async def predict(
    file: UploadFile = File(
        ...,
        description=(
            "Imagen de célula cervical en formato JPEG, PNG o TIFF. "
            "Se redimensionará internamente a 224×224 px en RGB."
        ),
    ),
) -> JSONResponse:
    """
    Endpoint de inferencia principal.

    Recibe una imagen de célula cervical, aplica el pipeline de
    preprocesamiento de la Capa Gold (224×224 px, RGB, normalización
    ImageNet) y ejecuta la inferencia con el modelo ONNX cargado.

    **Clases Bethesda soportadas:**
    - `NILM`   — Negativo para lesión intraepitelial o malignidad
    - `ASC-US` — Células escamosas atípicas de significado indeterminado
    - `ASC-H`  — Células escamosas atípicas (no excluye HSIL)
    - `LSIL`   — Lesión intraepitelial escamosa de bajo grado
    - `HSIL`   — Lesión intraepitelial escamosa de alto grado
    - `SCC`    — Carcinoma de células escamosas

    **Formato de respuesta:**
    ```json
    {
        "clase_predicha": "LSIL",
        "confianza": 94.37,
        "probabilidades": { "NILM": 1.20, "ASC-US": 0.85, ... },
        "latencia_ms": 12.4,
        "imagen_recibida": "celula_001.jpg",
        "modelo": "model.onnx",
        "advertencia_clinica": "..."
    }
    ```
    """
    # ── 1. Verificar disponibilidad del modelo ────────────────────────────────
    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"El modelo ONNX no está cargado. "
                f"Coloca '{MODEL_PATH.name}' en '{MODEL_PATH.parent}' "
                f"y reinicia el servidor con: uvicorn main:app --reload"
            ),
        )

    # ── 2. Validar el tipo MIME del archivo recibido ──────────────────────────
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Tipo de archivo no soportado: '{file.content_type}'. "
                f"Tipos aceptados: {sorted(allowed_types)}"
            ),
        )

    # ── 3. Leer los bytes de la imagen ───────────────────────────────────────
    image_bytes: bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo recibido está vacío.",
        )

    # ── 4. Preprocesar la imagen → tensor NCHW (1, 3, 224, 224) ─────────────
    try:
        input_tensor: np.ndarray = preprocess_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # ── 5. Ejecutar inferencia ONNX Runtime ───────────────────────────────────
    try:
        t_start = time.perf_counter()
        ort_outputs = app_state["ort_session"].run(
            None,  # None = obtener todos los tensores de salida del modelo
            {app_state["input_name"]: input_tensor},
        )
        latency_ms: float = (time.perf_counter() - t_start) * 1_000
    except Exception as exc:
        logger.error("Error durante la inferencia ONNX: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante la inferencia del modelo: {exc}",
        ) from exc

    # ── 6. Postprocesar la salida del modelo ──────────────────────────────────
    # ort_outputs[0] → shape (1, N_CLASES) con logits crudos o probabilidades
    raw_output: np.ndarray = ort_outputs[0][0]     # shape (N_CLASES,)

    # Aplicar softmax para obtener probabilidades (0–1) desde logits crudos
    probabilities: np.ndarray = softmax(raw_output)

    # Clase con mayor probabilidad (argmax)
    predicted_idx: int   = int(np.argmax(probabilities))
    predicted_class: str = BETHESDA_CLASSES[predicted_idx]
    confidence_pct: float = round(float(probabilities[predicted_idx]) * 100, 2)

    # Diccionario completo: todas las clases con su probabilidad en porcentaje
    prob_dict: dict[str, float] = {
        cls: round(float(prob) * 100, 2)
        for cls, prob in zip(BETHESDA_CLASSES, probabilities)
    }

    logger.info(
        "Predicción: clase='%s' | confianza=%.2f%% | latencia=%.1f ms | archivo='%s'",
        predicted_class,
        confidence_pct,
        latency_ms,
        file.filename,
    )

    # ── 7. Construir y retornar la respuesta JSON ─────────────────────────────
    return JSONResponse(
        content={
            "clase_predicha": predicted_class,
            "confianza": confidence_pct,
            "probabilidades": prob_dict,
            "latencia_ms": round(latency_ms, 1),
            "imagen_recibida": file.filename,
            "modelo": MODEL_PATH.name,
            "advertencia_clinica": (
                "Este resultado es orientativo y NO constituye un diagnóstico médico. "
                "Debe ser revisado e interpretado por un profesional de la salud cualificado."
            ),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA ALTERNATIVO (sin CLI de uvicorn)
# Útil para debug con: python main.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
