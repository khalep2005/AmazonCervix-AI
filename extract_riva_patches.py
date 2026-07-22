import os
import pandas as pd
import cv2

# ==============================================================================
# PIPELINE DE INGENIERÍA DE DATOS: EXTRACCIÓN DE PARCHES BIOMÉDICOS (DATASET RIVA)
# ==============================================================================

# Configuración de rutas institucionales
BASE_DIR = "GOLD"
METADATA_PATH = os.path.join(BASE_DIR, "_documentation", "riva_annotations.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "classification", "riva", "gold_cellular_patches")

# Parámetros técnicos de normalización de dimensiones (Matriz de 256x256 píxeles)
PATCH_SIZE = 256
HALF_SIZE = PATCH_SIZE // 2

# Inicialización del entorno de salida en la Capa Gold
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Cargando registro maestro de anotaciones citológicas (riva_annotations.csv)...")
df = pd.read_csv(METADATA_PATH)

# Selección de subconjunto de control (20 instancias de prueba) para validación rápida
df_subset = df.head(20)

print("[INFO] Iniciando el procesamiento y recorte de regiones de interés celular (ROI)...")

for index, row in df_subset.iterrows():
    img_id = row['image_id']
    split = row['split']            # Partición: train, val, test
    img_class = row['image_class']  # Categoría Bethesda del parche global
    
    # Construcción de la ruta dinámica hacia el repositorio base de imágenes
    img_name = f"{img_id}.png"
    img_path = os.path.join(BASE_DIR, "classification", "riva", split, img_class, img_name)
    
    if not os.path.exists(img_path):
        print(f"[ERROR] Instancia no localizada en el Data Lake: {img_path}")
        continue
        
    # Carga de la imagen citológica en la matriz bidimensional
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # Extracción de coordenadas espaciales validadas por los especialistas
    cx, cy = int(row['x_pixel']), int(row['y_pixel'])
    cell_label = row['keypoint_label']  # Diagnóstico específico de la célula (Bethesda)
    
    # Cálculo matemático de límites del Bounding Box perimetral
    x1 = max(0, cx - HALF_SIZE)
    y1 = max(0, cy - HALF_SIZE)
    x2 = min(w, cx + HALF_SIZE)
    y2 = min(h, cy + HALF_SIZE)
    
    # Extracción geométrica de la región celular
    cell_patch = img[y1:y2, x1:x2]
    
    # Estructuración jerárquica por taxonomía clínica (NILM, ASCH, LSIL, HSIL)
    label_dir = os.path.join(OUTPUT_DIR, cell_label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Persistencia del parche en el sistema de archivos distribuido
    output_filename = f"patch_{row['annotation_id']}.png"
    output_path = os.path.join(label_dir, output_filename)
    
    cv2.imwrite(output_path, cell_patch)
    print(f"[SUCCESS] ROI procesada e indexada: {output_filename} -> Categoría: {cell_label}")

print(f"\n[INFO] Pipeline ejecutado con éxito. Repositorio Gold generado en: {OUTPUT_DIR}")