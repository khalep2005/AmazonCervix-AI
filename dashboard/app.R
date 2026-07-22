# ==============================================================================
# AmazonCervix-AI — Dashboard Interactivo
# ==============================================================================
#
# Estructura:
#   app.R                        ← este archivo (orquestador)
#   modules/mod_vigilancia.R     ← Módulo 1: KPIs + gráficos ggplot2
#   modules/mod_mapa.R           ← Módulo 2: mapa de calor leaflet
#   modules/mod_prediagnostico.R ← Módulo 3: Pre-Diagnóstico CNN / ONNX
#   modules/mod_desempeno.R      ← Módulo 4: Análisis de Desempeño (AUC-ROC, CM)
#   modules/mod_carga.R          ← Módulo 5: Carga de Evaluaciones desde Campo
#
# Ejecución:
#   shiny::runApp("dashboard/app.R")   — desde la raíz del proyecto
#   — ó —  abrir app.R en RStudio y clic en "Run App"
#
# Dependencias (instalar una sola vez):
#   install.packages(c("shiny","bslib","httr","jsonlite",
#                      "ggplot2","scales","bsicons",
#                      "leaflet","leaflet.extras","magrittr",
#                      "base64enc","DT","readxl"))
# ==============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 0. LIBRERÍAS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

library(shiny)
library(bslib)
library(httr)
library(jsonlite)
library(ggplot2)
library(leaflet)
library(leaflet.extras)
library(magrittr)   # operador %>% para leaflet
library(base64enc)  # codificación imagen → URI data:// para preview en UI
library(DT)         # tablas interactivas (módulo Carga)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE MÓDULOS
# ─────────────────────────────────────────────────────────────────────────────

source("modules/mod_vigilancia.R")
source("modules/mod_mapa.R")
source("modules/mod_prediagnostico.R")
source("modules/mod_desempeno.R")
source("modules/mod_carga.R")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

API_URL     <- "http://127.0.0.1:8000"
APP_TITLE   <- "AmazonCervix-AI"
APP_VERSION <- "v5.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMA VISUAL (bslib)
# ─────────────────────────────────────────────────────────────────────────────

cervix_theme <- bs_theme(
  bootswatch   = "flatly",
  primary      = "#2C7A4B",   # verde selva / Amazon
  secondary    = "#5B8DB8",   # azul médico
  success      = "#27AE60",
  warning      = "#F39C12",
  danger       = "#E74C3C",
  base_font    = font_google("Inter"),
  heading_font = font_google("Inter", wght = 700),
  font_scale   = 0.95
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. INTERFAZ DE USUARIO (UI) — orquestadora
# ─────────────────────────────────────────────────────────────────────────────

ui <- page_navbar(

  title = tags$span(
    tags$strong(APP_TITLE),
    tags$small(APP_VERSION,
               style = "font-size:0.7em; color:#aec6cf; margin-left:6px;")
  ),
  id      = "main_navbar",
  theme   = cervix_theme,
  bg      = "#2C7A4B",
  inverse = TRUE,

  # ── Badge de estado de la API ──────────────────────────────────────────────
  nav_spacer(),
  nav_item(
    tags$span(
      id    = "api_status_badge",
      class = "badge bg-secondary",
      style = "font-size:0.75em; cursor:default;",
      uiOutput("api_status_ui")
    )
  ),

  # ── Pestaña 1: Vigilancia Epidemiológica ───────────────────────────────────
  nav_panel(
    title = tagList(icon("chart-line"), " Vigilancia Epidemiológica"),
    value = "tab_vigilancia",
    div(class = "container-fluid pt-3",
        h3("Vigilancia Epidemiológica", class = "text-primary mb-1"),
        p("Distribución y tendencias de clasificaciones Bethesda en el tiempo.",
          class = "text-muted"),
        hr()),
    mod_vigilancia_ui("mod_vigilancia")    # ← UI del módulo 1
  ),

  # ── Pestaña 2: Mapa de Calor ───────────────────────────────────────────────
  nav_panel(
    title = tagList(icon("fire"), " Mapa de Calor"),
    value = "tab_mapa_calor",
    div(class = "container-fluid pt-3",
        h3("Mapa de Calor", class = "text-primary mb-1"),
        p("Distribución geográfica de lesiones por centro de atención.",
          class = "text-muted"),
        hr()),
    mod_mapa_ui("mod_mapa")               # ← UI del módulo 2
  ),

  # ── Pestaña 3: Pre-Diagnóstico Automático (CNN) ───────────────────────────
  nav_panel(
    title = tagList(icon("brain"), " Inferencia de IA (Capa Gold)"),
    value = "tab_inferencia",
    div(class = "container-fluid pt-3",
        h3("Pre-Diagnóstico Automático — CNN / ONNX",
           class = "text-primary mb-1"),
        p("Clasificaci\u00f3n en tiempo real de c\u00e9lulas cervicales mediante el modelo ONNX en FastAPI.",
          class = "text-muted"),
        hr()),
    mod_prediagnostico_ui("mod_prediagnostico")   # ← UI del módulo 3
  ),

  # ── Pestaña 4: Análisis de Desempeño del Modelo ───────────────────────────
  nav_panel(
    title = tagList(icon("chart-bar"), " Desempeño del Modelo"),
    value = "tab_desempeno",
    div(class = "container-fluid pt-3",
        h3("Análisis de Desempeño del Modelo — Capa Gold",
           class = "text-primary mb-1"),
        p("Matriz de Confusión, Curvas AUC-ROC y métricas globales por clase Bethesda.",
          class = "text-muted"),
        hr()),
    mod_desempeno_ui("mod_desempeno")    # ← UI del módulo 4
  ),

  # ── Pestaña 5: Carga de Evaluaciones Citológicas ───────────────────────────
  nav_panel(
    title = tagList(icon("file-csv"), " Carga de Evaluaciones"),
    value = "tab_carga",
    div(class = "container-fluid pt-3",
        h3("Carga de Evaluaciones Citológicas desde Campo",
           class = "text-primary mb-1"),
        p("Importación, validación y sincronización con la Capa Bronze del Data Lake.",
          class = "text-muted"),
        hr()),
    mod_carga_ui("mod_carga")            # ← UI del módulo 5
  )

) # /page_navbar


# ─────────────────────────────────────────────────────────────────────────────
# 5. SERVIDOR (SERVER) — orquestador
# ─────────────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  # ── 5.1 URL base de la API ─────────────────────────────────────────────────
  # Leer desde variable de entorno; si no existe, usar la constante global.
  api_url <- Sys.getenv("API_URL", unset = API_URL)

  # ── 5.2 Health-check al iniciar la sesión ──────────────────────────────────
  api_online <- reactiveVal(FALSE)

  tryCatch(
    expr = {
      resp <- GET(api_url, timeout(seconds = 5))
      if (http_status(resp)$category == "Success") {
        body <- fromJSON(rawToChar(resp$content))
        api_online(TRUE)
        message("[AmazonCervix-AI] \u2705 API en línea — ", api_url,
                " | Modelo cargado: ", body$modelo_cargado)
      } else {
        warning("[AmazonCervix-AI] \u26a0\ufe0f  API respondió: ",
                http_status(resp)$message)
      }
    },
    error = function(e) {
      warning("[AmazonCervix-AI] \u274c Sin conexión a '", api_url,
              "'\n  Detalle: ", conditionMessage(e),
              "\n  Ejecuta: uvicorn main:app --reload (en /api)")
    }
  )

  # ── 5.3 Badge reactivo en la navbar ────────────────────────────────────────
  output$api_status_ui <- renderUI({
    if (api_online()) {
      tags$span(icon("circle-check"), " API Online",
                style = "color:#27AE60; font-weight:600;")
    } else {
      tags$span(icon("circle-xmark"), " API Offline",
                style = "color:#E74C3C; font-weight:600;")
    }
  })

  # ── 5.4 Llamadas a los servidores de módulos ────────────────────────────────
  mod_vigilancia_server("mod_vigilancia",
                        api_url    = api_url,
                        api_online = api_online)

  mod_mapa_server("mod_mapa",
                  api_url    = api_url,
                  api_online = api_online)

  mod_prediagnostico_server("mod_prediagnostico",
                            api_url    = api_url,
                            api_online = api_online)

  mod_desempeno_server("mod_desempeno",
                       api_url    = api_url,
                       api_online = api_online)

  mod_carga_server("mod_carga",
                   api_url    = api_url,
                   api_online = api_online)

  # >> mod_inferencia_server("mod_inferencia", api_url, api_online)

} # /server


# ─────────────────────────────────────────────────────────────────────────────
# 6. PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

shinyApp(ui = ui, server = server)
