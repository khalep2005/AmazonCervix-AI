# ==============================================================================
# AmazonCervix-AI — Módulo 3: Pre-Diagnóstico Automático (CNN / ONNX)
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES PRIVADAS DEL MÓDULO
# ─────────────────────────────────────────────────────────────────────────────

# Clases Bethesda en orden de severidad (deben coincidir con BETHESDA_CLASSES en main.py)
.PD_CLASES <- c("NILM", "ASC-US", "ASC-H", "LSIL", "HSIL", "SCC")

# Colores semánticos por clase
.PD_COLORES <- c(
  "NILM"   = "#2C7A4B",
  "ASC-US" = "#5B8DB8",
  "ASC-H"  = "#F39C12",
  "LSIL"   = "#E67E22",
  "HSIL"   = "#E74C3C",
  "SCC"    = "#8E1A0E"
)

# Ícono de alerta por clase (Bootstrap Icons)
.PD_ICONOS <- c(
  "NILM"   = "check-circle-fill",
  "ASC-US" = "exclamation-circle-fill",
  "ASC-H"  = "exclamation-triangle-fill",
  "LSIL"   = "exclamation-triangle-fill",
  "HSIL"   = "x-octagon-fill",
  "SCC"    = "x-octagon-fill"
)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

mod_prediagnostico_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(

    # ── Panel lateral: carga y controles ──────────────────────────────────────
    sidebar = sidebar(
      width  = 300,
      title  = tagList(bsicons::bs_icon("cpu-fill"), " Parámetros de Inferencia"),
      bg     = "#f8fafb",
      open   = TRUE,

      # Carga de imagen
      fileInput(
        inputId  = ns("imagen"),
        label    = tags$strong("Imagen de recorte celular"),
        accept   = c("image/png", "image/jpeg", "image/tiff"),
        multiple = FALSE,
        buttonLabel = tagList(bsicons::bs_icon("upload"), " Explorar"),
        placeholder = "Sin imagen seleccionada"
      ),

      # Nota informativa
      tags$p(
        class = "text-muted small",
        bsicons::bs_icon("info-circle"),
        " La imagen será redimensionada a 224\u00d7224 px en RGB (est\u00e1ndar Capa Gold)."
      ),

      hr(),

      # Botón de inferencia
      div(
        class = "d-grid",
        actionButton(
          inputId = ns("btn_inferencia"),
          label   = tagList(
            bsicons::bs_icon("play-circle-fill"), " Ejecutar Inferencia"
          ),
          class = "btn btn-success btn-lg",
          width = "100%"
        )
      ),

      hr(),

      # Estado de la API
      tags$p(tags$strong("Estado de la API:"), class = "mb-1 small"),
      uiOutput(ns("api_badge")),

      hr(),

      # Información técnica del modelo
      tags$div(
        class = "small text-muted",
        tags$p(tags$strong("Modelo:"), " AmazonCervix-CNN (ONNX)", class = "mb-1"),
        tags$p(tags$strong("Entrada:"), " 224\u00d7224 px · RGB · float32", class = "mb-1"),
        tags$p(tags$strong("Backend:"), " FastAPI + ONNX Runtime", class = "mb-1"),
        tags$p(tags$strong("Endpoint:"), " POST /predict", class = "mb-0")
      )
    ),

    # ── Área principal: resultados ────────────────────────────────────────────
    div(
      class = "container-fluid py-3",

      # ── Fila superior: preview + resultado principal ─────────────────────────
      layout_columns(
        col_widths = c(4, 8),

        # Preview de la imagen cargada
        card(
          full_screen = FALSE,
          card_header(
            class = "d-flex align-items-center gap-2",
            bsicons::bs_icon("image"),
            " Imagen cargada"
          ),
          card_body(
            class = "p-2 text-center",
            uiOutput(ns("preview_imagen"))
          )
        ),

        # Resultado principal: value_boxes de clasificación y confianza
        card(
          full_screen = FALSE,
          card_header(
            class = "d-flex align-items-center gap-2",
            bsicons::bs_icon("clipboard2-pulse"),
            " Resultado del Pre-Diagnóstico"
          ),
          card_body(
            # Spinner mientras se procesa
            uiOutput(ns("resultado_ui"))
          )
        )
      ),

      br(),

      # ── Fila inferior: desglose de probabilidades ────────────────────────────
      card(
        full_screen = TRUE,
        card_header(
          class = "d-flex align-items-center gap-2",
          bsicons::bs_icon("bar-chart-steps"),
          " Desglose de Probabilidades por Clase Bethesda"
        ),
        card_body(
          class = "p-2",
          uiOutput(ns("desglose_probabilidades"))
        )
      ),

      br(),

      # ── Advertencia clínica ──────────────────────────────────────────────────
      div(
        class = "alert alert-warning d-flex align-items-center gap-2",
        role  = "alert",
        bsicons::bs_icon("exclamation-triangle-fill", size = "1.2em"),
        tags$span(
          tags$strong("Advertencia cl\u00ednica: "),
          "Este resultado es orientativo y ",
          tags$strong("NO constituye un diagn\u00f3stico m\u00e9dico."),
          " Debe ser revisado e interpretado por un profesional de la salud cualificado."
        )
      )
    ) # /div main area

  ) # /layout_sidebar
}


# ─────────────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────────────

mod_prediagnostico_server <- function(id, api_url, api_online) {
  moduleServer(id, function(input, output, session) {

    # ── Badge de estado de la API en el sidebar ───────────────────────────────
    output$api_badge <- renderUI({
      if (api_online()) {
        tags$span(
          class = "badge bg-success",
          bsicons::bs_icon("circle-fill"), " En l\u00ednea"
        )
      } else {
        tags$span(
          class = "badge bg-danger",
          bsicons::bs_icon("circle-fill"), " Sin conexi\u00f3n"
        )
      }
    })

    # ── Preview de la imagen ──────────────────────────────────────────────────
    output$preview_imagen <- renderUI({
      if (is.null(input$imagen)) {
        tags$div(
          class = "text-muted py-5",
          bsicons::bs_icon("image", size = "3em"),
          tags$p("Sin imagen cargada", class = "mt-2 small")
        )
      } else {
        tags$img(
          src   = base64enc::dataURI(
            file = input$imagen$datapath,
            mime = input$imagen$type
          ),
          style = "max-width:100%; max-height:260px; border-radius:8px;
                   object-fit:contain; border:1px solid #dee2e6;"
        )
      }
    })

    # ── Resultado reactivo (se dispara con el botón) ──────────────────────────
    # Usamos eventReactive para ejecutar la inferencia solo al hacer clic.
    resultado <- eventReactive(input$btn_inferencia, {

      req(input$imagen)   # exige imagen antes de llamar a la API

      # ── Intento de llamada REAL a la API FastAPI ────────────────────────────
      resultado_api <- tryCatch(
        expr = {
          resp <- POST(
            url  = paste0(api_url, "/predict"),
            body = list(
              file = upload_file(
                path     = input$imagen$datapath,
                type     = input$imagen$type
              )
            ),
            encode  = "multipart",
            timeout(seconds = 15)
          )

          if (http_status(resp)$category == "Success") {
            # Parsear el JSON de respuesta de la API
            body <- fromJSON(rawToChar(resp$content))
            list(
              origen          = "api",
              clase           = body$clase_predicha,
              confianza       = body$confianza,
              probabilidades  = body$probabilidades,
              latencia_ms     = body$latencia_ms
            )
          } else {
            NULL   # respuesta no exitosa → cae al mock
          }
        },
        error = function(e) {
          message("[mod_prediagnostico] API no disponible: ", conditionMessage(e))
          NULL   # cualquier error de red → cae al mock
        }
      )

      # ── MOCK: respuesta simulada cuando la API no está disponible ───────────
      # Genera probabilidades aleatorias realistas para probar la UI.
      if (is.null(resultado_api)) {
        set.seed(NULL)   # semilla variable para aleatoriedad real

        # Simular logits y aplicar softmax manual
        logits <- rnorm(6, mean = 0, sd = 1.5)
        probs  <- exp(logits) / sum(exp(logits))
        names(probs) <- .PD_CLASES

        idx_max    <- which.max(probs)
        clase_pred <- .PD_CLASES[idx_max]
        confianza  <- round(probs[idx_max] * 100, 2)

        prob_list  <- as.list(round(probs * 100, 2))

        list(
          origen         = "mock",
          clase          = clase_pred,
          confianza      = confianza,
          probabilidades = prob_list,
          latencia_ms    = round(runif(1, 8, 45), 1)
        )
      } else {
        resultado_api
      }
    })

    # ── Render del resultado principal (value_boxes) ──────────────────────────
    output$resultado_ui <- renderUI({

      # Estado vacío antes de la primera inferencia
      if (is.null(resultado())) {
        return(
          tags$div(
            class = "text-center text-muted py-5",
            bsicons::bs_icon("arrow-left-circle", size = "2em"),
            tags$p("Carga una imagen y pulsa \u201cEjecutar Inferencia\u201d",
                   class = "mt-2")
          )
        )
      }

      res   <- resultado()
      clase <- res$clase
      color <- .PD_COLORES[[clase]]
      icono <- .PD_ICONOS[[clase]]

      tagList(

        # Indicador de origen (API real vs mock)
        if (res$origen == "mock") {
          div(
            class = "alert alert-info py-1 px-2 mb-3 small",
            bsicons::bs_icon("info-circle"),
            " Resultado simulado (API offline). Los datos son de demostración."
          )
        },

        # KPIs principales: Clasificación + Confianza
        layout_column_wrap(
          width = 1/2,
          fill  = FALSE,

          # KPI 1 — Clase Bethesda predicha
          value_box(
            title    = "Clasificaci\u00f3n Bethesda",
            value    = tags$span(clase, style = "font-size:1.8em; font-weight:700;"),
            showcase = bsicons::bs_icon(icono),
            theme    = value_box_theme(bg = color, fg = "#ffffff"),
            p(switch(clase,
                "NILM"   = "Negativo para lesi\u00f3n intraepitelial o malignidad",
                "ASC-US" = "C\u00e9lulas escamosas at\u00edpicas de significado indeterminado",
                "ASC-H"  = "At\u00edpicas: no se puede excluir HSIL",
                "LSIL"   = "Lesi\u00f3n intraepitelial escamosa de bajo grado",
                "HSIL"   = "Lesi\u00f3n intraepitelial escamosa de alto grado",
                "SCC"    = "Carcinoma de c\u00e9lulas escamosas"
              ),
              class = "mb-0 small opacity-90"
            )
          ),

          # KPI 2 — Nivel de confianza
          value_box(
            title    = "Nivel de Confianza",
            value    = tags$span(
              paste0(res$confianza, "%"),
              style = "font-size:1.8em; font-weight:700;"
            ),
            showcase = bsicons::bs_icon("speedometer2"),
            theme    = value_box_theme(
              bg = if (res$confianza >= 85) "#27AE60" else "#F39C12",
              fg = "#ffffff"
            ),
            p(paste0("Latencia: ", res$latencia_ms, " ms"),
              class = "mb-0 small opacity-90")
          )
        )
      )
    })

    # ── Render del desglose de probabilidades ─────────────────────────────────
    output$desglose_probabilidades <- renderUI({

      if (is.null(resultado())) {
        return(
          tags$p(class = "text-muted small py-2",
                 "El desglose aparecerá tras ejecutar la inferencia.")
        )
      }

      res   <- resultado()
      probs <- res$probabilidades

      # Ordenar de mayor a menor probabilidad
      probs_ord <- sort(unlist(probs), decreasing = TRUE)

      tags$div(
        class = "px-2",
        lapply(names(probs_ord), function(cls) {
          pct   <- round(probs_ord[[cls]], 1)
          color <- .PD_COLORES[[cls]]

          tags$div(
            class = "mb-3",
            # Etiqueta + valor
            tags$div(
              class = "d-flex justify-content-between mb-1",
              tags$span(
                tags$span(style = paste0("color:", color, "; font-size:1.1em;"),
                          "\u25cf "),
                tags$strong(cls),
                class = "small"
              ),
              tags$span(paste0(pct, "%"),
                        class = "small fw-bold",
                        style = paste0("color:", color))
            ),
            # Barra de progreso Bootstrap
            tags$div(
              class = "progress",
              style = "height: 8px; border-radius: 4px;",
              tags$div(
                class = "progress-bar",
                role  = "progressbar",
                style = paste0(
                  "width:", pct, "%;",
                  "background-color:", color, ";",
                  "border-radius: 4px;"
                ),
                `aria-valuenow` = pct,
                `aria-valuemin` = "0",
                `aria-valuemax` = "100"
              )
            )
          )
        })
      )
    })

  }) # /moduleServer
}
