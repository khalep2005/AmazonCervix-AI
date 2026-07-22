# ==============================================================================
# AmazonCervix-AI — Módulo 5: Carga de Evaluaciones Citológicas desde Campo
# ==============================================================================

library(DT)   # carga aquí para garantizar disponibilidad en el módulo


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA — dataset de demostración (10 evaluaciones de campo)
# ─────────────────────────────────────────────────────────────────────────────

.CARGA_MOCK <- data.frame(
  ID_Muestra      = paste0("ACV-2024-", sprintf("%03d", 1:10)),
  Fecha_Toma      = as.character(seq(
    as.Date("2024-06-01"), by = "5 days", length.out = 10
  )),
  Centro_Salud    = sample(
    c("Hospital Regional Loreto", "Centro Médico Punchana",
      "Puesto de Salud Nanay",   "IPRESS Iquitos Norte",
      "Centro Médico Belén"),
    size = 10, replace = TRUE
  ),
  Edad_Paciente   = sample(22:58, size = 10, replace = TRUE),
  Clase_Bethesda  = sample(
    c("NILM", "ASC-US", "LSIL", "HSIL"),
    size = 10, replace = TRUE,
    prob = c(0.50, 0.20, 0.20, 0.10)
  ),
  Notas_Clinicas  = sample(
    c("Sin hallazgos relevantes.",
      "Paciente refiere sangrado intermenstrual.",
      "Historia de VPH positivo previo.",
      "Control post-tratamiento LEEP.",
      "Primera consulta citológica.",
      "Colposcopia pendiente.",
      "Requiere biopsia dirigida."),
    size = 10, replace = TRUE
  ),
  stringsAsFactors = FALSE
)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

mod_carga_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(

    # ── Panel lateral: controles de carga ─────────────────────────────────────
    sidebar = sidebar(
      width  = 310,
      title  = tagList(bsicons::bs_icon("cloud-upload-fill"),
                       " Carga desde Campo"),
      bg     = "#f8fafb",
      open   = TRUE,

      # ── Carga de archivo ──────────────────────────────────────────────────
      fileInput(
        inputId     = ns("archivo"),
        label       = tags$strong("Archivo de evaluaciones"),
        accept      = c(".csv", ".xlsx", ".xls",
                        "text/csv", "text/comma-separated-values",
                        "application/vnd.openxmlformats-officedocument"
                          |> paste0(".spreadsheetml.sheet")),
        multiple    = FALSE,
        buttonLabel = tagList(bsicons::bs_icon("folder2-open"), " Explorar"),
        placeholder = "Sin archivo seleccionado"
      ),

      tags$p(
        class = "text-muted small",
        bsicons::bs_icon("info-circle"),
        " Formatos aceptados: CSV (UTF-8) \u00b7 Excel (.xlsx / .xls).",
        tags$br(),
        "Columnas requeridas: ",
        tags$code("ID_Muestra, Fecha_Toma, Centro_Salud,"),
        tags$br(),
        tags$code("Edad_Paciente, Clase_Bethesda, Notas_Clinicas")
      ),

      hr(),

      # ── Validación de esquema ──────────────────────────────────────────────
      uiOutput(ns("validacion_ui")),

      hr(),

      # ── KPIs rápidos del archivo cargado ──────────────────────────────────
      uiOutput(ns("resumen_carga")),

      hr(),

      # ── Botón de sincronización ────────────────────────────────────────────
      div(
        class = "d-grid gap-2",

        actionButton(
          inputId = ns("btn_sync"),
          label   = tagList(
            bsicons::bs_icon("database-fill-up"), " Sincronizar con Data Lake"
          ),
          class = "btn btn-primary btn-lg",
          width = "100%"
        ),

        tags$p(
          class = "text-muted small text-center mt-1 mb-0",
          bsicons::bs_icon("layers-fill"),
          " Destino: Capa Bronze del pipeline ETL"
        )
      )
    ), # /sidebar

    # ── Área principal: vista previa y estado ─────────────────────────────────
    div(
      class = "container-fluid py-3",

      # ── Card principal: DataTable ────────────────────────────────────────
      card(
        full_screen = TRUE,
        card_header(
          class = "d-flex align-items-center justify-content-between",
          tags$span(
            class = "d-flex align-items-center gap-2",
            bsicons::bs_icon("table"),
            " Vista Previa de Evaluaciones"
          ),
          uiOutput(ns("badge_filas"))   # contador de filas en el header
        ),
        card_body(
          class = "p-0",
          DT::dataTableOutput(ns("tabla_datos"))
        )
      ),

      br(),

      # ── Card de instrucciones (se oculta tras cargar archivo) ─────────────
      uiOutput(ns("instrucciones_ui"))

    ) # /div main area
  ) # /layout_sidebar
}


# ─────────────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────────────

mod_carga_server <- function(id, api_url, api_online) {
  moduleServer(id, function(input, output, session) {

    # Columnas que debe tener un archivo válido
    COLS_REQUERIDAS <- c("ID_Muestra", "Fecha_Toma", "Centro_Salud",
                         "Edad_Paciente", "Clase_Bethesda", "Notas_Clinicas")

    # ── Reactive: datos activos (mock por defecto, archivo si se sube) ────────
    df_activo <- reactive({
      if (is.null(input$archivo)) {
        return(.CARGA_MOCK)   # datos de demostración
      }

      ext <- tools::file_ext(input$archivo$name)

      tryCatch({
        if (tolower(ext) == "csv") {
          read.csv(input$archivo$datapath,
                   encoding = "UTF-8", stringsAsFactors = FALSE)
        } else if (tolower(ext) %in% c("xlsx", "xls")) {
          # Requiere: install.packages("readxl")
          readxl::read_excel(input$archivo$datapath)
        } else {
          showNotification("Formato no soportado.", type = "error")
          .CARGA_MOCK
        }
      }, error = function(e) {
        showNotification(
          paste0("Error al leer el archivo: ", conditionMessage(e)),
          type = "error", duration = 8
        )
        .CARGA_MOCK
      })
    })

    # ── Reactive: validar columnas requeridas ─────────────────────────────────
    esquema_ok <- reactive({
      all(COLS_REQUERIDAS %in% names(df_activo()))
    })

    # ── UI: badge de validación de esquema ────────────────────────────────────
    output$validacion_ui <- renderUI({
      if (is.null(input$archivo)) return(NULL)

      if (esquema_ok()) {
        tags$div(
          class = "alert alert-success py-2 px-3 small mb-0",
          bsicons::bs_icon("check-circle-fill"),
          " Esquema v\u00e1lido \u2014 todas las columnas requeridas presentes."
        )
      } else {
        faltantes <- setdiff(COLS_REQUERIDAS, names(df_activo()))
        tags$div(
          class = "alert alert-danger py-2 px-3 small mb-0",
          bsicons::bs_icon("x-circle-fill"),
          tags$strong(" Columnas faltantes: "),
          paste(faltantes, collapse = ", ")
        )
      }
    })

    # ── UI: resumen rápido del dataset activo ─────────────────────────────────
    output$resumen_carga <- renderUI({
      df  <- df_activo()
      n   <- nrow(df)
      src <- if (is.null(input$archivo)) "Demo (mock data)" else input$archivo$name

      tags$div(
        class = "small",
        tags$p(
          bsicons::bs_icon("file-earmark-text"),
          tags$strong(" Fuente: "), src,
          class = "mb-1 text-muted"
        ),
        tags$p(
          bsicons::bs_icon("list-ol"),
          tags$strong(" Registros: "),
          tags$span(n, class = "text-primary fw-bold"),
          class = "mb-1"
        ),
        if ("Clase_Bethesda" %in% names(df)) {
          dist <- table(df$Clase_Bethesda)
          tags$p(
            bsicons::bs_icon("pie-chart-fill"),
            tags$strong(" Distribuci\u00f3n: "),
            paste(
              paste0(names(dist), " (", dist, ")"),
              collapse = " \u00b7 "
            ),
            class = "mb-0 text-muted"
          )
        }
      )
    })

    # ── UI: badge de conteo de filas en el card header ─────────────────────────
    output$badge_filas <- renderUI({
      n <- nrow(df_activo())
      tags$span(
        class = "badge bg-primary",
        paste0(n, " registros")
      )
    })

    # ── UI: instrucciones (solo cuando hay datos mock) ─────────────────────────
    output$instrucciones_ui <- renderUI({
      if (!is.null(input$archivo)) return(NULL)

      card(
        class = "border-dashed",
        card_body(
          class = "text-center py-4",
          tags$div(
            style = "color:#5B8DB8; font-size:2.5em;",
            bsicons::bs_icon("cloud-arrow-up")
          ),
          tags$h5("Sube tu archivo de campo", class = "mt-2 mb-1"),
          tags$p(
            class = "text-muted small",
            "Usa el panel lateral para cargar un CSV o Excel con las ",
            "evaluaciones citol\u00f3gicas recogidas en campo.",
            tags$br(),
            "La tabla de arriba muestra datos de demostración hasta que ",
            "cargues tu archivo real."
          )
        )
      )
    })

    # ── DataTable: vista previa interactiva ───────────────────────────────────
    output$tabla_datos <- DT::renderDataTable({
      df <- df_activo()

      DT::datatable(
        df,
        rownames  = FALSE,
        class     = "table table-hover table-sm",
        extensions = "Buttons",
        options   = list(
          pageLength   = 10,
          lengthMenu   = list(c(5, 10, 25, -1), c("5", "10", "25", "Todos")),
          scrollX      = TRUE,
          autoWidth    = TRUE,
          dom          = "Bfrtip",
          buttons      = list(
            list(extend = "csv",   text = "Exportar CSV"),
            list(extend = "excel", text = "Exportar Excel")
          ),
          language     = list(
            search      = "Buscar:",
            lengthMenu  = "Mostrar _MENU_ registros",
            info        = "Mostrando _START_\u2013_END_ de _TOTAL_ registros",
            paginate    = list(previous = "Anterior", `next` = "Siguiente"),
            zeroRecords = "No se encontraron registros.",
            emptyTable  = "Sin datos disponibles."
          ),
          columnDefs   = list(
            list(width = "120px", targets = 0),   # ID_Muestra
            list(width = "100px", targets = 1),   # Fecha_Toma
            list(className = "dt-center", targets = c(1, 3, 4))
          )
        )
      ) |>
        # Colorear la columna Clase_Bethesda por severidad
        DT::formatStyle(
          columns         = "Clase_Bethesda",
          backgroundColor = DT::styleEqual(
            c("NILM",    "ASC-US",  "ASC-H",   "LSIL",    "HSIL",    "SCC"),
            c("#D5EFE0", "#D6E8F5", "#FEF3CD", "#FFE4C4", "#FADBD8", "#EBDEF0")
          ),
          fontWeight = "bold"
        )
    }, server = TRUE)

    # ── Botón Sincronizar: simula ingesta a Capa Bronze ───────────────────────
    observeEvent(input$btn_sync, {

      df <- df_activo()

      # Verificar esquema antes de sincronizar
      if (!esquema_ok() && !is.null(input$archivo)) {
        showNotification(
          "\u274c El archivo no tiene el esquema v\u00e1lido. Corrígelo antes de sincronizar.",
          type     = "error",
          duration = 8
        )
        return()
      }

      # Mostrar notificación de "procesando"
      id_notif <- showNotification(
        tagList(
          tags$span(
            class = "spinner-border spinner-border-sm me-2",
            role  = "status"
          ),
          " Sincronizando con la Capa Bronze\u2026"
        ),
        type     = "message",
        duration = NULL,   # persiste hasta que la quitemos manualmente
        closeButton = FALSE
      )

      # Simular latencia de red / escritura al Data Lake
      Sys.sleep(1.5)

      # ── Llamada REAL a la API (descomentar cuando el endpoint exista) ───────
      # tryCatch({
      #   resp <- POST(
      #     url    = paste0(api_url, "/ingest/evaluaciones"),
      #     body   = list(data = toJSON(df, auto_unbox = TRUE)),
      #     encode = "json",
      #     timeout(seconds = 30)
      #   )
      #   if (http_status(resp)$category != "Success") stop(resp$status_code)
      # }, error = function(e) {
      #   removeNotification(id_notif)
      #   showNotification(
      #     paste0("\u274c Error al sincronizar: ", conditionMessage(e)),
      #     type = "error", duration = 8
      #   )
      #   return()
      # })

      # Quitar spinner y mostrar éxito
      removeNotification(id_notif)

      showNotification(
        tagList(
          bsicons::bs_icon("check-circle-fill"),
          tags$strong(
            paste0(" \u2705 ", nrow(df),
                   " evaluaciones sincronizadas correctamente con la Capa Bronze.")
          )
        ),
        type     = "message",
        duration = 6
      )
    })

  }) # /moduleServer
}
