# ==============================================================================
# AmazonCervix-AI — Módulo 2: Mapa de Calor Epidemiológico
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES — compartida dentro del módulo
# ─────────────────────────────────────────────────────────────────────────────

.COLORES_BETHESDA_MAPA <- c(
  "NILM"   = "#2C7A4B",
  "ASC-US" = "#5B8DB8",
  "ASC-H"  = "#F39C12",
  "LSIL"   = "#E67E22",
  "HSIL"   = "#E74C3C",
  "SCC"    = "#8E1A0E"
)

# Clases disponibles en el selector (orden de menor a mayor riesgo)
.CLASES_BETHESDA <- names(.COLORES_BETHESDA_MAPA)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA — generado una sola vez al cargar el módulo
# 150 puntos en la región Loreto / Iquitos (selva amazónica peruana)
# lat ∈ [-5.20, -3.10]   lng ∈ [-74.80, -72.20]
# ─────────────────────────────────────────────────────────────────────────────

set.seed(42)
.N <- 150L

.DF_GEO <- data.frame(
  lat = runif(.N, min = -5.20, max = -3.10),
  lng = runif(.N, min = -74.80, max = -72.20),

  # Distribución de clases refleja prevalencia real de Bethesda
  clase = sample(
    x       = .CLASES_BETHESDA,
    size    = .N,
    replace = TRUE,
    prob    = c(0.52, 0.14, 0.06, 0.17, 0.08, 0.03)
  ),

  # Intensidad del heatmap: valores altos = lesiones de alto grado
  # Escala 0-1 normalizada para addHeatmap()
  intensidad = sample(
    x       = c(0.10, 0.25, 0.40, 0.60, 0.80, 1.00),
    size    = .N,
    replace = TRUE,
    prob    = c(0.52, 0.14, 0.06, 0.17, 0.08, 0.03)
  ),

  centro = sample(
    x = c("Hospital Regional Loreto", "Centro Médico Punchana",
           "Puesto de Salud Nanay",   "IPRESS Iquitos Norte",
           "Centro Médico Belén",     "Puesto Salud San Juan"),
    size    = .N,
    replace = TRUE
  ),

  stringsAsFactors = FALSE
)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

mod_mapa_ui <- function(id) {
  ns <- NS(id)

  layout_sidebar(

    # ── Panel lateral: filtros ───────────────────────────────────────────────
    sidebar = sidebar(
      width = 270,
      title = tagList(bsicons::bs_icon("funnel-fill"), " Filtros"),
      bg    = "#f8fafb",
      open  = TRUE,

      # Selector de categorías Bethesda
      checkboxGroupInput(
        inputId  = ns("bethesda_filter"),
        label    = tags$strong("Categoría Bethesda"),
        choices  = .CLASES_BETHESDA,
        selected = c("ASC-US", "ASC-H", "LSIL", "HSIL", "SCC"),  # positivos
        width    = "100%"
      ),

      hr(),

      # Capa heatmap
      checkboxInput(
        inputId = ns("mostrar_heatmap"),
        label   = tags$span(
          bsicons::bs_icon("thermometer-half"), " Mostrar capa de calor"
        ),
        value = TRUE
      ),

      # Capa de marcadores individuales
      checkboxInput(
        inputId = ns("mostrar_markers"),
        label   = tags$span(
          bsicons::bs_icon("geo-alt-fill"), " Mostrar marcadores"
        ),
        value = FALSE
      ),

      hr(),

      # Leyenda de colores
      tags$p(tags$strong("Referencia de color:"),
             class = "mb-1 small text-muted"),
      tags$div(
        class = "d-flex flex-column gap-1 small",
        lapply(.CLASES_BETHESDA, function(cls) {
          tags$span(
            tags$span(
              style = paste0("color:", .COLORES_BETHESDA_MAPA[[cls]],
                             "; font-size:1.1em;"),
              "\u25cf"   # ●
            ),
            paste("", cls)
          )
        })
      ),

      hr(),

      # Resumen reactivo de casos
      uiOutput(ns("resumen_casos"))
    ),

    # ── Área principal: mapa leaflet ─────────────────────────────────────────
    card(
      full_screen = TRUE,
      padding     = 0,
      card_header(
        class = "d-flex align-items-center gap-2",
        bsicons::bs_icon("map-fill"),
        " Distribución Geoespacial de Lesiones Cervicouterinas",
        tags$small(
          class = "ms-auto text-muted fw-normal",
          "Región Loreto \u2014 Perú (datos simulados)"
        )
      ),
      leafletOutput(ns("mapa_leaflet"), height = "600px")
    )

  ) # /layout_sidebar
}


# ─────────────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────────────

mod_mapa_server <- function(id, api_url, api_online) {
  moduleServer(id, function(input, output, session) {

    # ── Reactive: puntos filtrados según selector ─────────────────────────────
    df_filtrado <- reactive({
      req(input$bethesda_filter)   # exige que haya al menos 1 clase seleccionada
      .DF_GEO[.DF_GEO$clase %in% input$bethesda_filter, ]
    })

    # ── Mapa base (se renderiza UNA sola vez) ─────────────────────────────────
    # Las capas dinámicas (heatmap, markers) se gestionan con leafletProxy.
    output$mapa_leaflet <- renderLeaflet({
      leaflet(options = leafletOptions(zoomControl = TRUE)) %>%
        # Tiles CartoDB Positron: fondo gris claro, ideal para capas de calor
        addProviderTiles(
          provider = providers$CartoDB.Positron,
          options  = providerTileOptions(maxZoom = 16)
        ) %>%
        # Vista inicial centrada en Iquitos / Loreto
        setView(lng = -73.5, lat = -4.15, zoom = 7) %>%
        # Control de capas (se poblarán dinámicamente)
        addLayersControl(
          overlayGroups = c("Calor", "Marcadores"),
          options       = layersControlOptions(collapsed = FALSE)
        )
    })

    # ── Observer: actualiza capas reactivamente ───────────────────────────────
    # Usa leafletProxy para no redibujar el mapa base completo.
    observe({
      datos <- df_filtrado()
      proxy <- leafletProxy(session$ns("mapa_leaflet"), data = datos)

      # Limpiar capas dinámicas previas
      proxy %>%
        clearHeatmap() %>%
        clearGroup("Marcadores")

      # ── Capa 1: Heatmap de densidad ─────────────────────────────────────────
      # FIX: se omite el parámetro `gradient` para evitar el error
      # "no applicable method for 'toPaletteFunc' applied to class 'list'".
      # La versión por defecto de leaflet.extras usa amarillo → rojo, que es
      # semánticamente correcta para zonas de alta concentración de positivos.
      if (isTRUE(input$mostrar_heatmap) && nrow(datos) > 0) {
        proxy %>%
          addHeatmap(
            lng       = ~lng,
            lat       = ~lat,
            intensity = ~intensidad,  # valor normalizado [0, 1]
            blur      = 22,
            max       = 0.8,          # umbral de saturación del calor
            radius    = 20,
            group     = "Calor"
            # gradient no se pasa — usa el gradiente nativo de Leaflet.heat
          )
      }

      # ── Capa 2: Marcadores individuales con popup ───────────────────────────
      if (isTRUE(input$mostrar_markers) && nrow(datos) > 0) {
        colores_caso <- .COLORES_BETHESDA_MAPA[datos$clase]
        proxy %>%
          addCircleMarkers(
            lng          = ~lng,
            lat          = ~lat,
            radius       = 6,
            color        = colores_caso,
            fillColor    = colores_caso,
            fillOpacity  = 0.85,
            opacity      = 1,
            weight       = 1.5,
            group        = "Marcadores",
            popup        = ~paste0(
              "<b>Clasificaci\u00f3n:</b> ", clase, "<br>",
              "<b>Centro:</b> ",             centro, "<br>",
              "<b>Coordenadas:</b> ",
              round(lat, 4), ", ", round(lng, 4)
            ),
            label        = ~clase,
            labelOptions = labelOptions(noHide = FALSE, textsize = "12px")
          )
      }
    })

    # ── Resumen reactivo de casos en el sidebar ───────────────────────────────
    output$resumen_casos <- renderUI({
      datos  <- df_filtrado()
      conteo <- sort(table(datos$clase), decreasing = TRUE)
      total  <- nrow(datos)

      tags$div(
        tags$p(
          tags$strong("Casos mostrados: "),
          tags$span(total, class = "text-primary fw-bold"),
          class = "mb-1 small"
        ),
        tags$ul(
          class = "list-unstyled small mb-0",
          lapply(names(conteo), function(cls) {
            tags$li(
              tags$span(
                style = paste0("color:", .COLORES_BETHESDA_MAPA[[cls]], ";"),
                "\u25cf "   # ●
              ),
              tags$strong(cls), ": ", conteo[[cls]], " casos"
            )
          })
        )
      )
    })

  }) # /moduleServer
}
