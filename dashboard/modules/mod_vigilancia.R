# ==============================================================================
# AmazonCervix-AI — Módulo 1: Vigilancia Epidemiológica
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────────────────────────────────────

.PALETA_BETHESDA <- c(
  "NILM"   = "#2C7A4B",   # Verde selva  — sin lesión
  "ASC-US" = "#5B8DB8",   # Azul medio   — atípicas indeterminadas
  "ASC-H"  = "#F39C12",   # Ámbar        — atípicas (no excluye HSIL)
  "LSIL"   = "#E67E22",   # Naranja      — bajo grado
  "HSIL"   = "#E74C3C",   # Rojo         — alto grado
  "SCC"    = "#8E1A0E"    # Rojo oscuro  — carcinoma
)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

mod_vigilancia_ui <- function(id) {
  ns <- NS(id)   # namespace: todos los IDs quedan bajo "mod_vigilancia-..."

  div(
    class = "container-fluid pb-4",

    # ── Fila de KPIs ──────────────────────────────────────────────────────────
    layout_column_wrap(
      width         = 1/3,
      fill          = FALSE,
      heights_equal = "row",

      # KPI 1 — Casos Totales Detectados
      value_box(
        title    = "Casos Totales Detectados",
        value    = "1,420",
        showcase = bsicons::bs_icon("clipboard2-pulse-fill"),
        theme    = value_box_theme(bg = "#2C7A4B", fg = "#ffffff"),
        p("Acumulado del período analizado", class = "mb-0 small opacity-75")
      ),

      # KPI 2 — Tiempo Promedio de Diagnóstico
      value_box(
        title    = "Tiempo Prom. de Diagnóstico",
        value    = "1.8 horas",
        showcase = bsicons::bs_icon("clock-history"),
        theme    = value_box_theme(bg = "#5B8DB8", fg = "#ffffff"),
        p("Desde la carga de imagen hasta la clasificación",
          class = "mb-0 small opacity-75")
      ),

      # KPI 3 — Precisión Global del Modelo
      value_box(
        title    = "Precisión Global del Modelo",
        value    = "94.5%",
        showcase = bsicons::bs_icon("graph-up-arrow"),
        theme    = value_box_theme(bg = "#27AE60", fg = "#ffffff"),
        p("Balanced accuracy — Capa Gold (RIVA + SIPaKMeD)",
          class = "mb-0 small opacity-75")
      )
    ), # /KPIs

    br(),

    # ── Fila de gráficos ──────────────────────────────────────────────────────
    layout_columns(
      col_widths = c(7, 5),

      # Gráfico 1 — Serie de tiempo
      card(
        full_screen = TRUE,
        card_header(
          class = "d-flex align-items-center gap-2",
          bsicons::bs_icon("activity"),
          " Evaluaciones Mensuales (últimos 6 meses)"
        ),
        card_body(
          class = "p-2",
          plotOutput(ns("serie_tiempo"), height = "320px")
        )
      ),

      # Gráfico 2 — Distribución Bethesda
      card(
        full_screen = TRUE,
        card_header(
          class = "d-flex align-items-center gap-2",
          bsicons::bs_icon("bar-chart-fill"),
          " Distribución de Diagnósticos Bethesda"
        ),
        card_body(
          class = "p-2",
          plotOutput(ns("bethesda_barras"), height = "320px")
        )
      )
    ) # /gráficos
  ) # /div
}


# ─────────────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────────────

mod_vigilancia_server <- function(id, api_url, api_online) {
  moduleServer(id, function(input, output, session) {

    # ── Mock data: serie temporal (últimos 6 meses) ───────────────────────────
    # TODO: reemplazar por GET(paste0(api_url, "/stats/serie_mensual"))
    df_serie <- data.frame(
      mes = factor(
        c("Ene", "Feb", "Mar", "Abr", "May", "Jun"),
        levels = c("Ene", "Feb", "Mar", "Abr", "May", "Jun")
      ),
      evaluaciones = c(198, 221, 245, 187, 263, 306)
    )

    # ── Gráfico 1: Área + línea — evaluaciones mensuales ─────────────────────
    output$serie_tiempo <- renderPlot({
      ggplot(df_serie, aes(x = mes, y = evaluaciones, group = 1)) +
        geom_area(fill = "#2C7A4B", alpha = 0.12) +
        geom_line(color = "#2C7A4B", linewidth = 1.2) +
        geom_point(
          color = "#2C7A4B", size = 3.5,
          shape = 21, fill = "white", stroke = 1.8
        ) +
        geom_text(
          aes(label = evaluaciones),
          vjust = -1.2, size = 3.2, color = "#555555", family = "sans"
        ) +
        scale_y_continuous(
          limits = c(0, max(df_serie$evaluaciones) * 1.25),
          expand = expansion(mult = c(0, 0))
        ) +
        labs(x = NULL, y = "N\u00b0 de Evaluaciones", title = NULL) +
        theme_minimal(base_size = 12) +
        theme(
          panel.grid.major.x = element_blank(),
          panel.grid.minor   = element_blank(),
          axis.text          = element_text(color = "#444444"),
          axis.title.y       = element_text(color = "#666666", size = 10),
          plot.margin        = margin(t = 8, r = 12, b = 4, l = 4)
        )
    }, res = 110)


    # ── Mock data: distribución Bethesda ──────────────────────────────────────
    # TODO: reemplazar por GET(paste0(api_url, "/stats/distribucion_bethesda"))
    df_bethesda <- data.frame(
      clase = factor(
        c("NILM", "ASC-US", "ASC-H", "LSIL", "HSIL", "SCC"),
        levels = c("NILM", "ASC-US", "ASC-H", "LSIL", "HSIL", "SCC")
      ),
      casos = c(742, 198, 87, 243, 121, 29)
    )

    # ── Gráfico 2: Barras horizontales — distribución Bethesda ───────────────
    output$bethesda_barras <- renderPlot({
      ggplot(
        df_bethesda,
        aes(x = reorder(clase, casos), y = casos, fill = clase)
      ) +
        geom_col(width = 0.65, show.legend = FALSE) +
        geom_text(
          aes(label = scales::comma(casos)),
          hjust = -0.2, size = 3.2, color = "#444444", family = "sans"
        ) +
        scale_fill_manual(values = .PALETA_BETHESDA) +
        scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
        coord_flip() +
        labs(x = NULL, y = "Casos detectados", title = NULL) +
        theme_minimal(base_size = 12) +
        theme(
          panel.grid.major.y = element_blank(),
          panel.grid.minor   = element_blank(),
          axis.text.y        = element_text(color = "#333333",
                                            face = "bold", size = 11),
          axis.text.x        = element_text(color = "#666666", size = 9),
          axis.title.x       = element_text(color = "#666666", size = 10),
          plot.margin        = margin(t = 4, r = 20, b = 4, l = 4)
        )
    }, res = 110)

  }) # /moduleServer
}
