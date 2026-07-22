# ==============================================================================
# AmazonCervix-AI — Módulo 4: Análisis de Desempeño del Modelo de IA
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

.DSP_CLASES <- c("NILM", "ASC-US", "ASC-H", "LSIL", "HSIL", "SCC")

# Métricas globales simuladas (Capa Gold — balanced test set)
.DSP_METRICAS <- list(
  accuracy  = 94.3,
  precision = 92.1,
  recall    = 91.8,
  f1        = 91.9
)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

mod_desempeno_ui <- function(id) {
  ns <- NS(id)

  div(
    class = "container-fluid pb-4",

    # ── Fila 1: KPIs de métricas globales ─────────────────────────────────────
    layout_column_wrap(
      width         = 1/4,
      fill          = FALSE,
      heights_equal = "row",

      # KPI 1 — Accuracy
      value_box(
        title    = "Accuracy Global",
        value    = paste0(.DSP_METRICAS$accuracy, "%"),
        showcase = bsicons::bs_icon("bullseye"),
        theme    = value_box_theme(bg = "#2C7A4B", fg = "#ffffff"),
        p("Exactitud sobre el test set Capa Gold",
          class = "mb-0 small opacity-75")
      ),

      # KPI 2 — Precisión
      value_box(
        title    = "Precisión (Macro)",
        value    = paste0(.DSP_METRICAS$precision, "%"),
        showcase = bsicons::bs_icon("patch-check-fill"),
        theme    = value_box_theme(bg = "#5B8DB8", fg = "#ffffff"),
        p("Promedio no ponderado por clase Bethesda",
          class = "mb-0 small opacity-75")
      ),

      # KPI 3 — Recall
      value_box(
        title    = "Recall / Sensibilidad",
        value    = paste0(.DSP_METRICAS$recall, "%"),
        showcase = bsicons::bs_icon("search-heart-fill"),
        theme    = value_box_theme(bg = "#27AE60", fg = "#ffffff"),
        p("Capacidad de detectar casos positivos reales",
          class = "mb-0 small opacity-75")
      ),

      # KPI 4 — F1-Score
      value_box(
        title    = "F1-Score (Macro)",
        value    = paste0(.DSP_METRICAS$f1, "%"),
        showcase = bsicons::bs_icon("bar-chart-line-fill"),
        theme    = value_box_theme(bg = "#E67E22", fg = "#ffffff"),
        p("Media arm\u00f3nica Precisi\u00f3n-Recall por clase",
          class = "mb-0 small opacity-75")
      )
    ), # /KPIs

    br(),

    # ── Fila 2: Gráficos principales ──────────────────────────────────────────
    layout_columns(
      col_widths = c(6, 6),

      # Tarjeta 1 — Matriz de Confusión
      card(
        full_screen = TRUE,
        card_header(
          class = "d-flex align-items-center gap-2",
          bsicons::bs_icon("grid-3x3-gap-fill"),
          " Matriz de Confusi\u00f3n \u2014 Test Set (Capa Gold)"
        ),
        card_body(
          class = "p-2",
          plotOutput(ns("confusion_matrix"), height = "420px")
        ),
        card_footer(
          class = "text-muted small",
          bsicons::bs_icon("info-circle"),
          " Filas: clase real \u00b7 Columnas: clase predicha \u00b7
          Diagonal principal = predicciones correctas"
        )
      ),

      # Tarjeta 2 — Curvas ROC
      card(
        full_screen = TRUE,
        card_header(
          class = "d-flex align-items-center gap-2",
          bsicons::bs_icon("graph-up"),
          " Curvas ROC Multi-Clase (OvR)"
        ),
        card_body(
          class = "p-2",
          plotOutput(ns("roc_curves"), height = "420px")
        ),
        card_footer(
          class = "text-muted small",
          bsicons::bs_icon("info-circle"),
          " Estrategia One-vs-Rest (OvR) \u00b7
          L\u00ednea punteada = clasificador aleatorio (AUC = 0.50)"
        )
      )
    ) # /layout_columns gráficos
  ) # /div container-fluid
}


# ─────────────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────────────

mod_desempeno_server <- function(id, api_url, api_online) {
  moduleServer(id, function(input, output, session) {

    # ── Mock data: Matriz de Confusión 6×6 ────────────────────────────────────
    # Valores que reflejan un modelo de buen desempeño:
    # diagonal alta, errores principalmente entre clases adyacentes en severidad.
    # TODO: reemplazar por GET(paste0(api_url, "/model/confusion_matrix"))
    conf_raw <- matrix(
      c(
        # NILM  ASCUS  ASCH  LSIL  HSIL  SCC   ← predicho
          538,   12,    3,    8,    2,    1,    # NILM   (real)
           14,  142,    6,    5,    3,    0,    # ASC-US
            4,    8,   62,    4,    5,    0,    # ASC-H
            9,    6,    4,  172,   10,    1,    # LSIL
            3,    4,    5,   11,   87,    3,    # HSIL
            1,    0,    1,    2,    5,   21     # SCC
      ),
      nrow = 6, byrow = TRUE,
      dimnames = list(Real     = .DSP_CLASES,
                      Predicho = .DSP_CLASES)
    )

    df_cm <- as.data.frame(as.table(conf_raw))
    names(df_cm) <- c("Real", "Predicho", "Conteo")

    # Normalizar por fila (porcentaje de recall por clase real)
    df_cm$Porcentaje <- ave(
      df_cm$Conteo, df_cm$Real,
      FUN = function(x) round(x / sum(x) * 100, 1)
    )

    # Orden de severidad para los ejes
    df_cm$Real     <- factor(df_cm$Real,     levels = rev(.DSP_CLASES))
    df_cm$Predicho <- factor(df_cm$Predicho, levels = .DSP_CLASES)

    # ── Gráfico 1: Matriz de Confusión (geom_tile) ───────────────────────────
    output$confusion_matrix <- renderPlot({
      ggplot(df_cm, aes(x = Predicho, y = Real, fill = Porcentaje)) +
        geom_tile(color = "white", linewidth = 0.6) +
        # Etiquetas: conteo + porcentaje
        geom_text(
          aes(
            label = paste0(Conteo, "\n(", Porcentaje, "%)"),
            color = ifelse(Porcentaje > 55, "white", "#222222")
          ),
          size   = 3.2,
          family = "sans",
          lineheight = 0.9
        ) +
        scale_fill_gradient(
          low    = "#EAF4EE",   # verde muy claro — valores bajos
          high   = "#1A5E35",   # verde oscuro    — diagonal principal
          name   = "% por fila",
          limits = c(0, 100)
        ) +
        scale_color_identity() +   # usa los colores del aes(color=...)
        scale_x_discrete(position = "top") +
        labs(
          x     = "Clase Predicha",
          y     = "Clase Real",
          title = NULL
        ) +
        theme_minimal(base_size = 12) +
        theme(
          axis.text.x       = element_text(face = "bold", color = "#333333",
                                            size = 10),
          axis.text.y       = element_text(face = "bold", color = "#333333",
                                            size = 10),
          axis.title        = element_text(color = "#555555", size = 11),
          panel.grid        = element_blank(),
          legend.position   = "right",
          legend.title      = element_text(size = 9),
          plot.margin       = margin(t = 4, r = 4, b = 4, l = 4)
        )
    }, res = 110)


    # ── Mock data: Curvas ROC ─────────────────────────────────────────────────
    # Se simulan curvas usando una función beta acumulada para cada clase,
    # parametrizada para reflejar AUC realistas distintos por clase.
    # TODO: reemplazar por GET(paste0(api_url, "/model/roc"))

    # Parámetros (forma1, forma2) y AUC teórico por clase
    roc_params <- list(
      "NILM"   = list(a = 12, b = 1.5, auc = 0.98),
      "ASC-US" = list(a =  6, b = 2.0, auc = 0.94),
      "ASC-H"  = list(a =  5, b = 2.5, auc = 0.91),
      "LSIL"   = list(a =  8, b = 1.8, auc = 0.96),
      "HSIL"   = list(a =  9, b = 1.6, auc = 0.97),
      "SCC"    = list(a =  7, b = 2.2, auc = 0.95)
    )

    # Clases a mostrar en el gráfico (todas)
    clases_roc <- names(roc_params)

    # Generar 200 puntos de curva por clase
    fpr_seq <- seq(0, 1, length.out = 200)

    df_roc <- do.call(rbind, lapply(clases_roc, function(cls) {
      p   <- roc_params[[cls]]
      tpr <- pbeta(fpr_seq, shape1 = p$a, shape2 = p$b)
      data.frame(
        FPR   = fpr_seq,
        TPR   = tpr,
        Clase = cls,
        AUC   = p$auc
      )
    }))

    df_roc$Clase <- factor(df_roc$Clase, levels = .DSP_CLASES)

    # Paleta de colores para las curvas (misma que el resto del dashboard)
    paleta_roc <- c(
      "NILM"   = "#2C7A4B",
      "ASC-US" = "#5B8DB8",
      "ASC-H"  = "#F39C12",
      "LSIL"   = "#E67E22",
      "HSIL"   = "#E74C3C",
      "SCC"    = "#8E1A0E"
    )

    # Etiquetas de leyenda con AUC incluido
    etiquetas_roc <- sapply(clases_roc, function(cls) {
      sprintf("%s  (AUC = %.2f)", cls, roc_params[[cls]]$auc)
    })
    names(etiquetas_roc) <- clases_roc

    # ── Gráfico 2: Curvas ROC (geom_line) ────────────────────────────────────
    output$roc_curves <- renderPlot({
      ggplot(df_roc, aes(x = FPR, y = TPR, color = Clase)) +
        # Línea diagonal de referencia (clasificador aleatorio)
        geom_abline(
          slope     = 1,
          intercept = 0,
          linetype  = "dashed",
          color     = "#AAAAAA",
          linewidth = 0.8
        ) +
        # Curvas ROC
        geom_line(linewidth = 1.3, alpha = 0.9) +
        # Punto origen y punto (1,1) para anclar correctamente
        geom_point(
          data  = data.frame(FPR = c(0, 1), TPR = c(0, 1)),
          aes(x = FPR, y = TPR),
          color = "#AAAAAA",
          size  = 2,
          inherit.aes = FALSE
        ) +
        scale_color_manual(
          values = paleta_roc,
          labels = etiquetas_roc,
          name   = "Clase Bethesda"
        ) +
        scale_x_continuous(
          labels = scales::percent_format(accuracy = 1),
          limits = c(0, 1),
          expand = expansion(mult = c(0, 0.02))
        ) +
        scale_y_continuous(
          labels = scales::percent_format(accuracy = 1),
          limits = c(0, 1),
          expand = expansion(mult = c(0.01, 0.02))
        ) +
        labs(
          x     = "Tasa de Falsos Positivos (1 \u2212 Especificidad)",
          y     = "Tasa de Verdaderos Positivos (Sensibilidad)",
          title = NULL
        ) +
        theme_minimal(base_size = 12) +
        theme(
          legend.position    = "right",
          legend.text        = element_text(size = 9, family = "mono"),
          legend.title       = element_text(size = 10, face = "bold"),
          legend.key.width   = unit(1.5, "cm"),
          axis.text          = element_text(color = "#444444"),
          axis.title         = element_text(color = "#555555", size = 10),
          panel.grid.minor   = element_blank(),
          panel.grid.major   = element_line(color = "#EEEEEE"),
          plot.margin        = margin(t = 8, r = 12, b = 4, l = 4)
        )
    }, res = 110)

  }) # /moduleServer
}
