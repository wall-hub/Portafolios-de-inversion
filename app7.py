import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, date
import time
import base64 

# ===============================================
# 0) CONFIGURACIÓN INICIAL Y ESTILOS
# ===============================================

st.set_page_config(
    page_title="Inversión Inteligente: Simulador de Portafolios", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estructura inicial de portafolios
PORTAFOLIOS_INICIALES = {
    "Conservador": {
        "AGG": 0.40, "IEF": 0.20, "VTI": 0.20, "IAU": 0.10, "VIG": 0.10
    },
    "Moderado": {
        "VTI": 0.25, "SPY": 0.15, "EFA": 0.15, "VWO": 0.15, "AGG": 0.20, "IAU": 0.05, "QQQ": 0.05
    },
    "Agresivo": {
        "QQQ": 0.30, "SPY": 0.20, "IWM": 0.15, "VGT": 0.15, "VWO": 0.10, "SMH": 0.05, "GOOGL": 0.025, "NVDA": 0.025
    }
}

# Inicializar st.session_state con los portafolios si no existen
if 'portafolios_personalizados' not in st.session_state:
    st.session_state['portafolios_personalizados'] = PORTAFOLIOS_INICIALES

# ===============================================
# 1) FUNCIONES DE CÁLCULO Y UTILIDADES
# ===============================================

def normalizar(series, invertir=False):
    """Normaliza una serie entre 0 y 1. Si invertir=True, 0 se convierte en 1 y viceversa."""
    if series.empty or series.max() == series.min():
        return pd.Series([0.5] * len(series), index=series.index)
        
    s = (series - series.min()) / (series.max() - series.min())
    if invertir:
        s = 1 - s
    return s

@st.cache_data(show_spinner="⏳ Descargando datos históricos de Yahoo Finance...")
def descargar_datos(tickers, years):
    """Descarga datos mensuales ajustados de yfinance y los guarda en caché."""
    data = yf.download(tickers, period=f"{years}y", interval="1mo", auto_adjust=True, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"].copy()
    
    data = data.sort_index().dropna(how="all", axis=0).dropna(how="all", axis=1)
    data = data.dropna(axis=1, thresh=int(0.9*len(data)))
    data = data.dropna(axis=0)
    return data

def calcular_retornos(precios):
    """Calcula retornos mensuales simples."""
    return precios.pct_change().dropna(how="any")

def retornos_portafolio(ret_activos, pesos_dict):
    """Calcula el retorno mensual ponderado del portafolio, usando pesos normalizados."""
    total_peso = sum(pesos_dict.values())
    if total_peso == 0:
        return pd.Series([], dtype='float64')
        
    pesos_normalizados = {k: v / total_peso for k, v in pesos_dict.items()}
    
    cols = [c for c in pesos_normalizados.keys() if c in ret_activos.columns]
    
    if not cols:
        return pd.Series([], dtype='float64')

    w = np.array([pesos_normalizados[c] for c in cols], dtype=float)
    return (ret_activos[cols] @ w)

def mc_trayectorias(ret_activos, pesos_dict, aporte_mensual, n_sim, fut_years):
    """Genera trayectorias Monte Carlo para el valor del portafolio con DCA."""
    
    total_peso = sum(pesos_dict.values())
    if total_peso == 0:
        meses = fut_years * 12
        return np.zeros((meses, n_sim), dtype=float)
        
    pesos_normalizados = {k: v / total_peso for k, v in pesos_dict.items()}
    
    cols = [c for c in pesos_normalizados.keys() if c in ret_activos.columns]
    
    if not cols or ret_activos[cols].shape[1] == 0:
        meses = fut_years * 12
        return np.zeros((meses, n_sim), dtype=float)

    w = np.array([pesos_normalizados[c] for c in cols], dtype=float)
    mu = ret_activos[cols].mean().values
    cov = ret_activos[cols].cov().values
    
    meses = fut_years * 12
    valores = np.zeros((meses, n_sim), dtype=float)
    
    try:
        r_act = np.random.multivariate_normal(mean=mu, cov=cov, size=(meses, n_sim))
    except np.linalg.LinAlgError:
        # st.warning ya se maneja en el flujo de ejecución principal
        cov_diag = np.diag(np.diag(cov))
        r_act = np.random.multivariate_normal(mean=mu, cov=cov_diag, size=(meses, n_sim))

    r_port = np.dot(r_act, w)
    
    V = np.zeros(n_sim)
    for m in range(meses):
        V = (V + aporte_mensual) * (1.0 + r_port[m, :])
        valores[m, :] = V
        
    return valores

def dca_backtest(rp, aporte):
    """Calcula el valor del portafolio bajo DCA históricamente."""
    V = 0.0
    vals = []
    for r in rp.values:
        V = (V + aporte) * (1.0 + r)
        vals.append(V)
    return pd.Series(vals, index=rp.index)

# ===============================================
# 2) UI LATERAL (SIDEBAR) Y CARGA DE DATOS
# ===============================================

st.sidebar.title("⚙️ Configuración del Análisis")
st.sidebar.markdown("---")

# INPUTS DE LA BARRA LATERAL
aporte_mensual = st.sidebar.number_input("💵 Aporte Mensual (USD)", value=200, step=50)
hist_years = st.sidebar.slider("🕰️ Años de Historia (Backtest)", 5, 20, 10)
fut_years = st.sidebar.slider("🔮 Años a Proyectar (Monte Carlo)", 1, 50, 20, step=1) 
n_sim = st.sidebar.slider("🎲 Nº Simulaciones MC", 100, 10000, 1000, step=100)

PORTAFOLIOS = st.session_state['portafolios_personalizados']

# **VISUALIZACIÓN DE COMPOSICIÓN ACTUAL**
st.sidebar.markdown("---")
st.sidebar.subheader("Actual ⚖️ Composición de Portafolios")

for name, pesos in PORTAFOLIOS.items():
    st.sidebar.caption(f"**{name}**")
    total_peso = sum(pesos.values())
    if total_peso > 0:
        pesos_norm = {k: v / total_peso for k, v in pesos.items()}
    else:
        pesos_norm = {k: 0 for k in pesos.keys()}

    df_pesos = pd.DataFrame(list(pesos_norm.items()), columns=["Ticker", "Peso"]).set_index("Ticker")
    df_pesos["Peso"] = (df_pesos["Peso"] * 100).round(1).astype(str) + '%'
    st.sidebar.dataframe(df_pesos, use_container_width=True, height=len(df_pesos) * 35 + 38)
st.sidebar.markdown("---")


# Preparación de tickers (se usa PORTAFOLIOS_INICIALES para descargar todo el set de activos)
all_tickers = set()
for p in PORTAFOLIOS_INICIALES.values(): 
    all_tickers.update(p.keys())

# Carga de datos
try:
    precios = descargar_datos(sorted(list(all_tickers)), hist_years)
    rets = calcular_retornos(precios)
    st.sidebar.success("✅ Datos cargados correctamente.")
    
except Exception as e:
    st.sidebar.error(f"❌ Error crítico al descargar datos: {e}")
    st.stop()

# ===============================================
# 3) UI PRINCIPAL: VISUALIZACIÓN
# ===============================================

st.header("Simulador de Portafolios: Analiza y Proyecta")
st.markdown("Ajusta los pesos de tus portafolios y observa cómo cambian el riesgo, el retorno y las proyecciones futuras.")
st.markdown("---")

# --- FRONTERA DE MARKOWITZ EN UN EXPANDER ---
with st.expander("🌐 Ver Frontera Eficiente de Markowitz (Riesgo vs. Retorno)", expanded=False):
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("""
        **Guía:**
        
        Muestra la **relación Riesgo (Volatilidad)** vs. **Retorno Esperado**.
        
        - **Curva Superior:** Frontera Eficiente (Portafolios óptimos).
        - **Objetivo:** Estar lo más **arriba** (más retorno) y a la **izquierda** (menos riesgo) posible.
        - **Puntos de Color:** Representan tus portafolios personalizados.
        """)
    
    with col1:
        n_portfolios_frontier = 3000
        means = rets.mean() * 12 
        cov_matrix = rets.cov() * 12 
        
        results_list = []
        
        for _ in range(n_portfolios_frontier):
            weights = np.random.random(len(rets.columns))
            weights /= np.sum(weights)
            
            p_ret = np.dot(weights, means)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            results_list.append([p_vol, p_ret])
            
        results_array = np.array(results_list)
        
        fig_ef, ax_ef = plt.subplots(figsize=(10, 6))
        
        sc = ax_ef.scatter(results_array[:, 0], results_array[:, 1], c=results_array[:, 1]/results_array[:, 0], 
                           marker='o', cmap='viridis', s=10, alpha=0.3, label='Simulaciones Aleatorias')
        
        colors = {'Conservador': 'blue', 'Moderado': 'orange', 'Agresivo': 'red'}
        
        for name, pesos in PORTAFOLIOS.items():
            rp_series = retornos_portafolio(rets, pesos)
            if not rp_series.empty:
                port_ret_anual = rp_series.mean() * 12
                port_vol_anual = rp_series.std() * np.sqrt(12)
                
                ax_ef.scatter(port_vol_anual, port_ret_anual, color=colors[name], s=150, edgecolors='black', label=name, zorder=5)
                ax_ef.text(port_vol_anual, port_ret_anual + 0.005, name, fontsize=9, ha='center', weight='bold')

        ax_ef.set_title('Frontera Eficiente de Markowitz')
        ax_ef.set_xlabel('Volatilidad Anualizada (Riesgo)')
        ax_ef.set_ylabel('Retorno Esperado Anualizado')
        plt.colorbar(sc, label='Ratio Sharpe (aprox)')
        ax_ef.legend()
        ax_ef.grid(True, alpha=0.3)
        st.pyplot(fig_ef)

# --- ANÁLISIS DETALLADO POR TABS ---

tabs = st.tabs(["🛡️ Conservador", "⚖️ Moderado", "🚀 Agresivo", "📊 Comparativa Final"])

results_mc_store = {}
historical_metrics = []

for i, (nombre_port, pesos_iniciales) in enumerate(PORTAFOLIOS_INICIALES.items()):
    with tabs[i]:
        st.subheader(f"{nombre_port} | Simulación Detallada")
        st.markdown("---")
        
        # --- SECCIÓN 1: PERSONALIZACIÓN DE PESOS ---
        with st.container(border=True):
            st.caption("Ajusta la Composición del Portafolio (Pesos en %)")
            
            current_pesos = st.session_state['portafolios_personalizados'][nombre_port]
            new_pesos = {}
            
            # Crear columnas dinámicas para los inputs de pesos
            cols = st.columns(len(current_pesos))
            
            for j, (ticker, peso) in enumerate(current_pesos.items()):
                with cols[j]:
                    peso_porc = st.number_input(
                        label=ticker, 
                        value=peso * 100, 
                        min_value=0.0, 
                        step=0.5, 
                        key=f"{nombre_port}_{ticker}",
                        help=f"Peso de {ticker} en % (ej: 40 para 40%)"
                    )
                    new_pesos[ticker] = peso_porc / 100
            
            if new_pesos != current_pesos:
                st.session_state['portafolios_personalizados'][nombre_port] = new_pesos
                
            total_input_peso = sum(new_pesos.values()) * 100
            
            if total_input_peso == 0:
                 st.warning("⚠️ La suma de pesos es 0%. Debes asignar pesos para que el portafolio funcione.")
            elif abs(total_input_peso - 100) > 0.1:
                 st.info(f"Suma de pesos ingresados: **{total_input_peso:.1f}%**. Los cálculos se realizarán con pesos normalizados al 100%.")
            else:
                 st.success("Suma de pesos: 100%. ¡Portafolio listo!")

        st.markdown("---")
        
        # --- SECCIÓN 2: EJECUCIÓN Y CÁLCULO DE MÉTRICAS ---
        rp = retornos_portafolio(rets, new_pesos)

        if rp.empty:
            st.warning("No se pudieron cargar datos para este portafolio con los activos y el periodo solicitado.")
            continue

        # --- BACKTEST ---
        serie_dca = dca_backtest(rp, aporte_mensual)
        
        total_invertido = len(serie_dca) * aporte_mensual
        valor_final_hist = serie_dca.iloc[-1]
        cagr_hist = (valor_final_hist / total_invertido) ** (1 / (len(serie_dca)/12)) - 1 if total_invertido > 0 else 0
        vol_anual = rp.std() * np.sqrt(12)
        sharpe = (rp.mean() * 12) / vol_anual if vol_anual > 0 else 0
        
        historical_metrics.append({"Portafolio": nombre_port, "Retorno": cagr_hist, "Volatilidad": vol_anual, "Sharpe": sharpe})
        
        # --- MONTE CARLO ---
        with st.spinner(f"Calculando {n_sim} proyecciones para {fut_years} años..."):
            mc_vals = mc_trayectorias(rets, new_pesos, aporte_mensual, n_sim, fut_years)
            
            if mc_vals.sum() == 0 and mc_vals.size > 0:
                st.error("No se pudo realizar la simulación Monte Carlo.")
                continue

            results_mc_store[nombre_port] = mc_vals[-1, :]
            
            mediana_final_mc = np.median(mc_vals[-1, :])
            total_aporte_mc = fut_years * 12 * aporte_mensual
            cagr_mc = (mediana_final_mc / total_aporte_mc) ** (1 / fut_years) - 1 if total_aporte_mc > 0 else 0


        # --- VIZUALIZACIÓN DE MÉTRICAS CLAVE (KPIs) ---
        st.header("Análisis de Rendimiento")
        
        col_risk, col_aporte, col_sharpe = st.columns(3)
        col_aporte.metric("Inversión Total Aportada", f"${total_invertido:,.0f}")
        col_risk.metric("Volatilidad Anual (Riesgo)", f"{vol_anual:.2%}")
        col_sharpe.metric("Ratio Sharpe (Eficiencia)", f"{sharpe:.2f}")

        st.subheader("1. Backtest Histórico (DCA)")
        
        col_h1, col_h2 = st.columns(2)
        col_h1.metric("💰 Valor Final (Histórico)", f"${valor_final_hist:,.0f}")
        col_h2.metric("📈 CAGR Histórico", f"{cagr_hist:.2%}")
        
        # Gráfico Backtest
        fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
        ax_bt.plot(serie_dca.index, serie_dca.values, label='Valor Portafolio', color='#2ecc71', linewidth=2)
        ax_bt.plot(serie_dca.index, [aporte_mensual * i for i in range(1, len(serie_dca)+1)], '--', label='Dinero Aportado', color='#3498db', alpha=0.8)
        ax_bt.set_title(f"Evolución Histórica DCA ({hist_years} años)", fontsize=14)
        ax_bt.legend()
        ax_bt.grid(True, alpha=0.2)
        st.pyplot(fig_bt)
        
        st.divider()
        
        # --- SECCIÓN 3: PROYECCIÓN MONTE CARLO (MODIFICADA) ---
        st.subheader(f"2. Proyección Monte Carlo ({fut_years} años)")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("🔮 Valor Final (Mediana)", f"${mediana_final_mc:,.0f}")
        col_m2.metric("🚀 CAGR Proyectado", f"{cagr_mc:.2%}")
        
        # Gráfico Monte Carlo
        p5 = np.percentile(mc_vals, 5, axis=1)
        p50 = np.percentile(mc_vals, 50, axis=1)
        p95 = np.percentile(mc_vals, 95, axis=1)
        
        fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
        
        # MODIFICACIÓN CLAVE: Crear el eje X de fechas
        start_date = date.today()
        # Generar las fechas mensuales desde la fecha actual
        date_range = pd.date_range(start=start_date, periods=len(p50), freq='M') 

        ax_mc.plot(date_range, p50, color='#3498db', label='Mediana (Escenario Base)', linewidth=2)
        ax_mc.fill_between(date_range, p5, p95, color='#3498db', alpha=0.15, label='Rango de Confianza P5 - P95')
        ax_mc.set_title(f"Distribución de Riqueza Proyectada a {fut_years} años", fontsize=14)
        
        # MODIFICACIÓN CLAVE: Cambiar la etiqueta del eje X
        ax_mc.set_xlabel("Años Futuros (desde la fecha actual)") 
        
        ax_mc.set_ylabel("Valor Portafolio (USD)")
        
        # Ajustar el formato del eje X para que muestre solo el año
        ax_mc.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        
        ax_mc.legend()
        ax_mc.grid(True, alpha=0.2)
        st.pyplot(fig_mc)
        
        # Estadísticas MC (Escenarios)
        peor_caso = np.percentile(results_mc_store[nombre_port], 5)
        mejor_caso = np.percentile(results_mc_store[nombre_port], 95)
        
        col_mc1, col_mc2, col_mc3 = st.columns(3)
        col_mc1.error(f"P5 (Peor Caso): **${peor_caso:,.0f}**")
        col_mc2.success(f"P50 (Mediana): **${mediana_final_mc:,.0f}**")
        col_mc3.warning(f"P95 (Mejor Caso): **${mejor_caso:,.0f}**")


# ===============================================
# 5) COMPARATIVA FINAL
# ===============================================
with tabs[3]:
    st.header("Análisis Comparativo y Descarga")
    st.markdown("Compara las métricas de rendimiento y riesgo de tus portafolios personalizados.")
    
    st.subheader("1. ⚖️ Distribución Final de Monte Carlo")
    
    if results_mc_store:
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        data_to_plot = [results_mc_store[k] for k in results_mc_store.keys()]
        labels = list(results_mc_store.keys())
        ax_box.boxplot(data_to_plot, labels=labels, vert=True, showfliers=False, 
                       patch_artist=True, boxprops=dict(facecolor='#3498db', color='#2980b9'))
        ax_box.set_title(f"Comparación del Valor Final Proyectado a {fut_years} años (Mediana y Rangos)", fontsize=14)
        ax_box.set_ylabel("Valor en USD")
        ax_box.grid(axis='y', alpha=0.2)
        st.pyplot(fig_box)
    else:
        st.warning("No se generaron datos de Monte Carlo. Asegúrate de que los portafolios tengan pesos válidos.")

    st.divider()

    # Dataframe de Métricas Históricas
    df_radar = pd.DataFrame(historical_metrics).set_index("Portafolio")
    st.subheader("2. 📈 Resumen de Métricas Históricas (Backtest)")
    st.dataframe(df_radar.style.format({
        "Retorno": "{:.2%}", 
        "Volatilidad": "{:.2%}",
        "Sharpe": "{:.2f}"
    }), use_container_width=True)
    
    # Botón de Descarga
    if not df_radar.empty:
        col_descarga, col_pdf = st.columns([1, 2])
        
        csv = df_radar.to_csv(sep=';').encode('utf-8')
        col_descarga.download_button(
            label="⬇️ Descargar Métricas (CSV)",
            data=csv,
            file_name=f'informe_metricas_portafolios_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            help="Descarga el cuadro de métricas históricas de los portafolios."
        )
        
        col_pdf.markdown("""
        **💡 Sugerencia de Informe PDF:** Para guardar la página completa (gráficas y tablas) como un informe PDF, usa la función de **Imprimir (Ctrl + P o Cmd + P)** de tu navegador y elige la opción **"Guardar como PDF"**.
        """)
    
    st.divider()

    # GRÁFICO RADAR
    st.subheader("3. 🎯 Perfil de Riesgo/Rendimiento Normalizado (Gráfico Radar)")

    if not df_radar.empty:
        indicadores = list(df_radar.columns) 
        
        radar_norm = pd.DataFrame({
            "Retorno": normalizar(df_radar["Retorno"]),
            "Volatilidad": normalizar(df_radar["Volatilidad"], invertir=True),
            "Sharpe": normalizar(df_radar["Sharpe"])
        })

        N = len(indicadores)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() 
        angles += angles[:1] 

        fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        


        
        for i, port in enumerate(radar_norm.index):
            stats = radar_norm.loc[port].values.flatten().tolist()
            stats += stats[:1] 

            ax_radar.plot(angles, stats, label=port, linewidth=2, alpha=0.7)
            ax_radar.fill(angles, stats, alpha=0.1)

        ax_radar.set_thetagrids(np.array(angles[:-1]) * 180/np.pi, indicadores, color='gray', size=11)
        
        ax_radar.set_yticks(np.arange(0.2, 1.1, 0.2))
        ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", size=8)
        ax_radar.set_ylim(0, 1)

        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        ax_radar.set_title("Perfil de Riesgo/Retorno (Normalizado a 1)", size=14, pad=20)
        st.pyplot(fig_radar)
    else:
        st.warning("No hay datos suficientes para generar la comparativa del gráfico Radar.")