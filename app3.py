import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import time
import base64 # Necesario para la descarga

# Configuración de la página
st.set_page_config(page_title="Simulador de Portafolios & Markowitz", layout="wide")

# =========================
# 1) CONFIGURACIÓN Y UTILIDADES
# =========================

PORTAFOLIOS = {
    "Conservador": {
        "AGG": 0.40, "IEF": 0.20, "VTI": 0.20, "IAU": 0.10, "VIG": 0.10
    },
    "Moderado": {
        "VTI": 0.25, "SPY": 0.15, "EFA": 0.15, "VWO": 0.15, "AGG": 0.20, "IAU": 0.05, "QQQ": 0.05
    },
    "Agresivo": {
        "QQQ": 0.30, "SPY": 0.20, "IWM": 0.15, "VGT": 0.15, "VWO": 0.10, "SMH": 0.05, "AAPL": 0.025, "NVDA": 0.025
    }
}

def normalizar(series, invertir=False):
    """Normaliza una serie entre 0 y 1. Si invertir=True, 0 se convierte en 1 y viceversa."""
    if series.empty or series.max() == series.min():
        return pd.Series([0.5] * len(series), index=series.index)
        
    s = (series - series.min()) / (series.max() - series.min())
    if invertir:
        s = 1 - s
    return s

@st.cache_data
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
    """Calcula el retorno mensual ponderado del portafolio."""
    cols = [c for c in pesos_dict.keys() if c in ret_activos.columns]
    
    if not cols:
        return pd.Series([], dtype='float64')

    w = np.array([pesos_dict[c] for c in cols], dtype=float)
    w = w / w.sum()
    return (ret_activos[cols] @ w)

def mc_trayectorias(ret_activos, pesos_dict, aporte_mensual, n_sim, fut_years):
    """Genera trayectorias Monte Carlo para el valor del portafolio con DCA."""
    cols = [c for c in pesos_dict.keys() if c in ret_activos.columns]
    
    if not cols or ret_activos[cols].shape[1] == 0:
        meses = fut_years * 12
        return np.zeros((meses, n_sim), dtype=float)

    w = np.array([pesos_dict[c] for c in cols], dtype=float)
    w = w / w.sum()

    mu = ret_activos[cols].mean().values
    cov = ret_activos[cols].cov().values
    
    meses = fut_years * 12
    valores = np.zeros((meses, n_sim), dtype=float)
    
    try:
        r_act = np.random.multivariate_normal(mean=mu, cov=cov, size=(meses, n_sim))
    except np.linalg.LinAlgError:
        # Fallback a covarianza diagonal
        st.warning(f"⚠️ Fallo en covarianza para activos: {cols}. Usando matriz diagonal.")
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

# =========================
# 2) UI LATERAL & CARGA DE DATOS
# =========================
st.sidebar.header("⚙️ Configuración de Simulación")

# INPUTS DE LA BARRA LATERAL
aporte_mensual = st.sidebar.number_input("Aporte Mensual (USD)", value=200, step=50)
hist_years = st.sidebar.slider("Años de Historia (Backtest)", 5, 20, 10)
fut_years = st.sidebar.slider("Años a Proyectar (Monte Carlo)", 5, 30, 20)
n_sim = st.sidebar.selectbox("Nº Simulaciones MC", [100, 500, 1000], index=1)

# **NUEVA CARACTERÍSTICA: MOSTRAR COMPOSICIÓN DE ACTIVOS EN SIDEBAR**
st.sidebar.markdown("---")
st.sidebar.markdown("### Composición de Portafolios (Tickers)")
for name, pesos in PORTAFOLIOS.items():
    st.sidebar.subheader(f"💼 {name}")
    df_pesos = pd.DataFrame(list(pesos.items()), columns=["Ticker", "Peso"]).set_index("Ticker")
    df_pesos["Peso"] = (df_pesos["Peso"] * 100).round(1).astype(str) + '%'
    st.sidebar.dataframe(df_pesos, use_container_width=True)
st.sidebar.markdown("---")


# Preparación de tickers
all_tickers = set()
for p in PORTAFOLIOS.values():
    all_tickers.update(p.keys())

# Carga de datos
try:
    with st.spinner("🌐 Descargando y procesando datos históricos..."):
        precios = descargar_datos(sorted(list(all_tickers)), hist_years)
    
    rets = calcular_retornos(precios)
    st.sidebar.success("✅ Datos cargados correctamente")
    
except Exception as e:
    st.sidebar.error(f"Error al descargar o procesar datos: {e}")
    st.stop()

# =========================
# 3) FRONTERA DE MARKOWITZ
# =========================

st.title("📈 Análisis de Portafolios de Inversión")
st.markdown("Visualización de Frontera Eficiente, Backtesting DCA y Monte Carlo.")

with st.expander("📊 Ver Frontera Eficiente de Markowitz", expanded=True):
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("""
        **¿Qué muestra este gráfico?**
        
        Muestra la relación **Riesgo vs. Retorno**.
        
        - **Puntos Grises:** 3,000 portafolios aleatorios generados.
        - **Puntos de Color:** Tus portafolios definidos.
        
        La curva superior (la frontera) muestra las combinaciones óptimas. Lo ideal es estar **arriba y a la izquierda** (mayor retorno con menor riesgo).
        """)
    
    with col1:
        n_portfolios_frontier = 3000
        means = rets.mean() * 12 # Anualizado
        cov_matrix = rets.cov() * 12 # Anualizado
        
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

# =========================
# 4) ANÁLISIS DETALLADO POR TABS
# =========================

tabs = st.tabs(["Conservador", "Moderado", "Agresivo", "Comparativa Final"])

results_mc_store = {}
historical_metrics = []

for i, (nombre_port, pesos) in enumerate(PORTAFOLIOS.items()):
    with tabs[i]:
        st.subheader(f"Portafolio {nombre_port}")
        
        # **NUEVA CARACTERÍSTICA: TABLA DE ACTIVOS EN LA PESTAÑA**
        st.markdown("##### Composición del Portafolio:")
        df_pesos_tab = pd.DataFrame(list(pesos.items()), columns=["Ticker", "Peso"]).set_index("Ticker")
        df_pesos_tab["Peso (%)"] = (df_pesos_tab["Peso"] * 100).round(2).astype(str) + '%'
        st.dataframe(df_pesos_tab.drop(columns=["Peso"]), use_container_width=True)
        st.markdown("---")

        rp = retornos_portafolio(rets, pesos)

        if rp.empty:
            st.warning("No se pudieron cargar datos para este portafolio con los activos y el periodo solicitados.")
            continue

        serie_dca = dca_backtest(rp, aporte_mensual)
        
        total_invertido = len(serie_dca) * aporte_mensual
        valor_final = serie_dca.iloc[-1]
        ganancia = valor_final - total_invertido
        cagr = (valor_final / total_invertido) ** (1 / (len(serie_dca)/12)) - 1
        vol_anual = rp.std() * np.sqrt(12)
        sharpe = (rp.mean() * 12) / vol_anual if vol_anual > 0 else 0
        
        historical_metrics.append({"Portafolio": nombre_port, "Retorno": cagr, "Volatilidad": vol_anual, "Sharpe": sharpe})
        
        # Mostrar KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Valor Final (Histórico)", f"${valor_final:,.0f}")
        c2.metric("Inversión Total", f"${total_invertido:,.0f}")
        c3.metric("CAGR (DCA)", f"{cagr:.2%}")
        c4.metric("Volatilidad Anual", f"{vol_anual:.2%}")
        c5.metric("Ratio Sharpe", f"{sharpe:.2f}")
        
        # Gráfico Backtest
        fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
        ax_bt.plot(serie_dca.index, serie_dca.values, label='Valor Portafolio', color='green')
        ax_bt.plot(serie_dca.index, [aporte_mensual * i for i in range(1, len(serie_dca)+1)], '--', label='Dinero Aportado', alpha=0.7)
        ax_bt.set_title(f"Backtest Histórico (DCA) - {nombre_port}")
        ax_bt.legend()
        ax_bt.grid(True, alpha=0.3)
        st.pyplot(fig_bt)
        
        st.divider()
        
        # Monte Carlo
        st.markdown(f"**Proyección Monte Carlo ({fut_years} años)**")
        with st.spinner(f"Ejecutando {n_sim} simulaciones..."):
            mc_vals = mc_trayectorias(rets, pesos, aporte_mensual, n_sim, fut_years)
            
            if mc_vals.sum() == 0 and mc_vals.size > 0:
                st.error("No se pudo realizar la simulación Monte Carlo. Verifique la disponibilidad de activos.")
                continue

            results_mc_store[nombre_port] = mc_vals[-1, :]
            
            p5 = np.percentile(mc_vals, 5, axis=1)
            p50 = np.percentile(mc_vals, 50, axis=1)
            p95 = np.percentile(mc_vals, 95, axis=1)
            
            fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
            timeline = np.arange(len(p50))
            ax_mc.plot(timeline, p50, color='blue', label='Mediana')
            ax_mc.fill_between(timeline, p5, p95, color='blue', alpha=0.2, label='Rango P5 - P95')
            ax_mc.set_title(f"Proyección de Riqueza Futura - {nombre_port}")
            ax_mc.set_xlabel("Meses Futuros")
            ax_mc.set_ylabel("Valor Portafolio (USD)")
            ax_mc.legend()
            ax_mc.grid(True, alpha=0.3)
            st.pyplot(fig_mc)
            
            # Estadísticas MC
            mediana_final = np.median(results_mc_store[nombre_port])
            peor_caso = np.percentile(results_mc_store[nombre_port], 5)
            mejor_caso = np.percentile(results_mc_store[nombre_port], 95)
            
            col_mc1, col_mc2, col_mc3 = st.columns(3)
            col_mc1.info(f"Escenario Pesimista (P5): **${peor_caso:,.0f}**")
            col_mc2.success(f"Escenario Base (Mediana): **${mediana_final:,.0f}**")
            col_mc3.warning(f"Escenario Optimista (P95): **${mejor_caso:,.0f}**")

# =========================
# 5) COMPARATIVA FINAL
# =========================
with tabs[3]:
    st.subheader("Comparación de Resultados Monte Carlo")
    
    if results_mc_store:
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        data_to_plot = [results_mc_store[k] for k in results_mc_store.keys()]
        labels = list(results_mc_store.keys())
        ax_box.boxplot(data_to_plot, labels=labels, vert=True, showfliers=False)
        ax_box.set_title(f"Distribución de Valor Final por Portafolio a {fut_years} años (MC)")
        ax_box.set_ylabel("Valor en USD")
        ax_box.grid(axis='y', alpha=0.3)
        st.pyplot(fig_box)
    else:
        st.warning("No se generaron datos de Monte Carlo. Verifique la disponibilidad de activos en cada pestaña.")

    st.divider()

    # Dataframe de Métricas Históricas
    df_radar = pd.DataFrame(historical_metrics).set_index("Portafolio")
    st.markdown("### Métricas Históricas (Backtest)")
    st.dataframe(df_radar.style.format({
        "Retorno": "{:.2%}", 
        "Volatilidad": "{:.2%}",
        "Sharpe": "{:.2f}"
    }))
    
    # **NUEVA CARACTERÍSTICA: BOTÓN DE DESCARGA (CSV)**
    if not df_radar.empty:
        csv = df_radar.to_csv(sep=';').encode('utf-8')
        st.download_button(
            label="⬇️ Descargar Métricas Históricas (CSV)",
            data=csv,
            file_name=f'informe_metricas_portafolios_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            help="Descarga el cuadro de métricas históricas de los portafolios."
        )
        
        st.markdown("""
        **💡 Sugerencia para PDF:** Si desea un informe PDF completo con todas las gráficas, use la función de **Imprimir (Ctrl + P o Cmd + P)** de su navegador y elija la opción "Guardar como PDF".
        """)
        st.divider()

    # GRÁFICO RADAR (CORREGIDO)
    st.markdown("### 🎯 Perfil de Riesgo y Rendimiento (Gráfico Radar)")

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

        # CORRECCIÓN: Usa angles[:-1] para que las etiquetas coincidan con los ejes.
        ax_radar.set_thetagrids(np.array(angles[:-1]) * 180/np.pi, indicadores, color='gray', size=11)
        
        ax_radar.set_yticks(np.arange(0.2, 1.1, 0.2))
        ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", size=8)
        ax_radar.set_ylim(0, 1)

        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        ax_radar.set_title("Perfil de Riesgo/Retorno Normalizado", size=14, pad=20)
        st.pyplot(fig_radar)
    else:
        st.warning("No hay datos suficientes para generar la comparativa del gráfico Radar.")