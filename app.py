# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app (Opción A) — Backtest DCA + Monte Carlo
Incluye:
- Visor por activo (precios y retornos)
- Monte Carlo y guardado de todas las trayectorias (ZIP con CSVs)
- Generación de reporte PDF con métricas y gráficos
"""
import streamlit as st
st.set_page_config(layout="wide", page_title="Backtest DCA & Monte Carlo", page_icon="📈")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from io import BytesIO
import zipfile
import os
import time

# ----------------------
# Config / Defaults
# ----------------------
os.makedirs("graficos2", exist_ok=True)
DEFAULT_HIST_YEARS = 20
DEFAULT_FUT_YEARS = 20
DEFAULT_APORTE = 200
DEFAULT_NSIM = 500    # valor por defecto razonable
DEFAULT_RF_ANUAL = 0.00

PORTAFOLIOS = {
    "Conservador": {"AGG":0.40,"IEF":0.20,"VTI":0.20,"IAU":0.10,"VIG":0.10},
    "Moderado": {"VTI":0.25,"SPY":0.15,"EFA":0.15,"VWO":0.15,"AGG":0.20,"IAU":0.05,"QQQ":0.05},
    "Agresivo": {"QQQ":0.30,"SPY":0.20,"IWM":0.15,"VGT":0.15,"VWO":0.10,"SMH":0.05,"AAPL":0.025,"NVDA":0.025}
}

# ----------------------
# Utilidades
# ----------------------
@st.cache_data(ttl=60*60*24)
def descargar_precios_mensuales(tickers, years=20):
    if isinstance(tickers, (set, tuple)):
        tickers = list(tickers)
    if len(tickers) == 0:
        return pd.DataFrame()
    data = yf.download(tickers, period=f"{years}y", interval="1mo", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"].copy()
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.sort_index().dropna(how="all", axis=0).dropna(how="all", axis=1)
    data = data.dropna(axis=1, thresh=int(0.9 * len(data)))
    data = data.dropna(axis=0)
    return data

def retornos_mensuales(precios_df):
    return precios_df.pct_change().dropna(how="any")

def retornos_portafolio(ret_activos, pesos_dict):
    cols = [c for c in pesos_dict.keys() if c in ret_activos.columns]
    if len(cols)==0:
        return pd.Series(dtype=float)
    w = np.array([pesos_dict[c] for c in cols], dtype=float)
    w = w / w.sum()
    return (ret_activos[cols] @ w)

def dca_backtest_series(rp, aporte_mensual):
    V = 0.0
    valores = []
    for r in rp.values:
        V = (V + aporte_mensual) * (1.0 + r)
        valores.append(V)
    return pd.Series(valores, index=rp.index, name="Valor_DCA"), rp.copy()

def mc_trayectorias(ret_activos, pesos_dict, aporte_mensual, n_sim=1000, fut_years=20):
    cols = [c for c in pesos_dict.keys() if c in ret_activos.columns]
    if len(cols)==0:
        return np.zeros((fut_years*12, n_sim))
    w = np.array([pesos_dict[c] for c in cols], dtype=float)
    w = w / w.sum()
    mu = ret_activos[cols].mean().values
    cov = ret_activos[cols].cov().values
    meses = fut_years * 12
    valores = np.zeros((meses, n_sim), dtype=float)
    for s in range(n_sim):
        V = 0.0
        r_acts = np.random.multivariate_normal(mean=mu, cov=cov, size=meses)
        for m in range(meses):
            r_port = float(np.dot(w, r_acts[m]))
            V = (V + aporte_mensual) * (1.0 + r_port)
            valores[m, s] = V
    return valores

def metrics_pack(serie_valor, rp, aporte_mensual, rf_anual=0.0):
    n_meses = len(serie_valor)
    valor_final = float(serie_valor.iloc[-1]) if n_meses>0 else 0.0
    aportes_totales = aporte_mensual * n_meses
    años = n_meses / 12.0
    cagr = np.nan
    if aportes_totales>0 and años>0:
        cagr = (valor_final / aportes_totales) ** (1.0 / años) - 1.0
    rf_mensual = (1 + rf_anual) ** (1/12) - 1
    vol_anual = float(rp.std() * np.sqrt(12)) if len(rp)>1 else np.nan
    sharpe = np.nan if vol_anual==0 or np.isnan(vol_anual) else (rp.mean() - rf_mensual) * 12 / vol_anual
    downside = rp[rp < rf_mensual]
    if len(downside) > 0:
        downside_std = np.std(downside, ddof=1) * np.sqrt(12)
        sortino = (rp.mean() - rf_mensual) * 12 / downside_std if downside_std>0 else np.nan
    else:
        sortino = np.nan
    cummax = serie_valor.cummax()
    drawdowns = (serie_valor / cummax) - 1.0
    max_dd = float(drawdowns.min()) if n_meses>0 else np.nan
    return {
        "Valor_Final": valor_final,
        "Aportes_Totales": aportes_totales,
        "Rent_Anualizada": cagr,
        "Vol_Anual": vol_anual,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_dd
    }

# ----------------------
# Gráficos helpers (figuras matplotlib)
# ----------------------
def fig_fan_chart(valores_mc, rets_index, nombre, aporte):
    from pandas.tseries.offsets import MonthBegin
    meses = valores_mc.shape[0]
    timeline = pd.period_range(start=rets_index[-1] + MonthBegin(1), periods=meses, freq='M').to_timestamp()
    p5 = np.percentile(valores_mc, 5, axis=1)
    p50 = np.percentile(valores_mc, 50, axis=1)
    p95 = np.percentile(valores_mc, 95, axis=1)
    fig, ax = plt.subplots(figsize=(9,4.5))
    ax.plot(timeline, p50, label='Mediana')
    ax.fill_between(timeline, p5, p95, alpha=0.25, label='P5–P95')
    ax.set_title(f"Fan chart MC — {nombre} | Aporte ${aporte}/mes")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Valor (USD)"); ax.legend()
    fig.tight_layout()
    return fig

def fig_hist_finales(finales, nombre, aporte):
    p5, p50, p95 = np.percentile(finales, [5,50,95])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(finales, bins=60, alpha=0.7)
    ax.axvline(p5, linestyle='--', label=f'P5={p5:,.0f}')
    ax.axvline(p50, linestyle='-', label=f'Mediana={p50:,.0f}')
    ax.axvline(p95, linestyle='--', label=f'P95={p95:,.0f}')
    ax.set_title(f"Distribución valores finales (MC) — {nombre}")
    ax.set_xlabel("Valor final (USD)"); ax.set_ylabel("Frecuencia"); ax.legend()
    fig.tight_layout()
    return fig

def fig_backtest_vs_aportes(serie_valor, aporte, nombre):
    aportes_acum = np.arange(1, len(serie_valor)+1) * aporte
    fig, ax = plt.subplots(figsize=(9,4.5))
    ax.plot(serie_valor.index, serie_valor.values, label='Valor DCA (histórico)')
    ax.plot(serie_valor.index, aportes_acum, linestyle='--', label='Aportes acumulados')
    ax.set_title(f"Backtest DCA — {nombre}"); ax.set_xlabel("Fecha"); ax.set_ylabel("USD"); ax.legend()
    fig.tight_layout()
    return fig

def fig_boxplot(mc_results_dict, aporte):
    data = [mc_results_dict[nombre] for nombre in mc_results_dict.keys()]
    labels = list(mc_results_dict.keys())
    fig, ax = plt.subplots(figsize=(8,4))
    ax.boxplot(data, labels=labels, vert=True, showfliers=False)
    ax.set_title(f"Comparativo MC — aporte ${aporte}/mes"); ax.set_ylabel("Valor final (USD)")
    fig.tight_layout()
    return fig

def fig_radar(resumen_df):
    from math import pi
    indicadores = ["Rent_Anualizada","Vol_Anual","Sharpe","Sortino","Max_Drawdown"]
    radar_df = resumen_df.groupby("Portafolio")[indicadores].mean()
    def normalizar(series, invertir=False):
        s = (series - series.min()) / (series.max() - series.min())
        if invertir:
            s = 1 - s
        return s
    radar_norm = pd.DataFrame({
        "Rent_Anualizada": normalizar(radar_df["Rent_Anualizada"]),
        "Vol_Anual": normalizar(radar_df["Vol_Anual"], invertir=True),
        "Sharpe": normalizar(radar_df["Sharpe"]),
        "Sortino": normalizar(radar_df["Sortino"]),
        "Max_Drawdown": normalizar(radar_df["Max_Drawdown"], invertir=True)
    })
    N = len(radar_norm.columns)
    ang = [n/float(N)*2*np.pi for n in range(N)]
    ang += ang[:1]
    fig = plt.figure(figsize=(6,6)); ax = plt.subplot(111, polar=True)
    for port in radar_norm.index:
        vals = radar_norm.loc[port].values.flatten().tolist(); vals += vals[:1]
        ax.plot(ang, vals, label=port); ax.fill(ang, vals, alpha=0.1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(radar_norm.columns); ax.set_ylim(0,1); ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    fig.tight_layout()
    return fig

# ----------------------
# UI (Opción A: menú principal con secciones)
# ----------------------
st.title("📈 Backtest DCA + Monte Carlo — Portafolios (Opción A)")
st.markdown("Menú: selecciona la sección en la barra lateral.")

section = st.sidebar.selectbox("Selecciona sección", ["Visor de series", "Monte Carlo & Análisis", "Exportar MC (ZIP)", "Generar PDF reporte"])

# Parámetros comunes
hist_years = st.sidebar.number_input("Años históricos", min_value=1, max_value=40, value=DEFAULT_HIST_YEARS)
fut_years = st.sidebar.number_input("Años a proyectar", min_value=1, max_value=40, value=DEFAULT_FUT_YEARS)
aporte = st.sidebar.number_input("Aporte mensual (USD)", min_value=1, value=DEFAULT_APORTE, step=10)
n_sim = st.sidebar.number_input("N° simulaciones (MC)", min_value=10, max_value=10000, value=DEFAULT_NSIM, step=10)
rf_anual = st.sidebar.number_input("Tasa libre de riesgo anual", value=float(DEFAULT_RF_ANUAL), format="%.4f")

# Descargar precios (para todos tickers)
todos_tickers = set()
for p in PORTAFOLIOS.values():
    todos_tickers.update(p.keys())
precios = descargar_precios_mensuales(sorted(list(todos_tickers)), years=hist_years)
if not precios.empty:
    rets = retornos_mensuales(precios)
else:
    rets = pd.DataFrame()

# ---------- Sección: Visor de series ----------
if section == "Visor de series":
    st.header("Visor por activo")
    st.write("Selecciona un ticker para ver su serie histórica y retornos mensuales.")
    ticker = st.selectbox("Ticker", options=sorted(list(precios.columns)) if not precios.empty else [])
    if ticker:
        st.subheader(f"Precios ajustados mensuales — {ticker}")
        st.line_chart(precios[ticker].dropna())
        st.subheader("Retornos mensuales (últimos 24 meses)")
        r = precios[ticker].pct_change().dropna()
        st.line_chart(r.tail(24))
        st.markdown("---")
        st.write("Tabla (últimas filas)")
        st.dataframe(precios[[ticker]].dropna().tail(12))
    else:
        st.info("No hay datos para mostrar. Revisa la conexión a internet o el período seleccionado.")

# ---------- Sección: Monte Carlo & Análisis ----------
elif section == "Monte Carlo & Análisis":
    st.header("Monte Carlo & Análisis")
    st.write("Selecciona portafolio y presiona 'Ejecutar' para generar MC y métricas.")
    chosen = st.selectbox("Portafolio", options=list(PORTAFOLIOS.keys()))
    run = st.button("Ejecutar análisis")
    if run:
        if precios.empty:
            st.error("No hay precios descargados. Revisa conexión / años históricos.")
        else:
            with st.spinner("Ejecutando backtest y Monte Carlo..."):
                pesos = PORTAFOLIOS[chosen]
                rp = retornos_portafolio(rets, pesos)
                if rp.empty or len(rp) < 2:
                    st.error("No hay suficientes retornos para este portafolio con el periodo elegido.")
                else:
                    serie_val, rp_hist = dca_backtest_series(rp, aporte)
                    met = metrics_pack(serie_val, rp_hist, aporte, rf_anual)
                    st.subheader("🔎 Métricas (Backtest histórico)")
                    st.table(pd.DataFrame(met, index=[0]).T.rename(columns={0:"Valor"}))

                    # Monte Carlo
                    n_sim_int = int(n_sim)
                    valores_mc = mc_trayectorias(rets, PORTAFOLIOS[chosen], aporte, n_sim=n_sim_int, fut_years=fut_years)
                    finales = valores_mc[-1, :]
                    st.subheader("🎲 Resultados Monte Carlo")
                    st.write(f"Simulaciones: {n_sim_int} | Años proyección: {fut_years}")
                    st.write(pd.Series(finales).describe().round(2))

                    # Figuras
                    f1 = fig_fan_chart(valores_mc, rets.index, chosen, aporte)
                    st.pyplot(f1)
                    st.write("Histograma de valores finales")
                    f2 = fig_hist_finales(finales, chosen, aporte)
                    st.pyplot(f2)
                    st.write("Backtest histórico vs Aportes")
                    f3 = fig_backtest_vs_aportes(serie_val, aporte, chosen)
                    st.pyplot(f3)

                    # Guardar MC en memoria (para exportar si se desea)
                    st.session_state.setdefault("last_mc", {})
                    st.session_state["last_mc"][chosen] = {
                        "valores_mc": valores_mc,
                        "finales": finales,
                        "met": met,
                        "serie_val": serie_val
                    }
                    st.success("Análisis completado. Puedes exportar resultados desde 'Exportar MC (ZIP)' o generar PDF.")
# ---------- Sección: Exportar MC ----------
elif section == "Exportar MC (ZIP)":
    st.header("Exportar todas las trayectorias Monte Carlo (ZIP)")
    st.write("Genera un ZIP con un CSV por portafolio con shape (meses x simulaciones).")
    # Comprobar si hay resultados en session_state; si no, calcular con parámetros actuales
    compute_if_missing = st.button("Generar/Actualizar todas las simulaciones (usar parámetros actuales)")
    if compute_if_missing:
        if precios.empty:
            st.error("No hay precios para simular.")
        else:
            with st.spinner("Calculando Monte Carlo para todos los portafolios..."):
                all_mc = {}
                for nombre, pesos in PORTAFOLIOS.items():
                    vals = mc_trayectorias(rets, pesos, aporte, n_sim=int(n_sim), fut_years=int(fut_years))
                    all_mc[nombre] = vals
                    st.write(f"- {nombre} calculado ({vals.shape[0]} meses x {vals.shape[1]} sims)")
                st.session_state["all_mc_zip"] = all_mc
                st.success("Simulaciones completas almacenadas en sesión.")
    # Si existe en sesión, permitir descarga
    if "all_mc_zip" in st.session_state:
        st.write("Se dispone de resultados listos para exportar.")
        all_mc = st.session_state["all_mc_zip"]
        # Construir ZIP en memoria
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for nombre, arr in all_mc.items():
                meses = arr.shape[0]
                df = pd.DataFrame(arr, index=pd.RangeIndex(start=1, stop=meses+1, name="Mes"))
                # columnas Sim_1 ... Sim_n
                df.columns = [f"Sim_{i+1}" for i in range(arr.shape[1])]
                csv_bytes = df.to_csv(index=True).encode('utf-8')
                zf.writestr(f"MC_{nombre}_meses{meses}_sims{arr.shape[1]}.csv", csv_bytes)
        zip_buffer.seek(0)
        st.download_button("📥 Descargar ZIP con CSVs (todas las trayectorias)", data=zip_buffer.getvalue(), file_name=f"MC_trayectorias_{datetime.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")
    else:
        st.info("No hay simulaciones en sesión. Pulsa 'Generar/Actualizar todas las simulaciones' para producir los CSVs.")

# ---------- Sección: Generar PDF ----------
elif section == "Generar PDF reporte":
    st.header("Generar reporte PDF (métricas + gráficos)")
    st.write("Genera un PDF con la información del portafolio seleccionado. Usa resultados en sesión si existen.")
    chosen = st.selectbox("Portafolio para el reporte", options=list(PORTAFOLIOS.keys()))
    gen = st.button("Generar PDF")
    if gen:
        if "last_mc" not in st.session_state or chosen not in st.session_state["last_mc"]:
            st.warning("No hay resultados en sesión para el portafolio. Ejecuta el análisis primero en 'Monte Carlo & Análisis'.")
        else:
            data = st.session_state["last_mc"][chosen]
            valores_mc = data["valores_mc"]
            finales = data["finales"]
            met = data["met"]
            serie_val = data["serie_val"]
            # Crear PDF en memoria (multiples figuras + tabla)
            pdf_buffer = BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                # Página 1: resumen texto y tabla
                fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 vertical
                ax.axis("off")
                texto = f"Reporte Monte Carlo & Backtest\nPortafolio: {chosen}\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                texto += "Métricas (Backtest histórico):\n"
                for k,v in met.items():
                    texto += f"- {k}: {v}\n"
                ax.text(0.01, 0.99, texto, va='top', fontsize=10, family='monospace')
                pdf.savefig(fig); plt.close(fig)
                # Fan chart
                pdf.savefig(fig_fan_chart(valores_mc, rets.index, chosen, aporte)); plt.close()
                # Histograma
                pdf.savefig(fig_hist_finales(finales, chosen, aporte)); plt.close()
                # Backtest
                pdf.savefig(fig_backtest_vs_aportes(serie_val, aporte, chosen)); plt.close()
                # Radar (construir resumen simple)
                # Para radar necesitamos un resumen_df con al menos este portafolio: creamos tabla ficticia con met
                resumen_df = pd.DataFrame([{**{"Portafolio": chosen}, **met}]).set_index("Portafolio")
                pdf.savefig(fig_radar(resumen_df)); plt.close()
            pdf_buffer.seek(0)
            st.download_button("📥 Descargar reporte PDF", data=pdf_buffer.getvalue(), file_name=f"Reporte_{chosen}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf")
