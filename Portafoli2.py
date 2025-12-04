# -*- coding: utf-8 -*-
"""
Pipeline completo para Backtesting DCA (20 años) y Simulación Monte Carlo (10,000 trayectorias a 20 años)
con generación de gráficos: fan charts temporales, histogramas y boxplots comparativos.

Notas importantes:
- Requiere conexión a internet para descargar datos con yfinance.
- Los cálculos DCA se hacen sobre retornos ponderados del portafolio (aproximación estándar):
  V_t = (V_{t-1} + aporte) * (1 + r_port_t)  con compra al inicio de cada mes.
- Monte Carlo genera trayectorias de valor del portafolio (no por activo) usando media/covarianza mensuales históricas.
- Los ETFs elegidos tienen buen historial, pero algunos no alcanzan 20 años exactos; el código sincroniza y usa el período común.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import time

# Crear carpeta para guardar gráficos
os.makedirs("graficos2", exist_ok=True)


# =========================
# 0) PARÁMETROS GENERALES
# =========================

np.random.seed(int(time.time()))

HIST_YEARS = 20          # años de historia para backtest
FUT_YEARS = 20           # años de proyección para Monte Carlo
APORTES = [200] # USD/mes
N_SIM = 100            # número de simulaciones
REBALA_FREQ = 12         # meses (anual) — para versión por retornos no se usa explícito
RF_ANUAL = 0.00          # tasa libre de riesgo anual (para Sharpe opcional)

# =========================
# 1) TICKERS DE PORTAFOLIOS
# =========================
PORTAFOLIOS = {
    "Conservador": {
        "AGG": 0.40,
        "IEF": 0.20,
        "VTI": 0.20,
        "IAU": 0.10,
        "VIG": 0.10
    },
    "Moderado": {
        "VTI": 0.25,
        "SPY": 0.15,
        "EFA": 0.15,
        "VWO": 0.15,
        "AGG": 0.20,
        "IAU": 0.05,
        "QQQ": 0.05
    },
    "Agresivo": {
        "QQQ": 0.30,
        "SPY": 0.20,
        "IWM": 0.15,
        "VGT": 0.15,
        "VWO": 0.10,
        "SMH": 0.05,
        "AAPL": 0.025,
        "NVDA": 0.025
    }
}

# =========================
# 2) UTILIDADES
# =========================

def descargar_precios_mensuales(tickers, years=20):
    """Descarga precios mensuales ajustados (Close ajustado si auto_adjust=True).
    Devuelve un DataFrame (fechas x tickers) con la frecuencia mensual unificada.
    """
    if isinstance(tickers, (set, tuple)):
        tickers = list(tickers)
    data = yf.download(tickers, period=f"{years}y", interval="1mo", auto_adjust=True, progress=False)
    # Si devuelve Panel con columnas múltiples, tomar 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"].copy()
    # Asegurar tipo DataFrame (si un ticker → Serie)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # Eliminar filas/columnas totalmente nulas y luego sincronizar periodo común
    data = data.sort_index().dropna(how="all", axis=0).dropna(how="all", axis=1)
    # Eliminar columnas con demasiados NA (p.ej., falta de historia); quedarnos con intersección de fechas válidas
    data = data.dropna(axis=1, thresh=int(0.9*len(data)))
    data = data.dropna(axis=0)
    return data


def retornos_mensuales(precios_df):
    """Retornos simples mensuales por activo, limpiando NA."""
    rets = precios_df.pct_change().dropna(how="any")
    return rets


def retornos_portafolio(ret_activos, pesos_dict):
    """Retorno mensual del portafolio como combinación ponderada de retornos de activos."""
    cols = [c for c in pesos_dict.keys() if c in ret_activos.columns]
    w = np.array([pesos_dict[c] for c in cols], dtype=float)
    w = w / w.sum()  # normalizar si faltó alguna columna
    rp = (ret_activos[cols] @ w)
    return rp


def dca_backtest_series(rp, aporte_mensual):
    """Genera la serie de valor del portafolio bajo DCA con compra al inicio de cada mes:
    V_t = (V_{t-1} + aporte) * (1 + r_t)
    Retorna Serie del valor y Serie de retornos del portafolio (r_t).
    """
    V = 0.0
    valores = []
    for r in rp.values:
        V = (V + aporte_mensual) * (1.0 + r)
        valores.append(V)
    serie_valor = pd.Series(valores, index=rp.index, name="Valor_DCA")
    return serie_valor, rp.copy()


def cagr_desde_aportes(valor_final, aporte_mensual, n_meses):
    """CAGR relativo al capital aportado: (VF / AportesTotales)^(1/Años) - 1"""
    aportes_totales = aporte_mensual * n_meses
    años = n_meses / 12.0
    if aportes_totales <= 0 or años <= 0:
        return np.nan
    return (valor_final / aportes_totales) ** (1.0 / años) - 1.0


def max_drawdown(serie_valor):
    cummax = serie_valor.cummax()
    dd = (serie_valor / cummax) - 1.0
    return dd.min()


def metrics_pack(serie_valor, rp, aporte_mensual):
    """
    Calcula métricas clave de rendimiento y riesgo:
    - Rentabilidad anualizada (CAGR sobre aportes)
    - Volatilidad anualizada
    - Ratio Sharpe
    - Ratio Sortino
    - Máximo Drawdown
    """
    n_meses = len(serie_valor)
    valor_final = float(serie_valor.iloc[-1])
    aportes_totales = aporte_mensual * n_meses
    años = n_meses / 12.0

    # --- CAGR (Rentabilidad anualizada respecto a los aportes) ---
    cagr = np.nan
    if aportes_totales > 0 and años > 0:
        cagr = (valor_final / aportes_totales) ** (1.0 / años) - 1.0

    # --- Retornos mensuales del portafolio ---
    rf_mensual = (1 + RF_ANUAL) ** (1/12) - 1
    excess_returns = rp - rf_mensual

    # --- Volatilidad (desviación estándar anualizada) ---
    vol_anual = float(rp.std() * np.sqrt(12))

    # --- Sharpe Ratio ---
    sharpe = np.nan if vol_anual == 0 else (rp.mean() - rf_mensual) * 12 / vol_anual

    # --- Sortino Ratio ---
    downside = rp[rp < rf_mensual]
    if len(downside) > 0:
        downside_std = np.std(downside, ddof=1) * np.sqrt(12)
        sortino = (rp.mean() - rf_mensual) * 12 / downside_std
    else:
        sortino = np.nan

    # --- Máximo Drawdown ---
    cummax = serie_valor.cummax()
    drawdowns = (serie_valor / cummax) - 1.0
    max_dd = float(drawdowns.min())

    return {
        "Valor_Final": valor_final,
        "Aportes_Totales": aportes_totales,
        "Rent_Anualizada": cagr,
        "Vol_Anual": vol_anual,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_dd
    }


def mc_trayectorias(ret_activos, pesos_dict, aporte_mensual, n_sim=N_SIM, fut_years=20):
    """Monte Carlo: genera matriz (meses x simulaciones) del valor del portafolio con DCA.
    - Muestrea retornos mensuales multivariados ~ N(mu, Sigma) por activo.
    - Agrega aportes al inicio de cada mes y aplica retorno del portafolio.
    """
    cols = [c for c in pesos_dict.keys() if c in ret_activos.columns]
    w = np.array([pesos_dict[c] for c in cols], dtype=float)
    w = w / w.sum()

    mu = ret_activos[cols].mean().values       # vector media mensual
    cov = ret_activos[cols].cov().values       # covarianza mensual

    meses = fut_years * 12
    valores = np.zeros((meses, n_sim), dtype=float)
    for s in range(n_sim):
        V = 0.0
        for m in range(meses):
            r_act = np.random.multivariate_normal(mean=mu, cov=cov)
            r_port = float(np.dot(w, r_act))
            V = (V + aporte_mensual) * (1.0 + r_port)
            valores[m, s] = V
    return valores  # shape: (mes, sim)

# =========================
# 3) DESCARGA, LIMPIEZA Y ALINEACIÓN
# =========================

todos_tickers = set()
for port in PORTAFOLIOS.values():
    todos_tickers.update(port.keys())

precios = descargar_precios_mensuales(sorted(list(todos_tickers)), years=HIST_YEARS)
rets = retornos_mensuales(precios)

# =========================
# 3) DESCARGA, LIMPIEZA Y ALINEACIÓN
# =========================

todos_tickers = set()
for port in PORTAFOLIOS.values():
    todos_tickers.update(port.keys())

# Nombre del archivo donde se guardarán los precios
csv_path = "graficos2/precios_historicos.csv"

# Si el archivo ya existe, lo cargamos para ahorrar tiempo
if os.path.exists(csv_path):
    print("📂 Cargando precios desde archivo CSV local...")
    precios = pd.read_csv(csv_path, index_col=0, parse_dates=True)
else:
    print("🌐 Descargando precios desde Yahoo Finance...")
    precios = descargar_precios_mensuales(sorted(list(todos_tickers)), years=HIST_YEARS)
    # Guardar los precios descargados en CSV
    precios.to_csv(csv_path, index=True)
    print(f"✅ Datos guardados en '{csv_path}' para uso futuro.")

rets = retornos_mensuales(precios)

# =========================
# 4) BACKTEST + MONTECARLO + GRÁFICOS
# =========================

resultados_backtest = {}
resultados_mc_finales = {a: {} for a in APORTES}  # para boxplot comparativo por aporte

for nombre, pesos in PORTAFOLIOS.items():
    # Crear texto de composición de portafolio
    tickers_text = ", ".join([f"{ticker} ({peso:.0%})" for ticker, peso in pesos.items()])

    # Filtrar retornos solo de los activos disponibles
    cols_validas = [c for c in pesos.keys() if c in rets.columns]
    if len(cols_validas) < 2:
        print(f"[AVISO] Portafolio {nombre}: pocos activos disponibles tras sincronizar historia.")
    rp = retornos_portafolio(rets, pesos)

    for aporte in APORTES:
        # ---- Backtest DCA (histórico) ----
        serie_valor, rp_hist = dca_backtest_series(rp, aporte)
        met = metrics_pack(serie_valor, rp_hist, aporte)
        resultados_backtest[(nombre, aporte)] = met
        print(f"\n[{nombre}] Aporte ${aporte}/mes | Valor final: {met['Valor_Final']:.2f} | Rent. anualizada: {met['Rent_Anualizada']:.2%} | Vol anual: {met['Vol_Anual']:.2%} | MaxDD: {met['Max_Drawdown']:.2%} | Sharpe: {met['Sharpe']:.2f} | Sortino: {met['Sortino']:.2f}")

        # ---- Monte Carlo (trayectorias completas) ----
        valores_mc = mc_trayectorias(rets, pesos, aporte, n_sim=N_SIM, fut_years=FUT_YEARS)
        finales = valores_mc[-1, :]
        resultados_mc_finales[aporte][nombre] = finales

        # ====== GRÁFICOS ======
        # (1) Fan chart temporal (p5-p50-p95) de 20 años
        # ====== Fan Chart Temporal ======
        from pandas.tseries.offsets import MonthBegin

        meses = valores_mc.shape[0]
        timeline = pd.period_range(start=rets.index[-1] + MonthBegin(1),
                                periods=meses, freq='M').to_timestamp()

        p5 = np.percentile(valores_mc, 5, axis=1)
        p50 = np.percentile(valores_mc, 50, axis=1)
        p95 = np.percentile(valores_mc, 95, axis=1)

        plt.figure(figsize=(10,5))
        plt.plot(timeline, p50, label='Mediana')
        plt.fill_between(timeline, p5, p95, alpha=0.3, label='P5–P95')
        plt.title(f"Fan chart Monte Carlo (Valor proyectado) — {nombre} | Aporte ${aporte}/mes")
        plt.text(0.01, 0.97, f"Composición: {tickers_text}", transform=plt.gca().transAxes,
         fontsize=8, va='top', ha='left', color='gray')
        plt.xlabel("Fecha")
        plt.ylabel("Valor del portafolio (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graficos2/fan_{nombre}_{aporte}.png", dpi=300)
        plt.close()

        # (2) Histograma del valor final (distribución)
        q5, q50, q95 = np.percentile(finales, [5,50,95])
        plt.figure(figsize=(8,5))
        plt.hist(finales, bins=60, alpha=0.7)
        plt.axvline(q5, linestyle='--', label=f'P5={q5:,.0f}')
        plt.axvline(q50, linestyle='-', label=f'Mediana={q50:,.0f}')
        plt.axvline(q95, linestyle='--', label=f'P95={q95:,.0f}')
        plt.title(f"Distribución de valores finales (MC) — {nombre} | ${aporte}/mes")
        plt.text(0.01, 0.97, f"Composición: {tickers_text}", transform=plt.gca().transAxes,
         fontsize=8, va='top', ha='left', color='gray')
        plt.xlabel("Valor final simulado (USD)")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graficos2/hist_{nombre}_{aporte}.png", dpi=300)
        plt.close()

        # (3) Curva histórica DCA vs. aportes acumulados (backtest)
        aportes_acum = np.arange(1, len(serie_valor)+1) * aporte
        plt.figure(figsize=(10,5))
        plt.plot(serie_valor.index, serie_valor.values, label='Valor DCA (histórico)')
        plt.plot(serie_valor.index, aportes_acum, linestyle='--', label='Aportes acumulados')
        plt.title(f"Backtest DCA 20 años — {nombre} | ${aporte}/mes")
        plt.text(0.01, 0.97, f"Composición: {tickers_text}", transform=plt.gca().transAxes,
         fontsize=8, va='top', ha='left', color='gray')
        plt.xlabel("Fecha")
        plt.ylabel("USD")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graficos2/backtest_{nombre}_{aporte}.png", dpi=300)
        plt.close()

# (4) Boxplots comparativos entre portafolios por cada nivel de aporte
for aporte in APORTES:
    data = [resultados_mc_finales[aporte][nombre] for nombre in PORTAFOLIOS.keys()]
    labels = list(PORTAFOLIOS.keys())
    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=labels, vert=True, showfliers=False)
    plt.title(f"Comparativo Monte Carlo: valor final por portafolio | Aporte ${aporte}/mes")
    plt.ylabel("Valor final simulado (USD)")
    plt.tight_layout()
    plt.savefig(f"graficos2/bloxplot_{nombre}_{aporte}.png", dpi=300)
    plt.close()
    

# (5) Tabla resumen final (impresión)
resumen_df = pd.DataFrame.from_dict(resultados_backtest, orient='index')
resumen_df.index = pd.MultiIndex.from_tuples(resumen_df.index, names=["Portafolio","Aporte_mensual"])
print("\n===== RESUMEN BACKTEST DCA (Histórico) =====")
print(resumen_df.round({
    "Valor_Final": 2,
    "Aportes_Totales": 2,
    "Rent_Anualizada": 4,
    "Vol_Anual": 4,
    "Sharpe": 2,
    "Sortino": 2,
    "Max_Drawdown": 4
}))

resumen_df.to_csv("graficos2/resumen_metricas.csv", index=True)

# =========================
# 6) GRÁFICO COMPARATIVO DE RATIOS
# =========================

# Extraer métricas promedio por portafolio
resumen_simple = resumen_df.groupby("Portafolio")[["Sharpe", "Sortino", "Vol_Anual"]].mean()

# Crear figura
plt.figure(figsize=(10,6))
bar_width = 0.25
x = np.arange(len(resumen_simple))

# Barras
plt.bar(x - bar_width, resumen_simple["Sharpe"], width=bar_width, label="Sharpe", alpha=0.8)
plt.bar(x, resumen_simple["Sortino"], width=bar_width, label="Sortino", alpha=0.8)
plt.bar(x + bar_width, resumen_simple["Vol_Anual"], width=bar_width, label="Volatilidad", alpha=0.8)

# Etiquetas
plt.xticks(x, resumen_simple.index, fontsize=11)
plt.title("Comparativo de Ratios de Desempeño por Portafolio", fontsize=13, weight="bold")
plt.ylabel("Valor", fontsize=11)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Guardar gráfico
plt.savefig("graficos2/ratios_comparativos.png", dpi=300)
plt.close()

print("\n✅ Gráfico comparativo 'ratios_comparativos.png' guardado en la carpeta 'graficos2'.")

# =========================
# 7) GRÁFICO RADAR (SPIDER CHART) DE INDICADORES
# =========================
from math import pi

# Selección de indicadores para comparar
indicadores = ["Rent_Anualizada", "Vol_Anual", "Sharpe", "Sortino", "Max_Drawdown"]

# Normalizar métricas (para graficar en escala 0–1)
def normalizar(series, invertir=False):
    s = (series - series.min()) / (series.max() - series.min())
    if invertir:
        s = 1 - s  # para métricas donde un valor menor es mejor (ej. Drawdown)
    return s

# Crear DataFrame con medias por portafolio
radar_df = resumen_df.groupby("Portafolio")[indicadores].mean()

# Normalización por cada métrica
radar_norm = pd.DataFrame({
    "Rent_Anualizada": normalizar(radar_df["Rent_Anualizada"]),
    "Vol_Anual": normalizar(radar_df["Vol_Anual"], invertir=True),  # menor volatilidad es mejor
    "Sharpe": normalizar(radar_df["Sharpe"]),
    "Sortino": normalizar(radar_df["Sortino"]),
    "Max_Drawdown": normalizar(radar_df["Max_Drawdown"], invertir=True)
})

# Preparar ángulos
N = len(indicadores)
angulos = [n / float(N) * 2 * np.pi for n in range(N)]
angulos += angulos[:1]  # cerrar el círculo

# Crear figura
plt.figure(figsize=(8, 8))
plt.title("Perfil de Riesgo y Rendimiento — Radar Chart", size=14, weight="bold", pad=20)

# Dibujar cada portafolio
for i, port in enumerate(radar_norm.index):
    valores = radar_norm.loc[port].values.flatten().tolist()
    valores += valores[:1]
    plt.polar(angulos, valores, label=port, linewidth=2, alpha=0.7)
    plt.fill(angulos, valores, alpha=0.1)

# Configuración estética
plt.xticks(angulos[:-1], indicadores, color='gray', size=10)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", size=8)
plt.ylim(0, 1)
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()

# Guardar gráfico
plt.savefig("graficos2/radar_portafolios.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Gráfico radar 'radar_portafolios.png' guardado en la carpeta 'graficos2'.")

# =========================
# 8) GUARDAR MÉTRICAS DE MONTE CARLO EN CSV (VERSIÓN FINAL MEJORADA)
# =========================
mc_metrics = []

for aporte, resultados in resultados_mc_finales.items():
    for nombre, finales in resultados.items():
        # === Configuración del portafolio ===
        cols_validas = [c for c in PORTAFOLIOS[nombre].keys() if c in rets.columns]
        w = np.array([PORTAFOLIOS[nombre][c] for c in cols_validas], dtype=float)
        w = w / w.sum()

        mu = rets[cols_validas].mean().values
        cov = rets[cols_validas].cov().values
        meses = FUT_YEARS * 12

        # === Generamos trayectorias Monte Carlo completas (valores mensuales) ===
        valores_mc = np.zeros((meses, N_SIM))
        retornos_mc = np.zeros((meses, N_SIM))

        for s in range(N_SIM):
            r_act = np.random.multivariate_normal(mean=mu, cov=cov, size=meses)
            r_port = np.dot(r_act, w)
            retornos_mc[:, s] = r_port

            V = 0.0
            for m in range(meses):
                V = (V + aporte) * (1 + r_port[m])
                valores_mc[m, s] = V

        # === Cálculo de métricas ===
        rent_mensual = retornos_mc.mean()
        vol_mensual = retornos_mc.std()
        vol_anual = vol_mensual * np.sqrt(12)
        rent_anualizada = (1 + rent_mensual) ** 12 - 1

        # Sharpe Ratio
        rf_mensual = (1 + RF_ANUAL) ** (1 / 12) - 1
        sharpe = (rent_mensual - rf_mensual) * 12 / vol_anual if vol_anual > 0 else np.nan

        # Sortino Ratio
        downside = retornos_mc[retornos_mc < rf_mensual]
        if len(downside) > 1:
            downside_std = np.std(downside, ddof=1) * np.sqrt(12)
            sortino = (rent_mensual - rf_mensual) * 12 / downside_std
        else:
            sortino = np.nan

        # Max Drawdown: calcularlo para cada simulación y tomar promedio o peor caso
        drawdowns = []
        for s in range(N_SIM):
            serie = valores_mc[:, s]
            cummax = np.maximum.accumulate(serie)
            dd = (serie / cummax) - 1.0
            drawdowns.append(dd.min())
        max_dd_prom = np.mean(drawdowns)   # promedio
        max_dd_peor = np.min(drawdowns)    # peor escenario (más conservador)

        # Percentiles finales del valor simulado
        finales_array = np.array(finales)
        p5, p50, p95 = np.percentile(finales_array, [5, 50, 95])
        min_val, max_val = finales_array.min(), finales_array.max()

        # Agregar resultados
        mc_metrics.append({
            "Fecha_Ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Portafolio": nombre,
            "Aporte_mensual": aporte,
            "Rent_Anualizada": rent_anualizada,
            "Vol_Anual": vol_anual,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max_Drawdown_Prom": max_dd_prom,
            "Max_Drawdown_Peor": max_dd_peor,
            "Valor_P5": p5,
            "Valor_P50": p50,
            "Valor_P95": p95,
            "Valor_Min": min_val,
            "Valor_Max": max_val
        })

# Crear DataFrame y guardar/actualizar CSV
mc_df = pd.DataFrame(mc_metrics)
csv_mc = "graficos2/montecarlo_metricas.csv"

if os.path.exists(csv_mc):
    prev = pd.read_csv(csv_mc)
    combined = pd.concat([prev, mc_df], ignore_index=True)
    combined.to_csv(csv_mc, index=False)
else:
    mc_df.to_csv(csv_mc, index=False)

print(f"✅ Resultados del Monte Carlo guardados/actualizados en '{csv_mc}'")

def frontera_markowitz(ret_activos, pasos=50):
    mean = ret_activos.mean()
    cov = ret_activos.cov()
    n = len(mean)

    results = np.zeros((pasos, 3))  # [riesgo, retorno, sharpe]
    pesos_list = []

    for i in range(pasos):
        w = np.random.random(n)
        w /= np.sum(w)
        ret = np.dot(w, mean)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = ret / vol if vol > 0 else np.nan

        results[i] = [vol, ret, sharpe]
        pesos_list.append(w)

    return results, pesos_list

rets = retornos_mensuales(precios)

results, pesos_posibles = frontera_markowitz(rets)
vols = results[:,0] * np.sqrt(12)
rets_ann = results[:,1] * 12


plt.figure(figsize=(8,6))
plt.scatter(vols, rets_ann, alpha=0.4, label="Portafolios simulados")

# Agregar posiciones reales de tus portafolios
for nombre, pesos in PORTAFOLIOS.items():
    rp = retornos_portafolio(rets, pesos)
    vol = rp.std() * np.sqrt(12)
    ret = rp.mean() * 12
    plt.scatter(vol, ret, s=120, marker="x", label=f"{nombre}")

plt.title("Frontera Eficiente de Markowitz (Risk vs Return)")
plt.xlabel("Volatilidad Anualizada")
plt.ylabel("Retorno anual esperado")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graficos2/frontera_markowitz.png", dpi=300)
plt.close()




