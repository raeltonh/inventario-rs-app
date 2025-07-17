import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress ARIMA convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import streamlit as st
import pandas as pd
import matplotlib as mpl
import math
from scipy.stats import norm
import altair as alt

# --- Data containers ---
consumption_rates = {}
initial_stock    = {}
color_prices     = {}

# --- Check for ARIMA availability ---
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ModuleNotFoundError:
    HAS_ARIMA = False

# --- Global styling ---
mpl.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.figsize": (4, 3),
})
st.set_page_config(page_title="Inventory Simulation – RS Policy", layout="wide")
st.title("📦 Inventory Simulation – RS Policy")

# --- Defaults ---
lead_time     = 4     # weeks
review_period = 8     # weeks
service_level = 0.95  # fraction
unit_price    = 40.0  # $/L
holding_rate  = 0.17  # fraction

# --- Base demands ---
base_c, base_w = 15000.0, 10000.0
var_c, var_w   = 12000.0, 8000.0

all_colors = ["Cyan","Magenta","Yellow","Black","Green","Red","DuoSoft","White"]
cmap = {
    "Cyan":"#00AEEF","Magenta":"#FF00FF","Yellow":"#FFFF00",
    "Black":"#000000","Green":"#00CC66","Red":"#FF4444",
    "DuoSoft":"#FFA500","White":"#FFFFFF"
}

# --- Sidebar settings ---
unit_opt = st.sidebar.radio("Demand unit:", ("m²","linear m"))
st.sidebar.header("⚙️ Settings")
enabled = st.sidebar.multiselect(
    "Features:",
    ["Scenario Comparison","Forecast (ARIMA)","Cost Optimization","Cost Summary"],
    default=["Scenario Comparison","Forecast (ARIMA)","Cost Optimization","Cost Summary"]
)
sel_colors = st.sidebar.multiselect("Ink colors:", all_colors, default=["Cyan","Magenta","Yellow"])
if st.sidebar.button("🔄 Reset inputs"):
    st.session_state.clear()
    st.experimental_rerun()

# --- Timeline validation ---
dates = ["Aug/25","Sep/25","Oct/25","Nov/25","Dec/25","Jan/26"]
wpm   = 4
if (lead_time + review_period) > len(dates)*wpm:
    st.error("Lead + review exceeds horizon")

# --- Common sidebar inputs ---
start_m       = st.sidebar.selectbox("Start month:", dates)
lead_time     = st.sidebar.number_input("Lead time (wk)", 1, 52, lead_time)
review_period = st.sidebar.number_input("Review period (wk)", 1, 52, review_period)
service_level = st.sidebar.slider("Service level (%)", 80, 99, int(service_level*100)) / 100
unit_price    = st.sidebar.number_input("Unit price $/L", 0.0, 1000.0, unit_price)
holding_rate  = st.sidebar.slider("Holding cost %", 1, 50, int(holding_rate*100)) / 100
lbl_unit      = unit_opt

d1c = st.sidebar.number_input(f"Demand¹ color ({lbl_unit})", 0.0, 1e6, base_c)
d2c = st.sidebar.number_input(f"Demand² color ({lbl_unit})", 0.0, 1e6, var_c)
d1w = st.sidebar.number_input(f"Demand¹ white ({lbl_unit})", 0.0, 1e6, base_w)
d2w = st.sidebar.number_input(f"Demand² white ({lbl_unit})", 0.0, 1e6, var_w)

# --- Helper to strip trailing zeros ---
def strip_zeros(x):
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return x

# --- Simulation function ---
def simulate(col, dem):
    usage    = round(dem * consumption_rates[col] / 1000, 2)
    std_mon  = usage * 0.1
    total    = lead_time + review_period
    dem_lead = round((usage / wpm) * total, 2)
    std_ld   = round(std_mon * math.sqrt(total / wpm), 2)
    z        = norm.ppf(service_level)
    safety   = round(z * std_ld, 2)
    Q        = round(dem_lead + safety, 2)
    S        = math.ceil(Q / 5) * 5

    # arrivals & orders
    step = math.floor(total / wpm)
    idx  = dates.index(start_m)
    arrs = []
    for _ in dates:
        idx += step
        if idx < len(dates):
            arrs.append(dates[idx])
    offset = math.ceil(lead_time / wpm)
    ords   = [dates[max(dates.index(m) - offset, 0)] for m in arrs]

    stock = initial_stock.get(col, 0)
    rows  = []
    for m in dates:
        a   = S if m in arrs else 0
        st0 = stock + a
        ed  = round(st0 - usage, 2)
        stat= "Shortage" if ed < safety else "OK"
        rows.append({
            "Month":     m,
            "Start+Arr": st0,
            "Usage":     usage,
            "Arrival":   a,
            "End Stock": ed,
            "Status":    stat
        })
        stock = ed

    df = pd.DataFrame(rows).round(2)
    df["Month"] = pd.Categorical(df["Month"], categories=dates, ordered=True)
    df = df.sort_values("Month")

    metrics = {
        "Usage":        usage,
        "Safety Stock": safety,
        "S":            S,
        "Cycle Cost":   (S/2) * unit_price * holding_rate,
        "Safety Cost":  safety * unit_price * holding_rate,
        "Inv Value":    S * unit_price,
        "Orders/yr":    round(12/(total/wpm), 2),
        "OrderMonths":  ords
    }
    return df, metrics

# --- Scenario Comparison ---
if "Scenario Comparison" in enabled:
    for col in sel_colors:
        st.markdown(f"## <span style='color:{cmap[col]};'>{col}</span>", unsafe_allow_html=True)
        consumption_rates[col] = st.number_input(f"Consumption (ml/{lbl_unit})", 0.0, 10.0, 3.45, key=f"c_{col}")
        initial_stock[col]     = st.number_input("Initial stock (L)", 0.0, 10000.0, 0.0, key=f"s_{col}")
        color_prices[col]      = st.number_input("Price ($/L)", 0.0, 1000.0, unit_price, key=f"p_{col}")

        df1, m1 = simulate(col, d1w if col == "White" else d1c)
        df2, m2 = simulate(col, d2w if col == "White" else d2c)

        s1 = strip_zeros(m1["S"])
        s2 = strip_zeros(m2["S"])
        st.write(f"**Order‑up‑to:** S1={s1} L   S2={s2} L")

        for tag, df, met in [("1", df1, m1), ("2", df2, m2)]:
            shorts = df.loc[df["Status"]=="Shortage", "Month"].tolist()
            if shorts:
                first = shorts[0]
                rec_month = dates[max(dates.index(first) - math.ceil(lead_time / wpm), 0)]
                st.warning(
                    f"⚠️ {col} Scenario {tag} shortage in {', '.join(shorts)}. "
                    f"Recommend {lead_time} wk before ({rec_month})."
                )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Scenario 1")
            chart1 = (
                alt.Chart(df1)
                   .mark_bar(color=cmap[col])
                   .encode(
                       x=alt.X("Month:N", sort=dates, title="Month"),
                       y=alt.Y("End Stock:Q", title="End Stock (L)"),
                       tooltip=["Month","Usage","Arrival","End Stock","Status"]
                   ).interactive()
            )
            st.altair_chart(chart1, use_container_width=True)
            st.table(df1.applymap(strip_zeros))

        with c2:
            st.subheader("Scenario 2")
            chart2 = (
                alt.Chart(df2)
                   .mark_bar(color=cmap[col])
                   .encode(
                       x=alt.X("Month:N", sort=dates, title="Month"),
                       y=alt.Y("End Stock:Q", title="End Stock (L)"),
                       tooltip=["Month","Usage","Arrival","End Stock","Status"]
                   ).interactive()
            )
            st.altair_chart(chart2, use_container_width=True)
            st.table(df2.applymap(strip_zeros))

# --- Forecast (ARIMA) consolidated ---
if "Forecast (ARIMA)" in enabled:
    st.header("📈 ARIMA Forecast (all colors)")
    idx = pd.to_datetime(dates, format="%b/%y").to_period("M").to_timestamp()
    rows = []
    for col in sel_colors:
        hist = [((d1w if col=="White" else d1c) * consumption_rates[col] / 1000) for _ in dates]
        ts   = pd.Series(hist, index=idx)
        for dt, v in ts.items():
            rows.append({"Date": dt, "Color": col, "Usage": round(v, 2)})
        if HAS_ARIMA:
            res = ARIMA(ts, order=(1,1,1)).fit()
            fc = res.get_forecast(steps=3).predicted_mean
            fidx = [idx[-1] + pd.DateOffset(months=i) for i in range(1,4)]
            for dt, v in zip(fidx, fc):
                rows.append({"Date": dt, "Color": col, "Usage": round(v, 2)})

    df_fc = pd.DataFrame(rows)
    chart = (
        alt.Chart(df_fc)
           .mark_line(point=True, strokeWidth=2)
           .encode(
               x=alt.X("Date:T", title="Date"),
               y=alt.Y("Usage:Q", title="Usage (L)"),
               color=alt.Color(
                   "Color:N",
                   scale=alt.Scale(domain=sel_colors, range=[cmap[c] for c in sel_colors]),
                   legend=alt.Legend(title="Ink Color")
               ),
               tooltip=["Date","Color","Usage"]
           ).interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# --- Cost Summary & Formulas ---
if "Cost Summary" in enabled:
    st.header("💰 Cost Summary")
    with st.expander("Summary & Formulas", expanded=False):
        cs = []
        for col in sel_colors:
            _, m = simulate(col, d1w if col=="White" else d1c)
            cs.append({
                "Color":      col,
                "Cycle Cost": f"${m['Cycle Cost']:.2f}",
                "Safety Cost":f"${m['Safety Cost']:.2f}",
                "Inv Value":  f"${m['Inv Value']:.2f}",
                "Orders/yr":  f"{m['Orders/yr']}"
            })
        st.table(pd.DataFrame(cs).set_index("Color"))
        st.markdown("""
**Formulas**  
- **Cycle Cost** = (S / 2) × unit_price × holding_rate  
- **Safety Cost** = safety_stock × unit_price × holding_rate  
- **Inventory Value** = S × unit_price  
- **Orders/yr** = 12 / ((lead_time + review_period) / 4)  

**Notes per metric**  
- **Cycle Cost:** cost to hold average cycle inventory  
- **Safety Cost:** cost of buffer inventory  
- **Inv Value:** total value of inventory at S level  
- **Orders/yr:** annual order frequency  

Holding cost reflects capital tie‑up even if no warehouse rent.  
Safety stock buffers against variability to meet service level.
""")

# --- Cost Optimization (expander) & recommendation ---
if "Cost Optimization" in enabled:
    with st.expander("🔍 Cost Optimization", expanded=False):
        Rlist = [4,8,12]
        SLlist= [0.90,0.95,0.99]
        opt = []
        for R in Rlist:
            for SL in SLlist:
                backup = (review_period, service_level)
                review_period, service_level = R, SL
                total_cost = 0
                for col in sel_colors:
                    _, m = simulate(col, d1w if col=="White" else d1c)
                    total_cost += m["Cycle Cost"] + m["Safety Cost"]
                opt.append({
                    "R (wk)":    R,
                    "SL":        f"{int(SL*100)}%",
                    "Total Cost":f"${total_cost:.2f}"
                })
                review_period, service_level = backup

        df_opt = pd.DataFrame(opt).sort_values("Total Cost").set_index("R (wk)")
        st.table(df_opt)
        best = df_opt.iloc[0]
        st.success(
            f"Optimal review period: {best.name} wk at SL={best['SL']} → {best['Total Cost']}"
        )
        st.markdown("""
**Formulas (Cost Optimization)**  
- For each combination of R and SL, **Total Cost** = Σ (Cycle Cost + Safety Cost) across colors:  
  - **Cycle Cost:** (S / 2) × unit_price × holding_rate  
  - **Safety Cost:** safety_stock × unit_price × holding_rate  

Use this to choose the review period and service level that minimize annual inventory cost.
""")
