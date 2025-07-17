import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import warnings
from scipy.stats import norm
import altair as alt

# --- Suprime avisos de convergência do ARIMA ---
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Verifica disponibilidade do ARIMA ---
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ModuleNotFoundError:
    HAS_ARIMA = False

# --- Estilo global (Matplotlib) ---
mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (4, 3)
})

# --- Configuração da página ---
st.set_page_config(page_title="Inventory Simulation - RS Policy", layout="wide")
st.title("📦 Inventory Simulation - RS Policy")

# --- Parâmetros padrão ---
lead_time       = 4            # tempo de espera (semanas)
review_period   = 8            # período de revisão (semanas)
service_level   = 0.95         # nível de serviço (95%)
unit_price      = 40.0         # preço por litro
holding_rate    = 0.17         # custo anual de armazenagem (% do valor)

# Cenários de demanda

demand1_color   = 15000.0      # demanda cenário 1 cores (m² ou m linear)
demand2_color   = 12000.0      # cenário 2 cores
demand1_white   = 10000.0      # cenário 1 branca
demand2_white   = 8000.0       # cenário 2 branca
selected_colors = ["Cyan", "Magenta"]

# --- Configurações da barra lateral ---
unit_option = st.sidebar.radio("Demand unit:", ("m²", "linear m"), index=0)
st.sidebar.header("⚙️ Settings")
enabled_features = st.sidebar.multiselect(
    "Enable features:",
    ["Scenario Comparison", "Forecast (ARIMA)", "Parameter Optimization", "Cost Summary"],
    default=["Scenario Comparison", "Forecast (ARIMA)", "Parameter Optimization", "Cost Summary"]
)
all_colors = ["Cyan","Magenta","Yellow","Black","Green","Red","DuoSoft","White"]
selected_colors = st.sidebar.multiselect("Select ink colors:", all_colors, default=selected_colors)
if st.sidebar.button("🔄 Reset all inputs"):
    st.session_state.clear()
    st.experimental_rerun()

# --- Linha do tempo e validação ---
dates = ["Aug/25","Sep/25","Oct/25","Nov/25","Dec/25","Jan/26"]
weeks_per_month = 4
if (lead_time + review_period) > len(dates) * weeks_per_month:
    st.error(
        f"Lead time + review period ({lead_time + review_period} weeks) "
        f"exceeds horizon ({len(dates)*weeks_per_month} weeks)."
    )

# --- Inputs adicionais na barra lateral ---
start_month    = st.sidebar.selectbox("Order start month:", dates, key='start_month')
lead_time      = st.sidebar.number_input("Lead time (weeks)", 1, 52, lead_time, key='lead_time')
review_period  = st.sidebar.number_input("Review period R (weeks)", 1, 52, review_period, key='review_period')
service_level  = st.sidebar.slider("Service level (%)", 80, 99, int(service_level*100), key='service_level') / 100
unit_price     = st.sidebar.number_input("Unit price ($/L)", 0.0, 1000.0, unit_price, key='unit_price')
holding_rate   = st.sidebar.slider("Annual holding cost rate (%)", 1, 50, int(holding_rate*100), key='holding_rate') / 100
label_unit     = 'm²' if unit_option=='m²' else 'linear m'

demand1_color = st.sidebar.number_input(
    f"Demand ({label_unit}) Scenario 1 (colors)", 0.0, 1e6, demand1_color, key='d1_color'
)
demand2_color = st.sidebar.number_input(
    f"Demand ({label_unit}) Scenario 2 (colors)", 0.0, 1e6, demand2_color, key='d2_color'
)
demand1_white = st.sidebar.number_input(
    f"Demand ({label_unit}) Scenario 1 (white)", 0.0, 1e6, demand1_white, key='d1_white'
)
demand2_white = st.sidebar.number_input(
    f"Demand ({label_unit}) Scenario 2 (white)", 0.0, 1e6, demand2_white, key='d2_white'
)

# --- Função de simulação ---
def simulate(color, demand_m2):
    # 1) cálculo de uso e variabilidade
    usage        = round((demand_m2 * consumption_rates[color]) / 1000, 2)
    std_month    = usage * 0.10

    # 2) demanda no lead+review
    total_lead   = lead_time + review_period
    demand_lead  = round((usage / weeks_per_month) * total_lead, 2)
    std_lead     = round(std_month * math.sqrt(total_lead / weeks_per_month), 2)

    # 3) cálculo do estoque de segurança
    z            = norm.ppf(service_level)
    safety_stock = round(z * std_lead, 2)

    # 4) nível de ressuprimento S
    Q            = round(demand_lead + safety_stock, 2)
    S            = math.ceil(Q / 5) * 5

    # 5) definição de meses de chegada e ordem
    step         = math.floor(total_lead / weeks_per_month)
    idx          = dates.index(start_month)
    arrival_months = []
    for _ in dates:
        idx += step
        if idx < len(dates):
            arrival_months.append(dates[idx])
    offset       = math.ceil(lead_time / weeks_per_month)
    order_months = [dates[max(dates.index(m) - offset, 0)] for m in arrival_months]

    # 6) construção da tabela de estoque
    stock = initial_stock.get(color, 0)
    rows = []
    for m in dates:
        arrival     = S if m in arrival_months else 0
        start_stock = stock + arrival
        end_stock   = round(start_stock - usage, 2)
        status      = "Shortage" if end_stock < safety_stock else "OK"
        is_order    = m in order_months
        rows.append({
            "Month": m,
            "Start+Arrival": start_stock,
            "Usage": usage,
            "Arrival": arrival,
            "End Stock": end_stock,
            "Status": status,
            "Order": is_order
        })
        stock = end_stock

    df = pd.DataFrame(rows)
    df['Month'] = pd.Categorical(df['Month'], categories=dates, ordered=True)
    df = df.sort_values('Month')

    # 7) métricas de custo
    cyc_cost      = (S/2) * unit_price * holding_rate
    saf_cost      = safety_stock * unit_price * holding_rate
    inv_value     = S * unit_price
    orders_per_yr = round(12 / (total_lead / weeks_per_month), 2)

    metrics = {
        'Usage': usage,
        'Safety Stock': safety_stock,
        'S': S,
        'Cycle Cost': cyc_cost,
        'Safety Cost': saf_cost,
        'Inventory Value': inv_value,
        'Orders/yr': orders_per_yr,
        'Order Months': order_months
    }
    return df, metrics

# --- Função de plotagem (Matplotlib) ---
def plot_df(df, color, safety_stock):
    cmap = {
        'Cyan':'#00AEEF','Magenta':'#FF00FF','Yellow':'#FFFF00',
        'Black':'#444444','Green':'#00CC66','Red':'#FF4444',
        'DuoSoft':'#FFA500','White':'#FFFFFF'
    }
    fig, ax = plt.subplots()
    ax.bar(df['Month'], df['End Stock'], color='lightgray', label='End Stock')
    ax.bar(df['Month'], df['Start+Arrival'] - df['End Stock'],
           bottom=df['End Stock'], color=cmap[color], label='Start+Arrival')
    ax.axhline(safety_stock, linestyle='--', color=cmap[color], label='Safety Stock')
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45)
    ax.legend()
    return fig

# --- Inputs & Visualização ---
st.header("Color Data")
consumption_rates, initial_stock, color_prices = {}, {}, {}
cols = st.columns(3)
for i, color in enumerate(selected_colors):
    with cols[i % 3]:
        consumption_rates[color] = st.number_input(
            f"Consumption (ml/{label_unit}) - {color}", 0.0, 10.0, 3.45, key=f"cons_{color}"
