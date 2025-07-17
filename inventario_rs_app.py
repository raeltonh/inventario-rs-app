import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import warnings
from scipy.stats import norm
import altair as alt

# --- Suppress ARIMA convergence warnings ---
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Check for ARIMA availability ---
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ModuleNotFoundError:
    HAS_ARIMA = False

# --- Global Matplotlib styling ---
mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (4, 3)
})

# --- Page configuration ---
st.set_page_config(page_title="Inventory Simulation - RS Policy", layout="wide")
st.title("📦 Inventory Simulation - RS Policy")

# --- Default parameters ---
lead_time = 4  # weeks
review_period = 8  # weeks
service_level = 0.95  # 95%
unit_price = 40.0  # $ per L
holding_rate = 0.17  # Annual holding cost rate

# Base demands
base_demand_color = 15000.0
base_demand_white = 10000.0
variation_color = 12000.0
variation_white = 8000.0
selected_colors = ["Cyan", "Magenta"]

# --- Sidebar ---
unit_option = st.sidebar.radio("Demand unit:", ("m²", "linear m"), index=0)
st.sidebar.header("⚙️ Settings")
enabled_features = st.sidebar.multiselect(
    "Enable features:",
    ["Scenario Comparison", "Forecast (ARIMA)", "Parameter Optimization", "Cost Summary"],
    default=["Scenario Comparison", "Forecast (ARIMA)", "Parameter Optimization", "Cost Summary"]
)
all_colors = ["Cyan","Magenta","Yellow","Black","Green","Red","DuoSoft","White"]
selected_colors = st.sidebar.multiselect("Select ink colors:", all_colors, default=selected_colors)
if st.sidebar.button("🔄 Reset inputs"):
    st.session_state.clear()
    st.experimental_rerun()

# --- Timeline ---
dates = ["Aug/25","Sep/25","Oct/25","Nov/25","Dec/25","Jan/26"]
weeks_per_month = 4
if (lead_time + review_period) > len(dates) * weeks_per_month:
    st.error(f"Lead + review ({lead_time+review_period} wk) exceeds horizon ({len(dates)*weeks_per_month} wk)")

# --- Sidebar input values ---
start_month = st.sidebar.selectbox("Start order month:", dates, key='start_month')
lead_time = st.sidebar.number_input("Lead time (weeks)", 1, 52, lead_time)
review_period = st.sidebar.number_input("Review period (weeks)", 1, 52, review_period)
service_level = st.sidebar.slider("Service level (%)", 80, 99, int(service_level*100)) / 100
unit_price = st.sidebar.number_input("Price ($/L)", 0.0, 1000.0, unit_price)
holding_rate = st.sidebar.slider("Holding cost rate (%)", 1, 50, int(holding_rate*100)) / 100
label_unit = 'm²' if unit_option == 'm²' else 'linear m'

d1c = st.sidebar.number_input(f"Demand ({label_unit}) Scenario1 color", 0.0, 1e6, base_demand_color)
d2c = st.sidebar.number_input(f"Demand ({label_unit}) Scenario2 color", 0.0, 1e6, variation_color)
d1w = st.sidebar.number_input(f"Demand ({label_unit}) Scenario1 white", 0.0, 1e6, base_demand_white)
d2w = st.sidebar.number_input(f"Demand ({label_unit}) Scenario2 white", 0.0, 1e6, variation_white)

# --- Simulation ---
consumption_rates = {}
initial_stock = {}

def simulate(color, demand):
    usage = round(demand * consumption_rates[color] / 1000, 2)
    std_mon = usage * 0.10
    total_lead = lead_time + review_period
    lead_demand = round(usage/weeks_per_month * total_lead, 2)
    std_lead = round(std_mon * math.sqrt(total_lead/weeks_per_month), 2)
    z = norm.ppf(service_level)
    safety = round(z * std_lead, 2)
    Q = round(lead_demand + safety, 2)
    S = math.ceil(Q/5)*5

    # schedule
    step = math.floor(total_lead/weeks_per_month)
    idx = dates.index(start_month)
    arrivals = []
    for _ in dates:
        idx += step
        if idx < len(dates): arrivals.append(dates[idx])
    offset = math.ceil(lead_time/weeks_per_month)
    orders = [dates[max(dates.index(m)-offset, 0)] for m in arrivals]

    stock = initial_stock.get(color, 0)
    rows = []
    for m in dates:
        arr = S if m in arrivals else 0
        start = stock + arr
        end = round(start-usage,2)
        status = 'Shortage' if end< safety else 'OK'
        rows.append({
            'Month':m,'Start+Arrival':start,
            'Usage':usage,'Arrival':arr,
            'End Stock':end,'Status':status,
            'Order':m in orders
        })
        stock = end

    df = pd.DataFrame(rows)
    df['Month'] = pd.Categorical(df['Month'], categories=dates, ordered=True)
    df = df.sort_values('Month')

    metrics = {
        'Usage':usage,'Safety Stock':safety,
        'S':S,'Cycle Cost':(S/2)*unit_price*holding_rate,
        'Safety Cost':safety*unit_price*holding_rate,
        'Inventory Value':S*unit_price,
        'Orders/yr':round(12/(total_lead/weeks_per_month),2),
        'Order Months':orders
    }
    return df, metrics

# --- Plot helper ---
def plot_df(df,color,safety):
    cmap={'Cyan':'#00AEEF','Magenta':'#FF00FF','Yellow':'#FFFF00',
          'Black':'#444444','Green':'#00CC66','Red':'#FF4444','DuoSoft':'#FFA500','White':'#FFFFFF'}
    fig,ax=plt.subplots()
    ax.bar(df['Month'],df['End Stock'],color='lightgray')
    ax.bar(df['Month'],df['Start+Arrival']-df['End Stock'],
           bottom=df['End Stock'],color=cmap[color])
    ax.axhline(safety,linestyle='--',color=cmap[color])
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates,rotation=45)
    return fig

# --- UI & Component rendering ---
st.header('Color Data')
cols=st.columns(3)
for i,c in enumerate(selected_colors):
    with cols[i%3]:
        consumption_rates[c]=st.number_input(f'Consumption (ml/{label_unit}) - {c}',0.0,10.0,3.45)
        df1,m1=simulate(c,d1w if c=='White' else d1c)
        df2,m2=simulate(c,d2w if c=='White' else d2c)
        st.markdown(f'**S1:**{m1["S"]}L  **S2:**{m2["S"]}L')
        initial_stock[c]=st.number_input(f'Initial stock (L)-{c}',0.0,1e4,0.0)
        color_prices[c]=st.number_input(f'Price ($/L)-{c}',0.0,1e3,unit_price)
