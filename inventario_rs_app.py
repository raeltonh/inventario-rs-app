import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from scipy.stats import norm

# --- Global Styling ---
mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (4, 3)
})

# --- Page Configuration ---
st.set_page_config(page_title="Inventory Simulation - RS Policy", layout="wide")
st.title("📦 Inventory Simulation - RS Policy")

# --- Default Parameters ---
lead_time = 4
review_period = 8
service_level = 0.95
unit_price = 40.0
holding_rate = 0.17

# Scenario demands
demand1_color = 15000.0
demand2_color = 12000.0
demand1_white = 10000.0
demand2_white = 8000.0
selected_colors = ["Cyan", "Magenta"]

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")

# Feature toggles
enabled_features = st.sidebar.multiselect(
    "Enable features:",
    ["Scenario Comparison", "Forecast (ARIMA)", "Parameter Optimization", "Cost Summary"],
    default=["Scenario Comparison", "Forecast (ARIMA)", "Parameter Optimization", "Cost Summary"]
)

# Color selection
all_colors = ["Cyan","Magenta","Yellow","Black","Green","Red","DuoSoft","White"]
selected_colors = st.sidebar.multiselect(
    "Select ink colors:", all_colors, default=selected_colors
)

# Reset
if st.sidebar.button("🔄 Reset all inputs"):
    st.session_state.clear()
    st.experimental_rerun()

# --- Input Parameters ---
dates = ["Aug/25","Sep/25","Oct/25","Nov/25","Dec/25","Jan/26"]
weeks_per_month = 4

start_month = st.sidebar.selectbox("Order start month:", dates)
lead_time = st.sidebar.number_input("Lead time (weeks)", 1, 52, lead_time)
review_period = st.sidebar.number_input("Review period R (weeks)", 1, 52, review_period)
service_level = st.sidebar.slider("Service level (%)", 80, 99, int(service_level*100)) / 100
unit_price = st.sidebar.number_input("Unit price ($/L)", 0.0, 1000.0, unit_price)
holding_rate = st.sidebar.slider("Annual holding cost rate (%)", 1, 50, int(holding_rate*100)) / 100

demand1_color = st.sidebar.number_input("Demand (m²) Scenario 1 (colors)", 0.0, 1e6, demand1_color)
demand2_color = st.sidebar.number_input("Demand (m²) Scenario 2 (colors)", 0.0, 1e6, demand2_color)
demand1_white = st.sidebar.number_input("Demand (m²) Scenario 1 (white)", 0.0, 1e6, demand1_white)
demand2_white = st.sidebar.number_input("Demand (m²) Scenario 2 (white)", 0.0, 1e6, demand2_white)

# --- Simulation ---
def simulate(color, demand):
    # Usage in liters/month
    usage = (demand * consumption_rates[color]) / 1000
    std_month = usage * 0.10

    # Demand over lead+review
    total_lead = lead_time + review_period
    demand_lead = (usage / weeks_per_month) * total_lead
    std_lead = std_month * math.sqrt(total_lead / weeks_per_month)

    # Safety stock
    z = norm.ppf(service_level)
    safety_stock = z * std_lead

    # Order-up-to level S
    Q = demand_lead + safety_stock
    if math.isnan(Q): Q = 0
    S = math.ceil(Q / 5) * 5

    # Inventory cycles
    step = math.ceil(total_lead / weeks_per_month)
    idx = dates.index(start_month)
    arrivals = []
    for _ in dates:
        idx += step
        if idx < len(dates): arrivals.append(dates[idx])
    offset = math.ceil(lead_time / weeks_per_month)
    orders = [dates[max(dates.index(m) - offset, 0)] for m in arrivals]

    stock = initial_stock.get(color, 0)
    rows = []
    for m in dates:
        arrival_qty = S if m in arrivals else 0
        start = stock + arrival_qty
        end = start - usage
        status = "Shortage" if end < safety_stock else "OK"
        rows.append({
            "Month": m,
            "Start+Arrival": start,
            "Usage": usage,
            "Arrival": arrival_qty,
            "End Stock": end,
            "Status": status,
            "Order": m in orders
        })
        stock = end
    df = pd.DataFrame(rows)

    # Cost metrics
    cyc_cost = (S/2) * unit_price * holding_rate
    saf_cost = safety_stock * unit_price * holding_rate
    inv_value = S * unit_price
    orders_per_year = 12 / (total_lead / weeks_per_month)

    return df, {
        'Usage': usage,
        'Safety Stock': safety_stock,
        'S': S,
        'Cycle Cost': cyc_cost,
        'Safety Cost': saf_cost,
        'Inventory Value': inv_value,
        'Orders/yr': orders_per_year
    }

# --- Plot ---
def plot_df(df, color, safety_stock):
    # Colored bars matching ink color
    cmap = {
        'Cyan':'#00AEEF','Magenta':'#FF00FF','Yellow':'#FFFF00',
        'Black':'#444444','Green':'#00CC66','Red':'#FF4444',
        'DuoSoft':'#FFA500','White':'#FFFFFF'
    }
    fig, ax = plt.subplots()
    bar_color = cmap.get(color, '#888888')
    ax.bar(df['Month'], df['End Stock'], color='lightgray', label='End Stock')
    ax.bar(df['Month'], df['Start+Arrival'] - df['End Stock'], bottom=df['End Stock'], color=bar_color, label='Start+Arrival')
    ax.axhline(safety_stock, linestyle='--', color=bar_color, label='Safety Stock')
    ax.legend()
    return fig

# --- Inputs & Visualization ---
st.header("Color Data")
consumption_rates = {}
initial_stock = {}
color_prices = {}
cols = st.columns(3)
for i, color in enumerate(selected_colors):
    with cols[i % 3]:
        consumption_rates[color] = st.number_input(f"Consumption (ml/m²) - {color}", 0.0, 10.0, 3.45)
        # Display calculated order-up-to levels (S) for each scenario
        _, m1 = simulate(color, demand1_white if color=='White' else demand1_color)
        _, m2 = simulate(color, demand2_white if color=='White' else demand2_color)
        st.markdown(f"**Order-up-to (S) Scenario 1:** {m1['S']} L   **Scenario 2:** {m2['S']} L")
        initial_stock[color] = st.number_input(f"Initial stock (L) - {color}", 0.0, 10000.0, 0.0)
        color_prices[color] = st.number_input(f"Price ($/L) - {color}", 0.0, 1000.0, unit_price)

# --- Cost Summary ---
if "Cost Summary" in enabled_features:
    st.header("Cost Summary")
    with st.expander("Summary & Formulas"):
        rows = []
        for color in selected_colors:
            _, m = simulate(color, demand1_white if color=='White' else demand1_color)
            rows.append({
                'Color': color,
                'Cycle Cost ($)': f"${m['Cycle Cost']:.2f}",
                'Safety Cost ($)': f"${m['Safety Cost']:.2f}",
                'Inv Value ($)': f"${m['Inventory Value']:.2f}",
                'Orders/yr': f"{m['Orders/yr']:.1f}",
                'Fill rate': f"{service_level*100:.0f}%"
            })
        st.table(pd.DataFrame(rows))
        st.markdown("""
**Formulas:**
- Cycle Cost = (S/2) × unit_price × holding_rate
- Safety Cost = safety_stock × unit_price × holding_rate
- Inv Value  = S × unit_price
- Orders/yr = 12 / ((lead_time + review_period)/4)
- Fill rate = service_level

**What each metric means:**
- **Cycle Cost:** annual cost to hold average cycle inventory.
- **Safety Cost:** annual cost to maintain safety stock.
- **Inv Value:** total value of inventory at level S.
- **Orders/yr:** estimated number of orders per year.
- **Fill rate:** probability of meeting demand without stockouts.

**Note:** Holding cost applies even without warehouse rent; it reflects financial cost of capital.
""")

# --- Scenario Comparison & Charts ---
if "Scenario Comparison" in enabled_features:
    for color in selected_colors:
        st.subheader(f"{color} Scenarios")
        df1, m1 = simulate(color, demand1_white if color=='White' else demand1_color)
        df2, m2 = simulate(color, demand2_white if color=='White' else demand2_color)
        # Display both scenarios side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Scenario 1")
            st.dataframe(df1)
            st.pyplot(plot_df(df1, color, m1['Safety Stock']))
            # Order notification for Scenario 1
            next_orders1 = [row['Month'] for _, row in df1.iterrows() if row['Order']]
            if next_orders1:
                st.info(f"🛒 Place order of **{m1['S']} L** in **{next_orders1[0]}** to avoid stockout.")
        with col2:
            st.subheader("Scenario 2")
            st.dataframe(df2)
            st.pyplot(plot_df(df2, color, m2['Safety Stock']))
            # Order notification for Scenario 2
            next_orders2 = [row['Month'] for _, row in df2.iterrows() if row['Order']]
            if next_orders2:
                st.info(f"🛒 Place order of **{m2['S']} L** in **{next_orders2[0]}** to avoid stockout.")

# --- Parameter Optimization ---
if "Parameter Optimization" in enabled_features:
    st.header("Parameter Optimization")
    R_vals = [4, 8, 12]
    SL_vals = [0.90, 0.95, 0.99]
    opt = []
    for R in R_vals:
        for SL in SL_vals:
            prevR, prevSL = review_period, service_level
            review_period, service_level = R, SL
            total = 0
            for color in selected_colors:
                _, m = simulate(color, demand1_white if color=='White' else demand1_color)
                total += m['Cycle Cost'] + m['Safety Cost']
            opt.append({'R (wk)': R, 'SL': f"{int(SL*100)}%", 'Total Cost ($)': total})
            review_period, service_level = prevR, prevSL
    # Display results and formulas in expander
    with st.expander("Optimization Results & Formulas"):
        df_opt = pd.DataFrame(opt).sort_values('Total Cost ($)')
        st.table(df_opt)
        st.markdown("""
**Formulas:**
- Total Cost = Cycle Cost + Safety Cost
- Cycle Cost = (S/2) × unit_price × holding_rate
- Safety Cost = safety_stock × unit_price × holding_rate

**What each metric means:**
- **R (wk):** review period in weeks.
- **SL:** service level as percentage.
- **Total Cost:** sum of cycle and safety stock costs per year.
""")
