import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Logistics — Route Optimiser",
    layout="wide",
    page_icon="🚛",
)


# ----------------------------------------------------------------------------
# Solver: Vogel's Approximation Method + MODI optimization
# ----------------------------------------------------------------------------

def vogel_initial_solution(supply, demand, cost):
    m, n = len(supply), len(demand)
    supply, demand = list(supply), list(demand)
    cost = cost.astype(float).copy()
    alloc = np.zeros((m, n))
    row_done, col_done = [False] * m, [False] * n

    def penalty(values):
        values = sorted(values)
        if len(values) == 0: return -1
        if len(values) == 1: return values[0]
        return values[1] - values[0]

    active_rows, active_cols = m, n
    while active_rows > 0 and active_cols > 0:
        best_pen, best_type, best_idx = -1, None, -1
        for i in range(m):
            if row_done[i]: continue
            p = penalty([cost[i, j] for j in range(n) if not col_done[j]])
            if p > best_pen: best_pen, best_type, best_idx = p, "row", i
        for j in range(n):
            if col_done[j]: continue
            p = penalty([cost[i, j] for i in range(m) if not row_done[i]])
            if p > best_pen: best_pen, best_type, best_idx = p, "col", j
        if best_type == "row":
            i = best_idx
            j = min((j for j in range(n) if not col_done[j]), key=lambda j: cost[i, j])
        else:
            j = best_idx
            i = min((i for i in range(m) if not row_done[i]), key=lambda i: cost[i, j])
        qty = min(supply[i], demand[j])
        alloc[i, j] += qty; supply[i] -= qty; demand[j] -= qty
        if supply[i] == 0 and not row_done[i]: row_done[i] = True; active_rows -= 1
        if demand[j] == 0 and not col_done[j]: col_done[j] = True; active_cols -= 1
    return alloc


def compute_uv(alloc, cost):
    m, n = alloc.shape
    u, v = [None] * m, [None] * n
    u[0] = 0
    for _ in range(2 * (m + n)):
        changed = False
        for i in range(m):
            for j in range(n):
                if alloc[i, j] > 0:
                    if u[i] is not None and v[j] is None: v[j] = cost[i, j] - u[i]; changed = True
                    elif v[j] is not None and u[i] is None: u[i] = cost[i, j] - v[j]; changed = True
        if not changed: break
    return [0 if x is None else x for x in u], [0 if x is None else x for x in v]


def find_loop(alloc, enter_cell):
    m, n = alloc.shape
    basic_cells = [(i, j) for i in range(m) for j in range(n) if alloc[i, j] > 0]
    basic_cells.append(enter_cell)
    def neighbors(cell):
        return [c for c in basic_cells if (c[0]==cell[0] and c[1]!=cell[1]) or (c[1]==cell[1] and c[0]!=cell[0])]
    path = [enter_cell]
    def dfs(cell, direction):
        if len(path) > 3 and cell == enter_cell: return True
        for nb in neighbors(cell):
            if nb in path[1:]: continue
            next_dir = "row" if nb[0] == cell[0] else "col"
            if next_dir == direction: continue
            path.append(nb)
            if dfs(nb, next_dir): return True
            path.pop()
        return False
    if not dfs(enter_cell, "col"): return None
    return path[:-1]


def modi_optimize(alloc, cost, max_iter=200):
    alloc = alloc.copy()
    m, n = alloc.shape
    for _ in range(max_iter):
        u, v = compute_uv(alloc, cost)
        enter_cell, min_d = None, 0
        for i in range(m):
            for j in range(n):
                if alloc[i, j] == 0:
                    d = cost[i, j] - u[i] - v[j]
                    if d < min_d - 1e-9: min_d, enter_cell = d, (i, j)
        if enter_cell is None: break
        loop = find_loop(alloc, enter_cell)
        if loop is None: break
        theta = min(alloc[c] for c in loop[1::2])
        for k, cell in enumerate(loop):
            alloc[cell] += theta if k % 2 == 0 else -theta
    return alloc


def solve_transportation(supply, demand, cost):
    supply, demand, cost = list(supply), list(demand), cost.copy()
    total_supply, total_demand = sum(supply), sum(demand)
    note = None
    if total_supply > total_demand:
        demand = demand + [total_supply - total_demand]
        cost = np.hstack([cost, np.zeros((cost.shape[0], 1))])
        note = f"Dummy destination added (cost 0): supply exceeds demand by {total_supply - total_demand:.0f} bags."
    elif total_demand > total_supply:
        supply = supply + [total_demand - total_supply]
        cost = np.vstack([cost, np.zeros((1, cost.shape[1]))])
        note = f"Dummy source added (cost 0): demand exceeds supply by {total_demand - total_supply:.0f} bags."
    initial = vogel_initial_solution(supply, demand, cost)
    optimal = modi_optimize(initial, cost)
    total_cost = float((optimal * cost).sum())
    return optimal, total_cost, note


# ----------------------------------------------------------------------------
# Default data
# ----------------------------------------------------------------------------

OPERATING_COSTS = np.array([
    [31.83, 19.59, 119.98, 10.61],
    [27.75, 28.57, 129.78,  8.98],
    [86.52, 37.55,  55.50, 65.30],
], dtype=float)

COMMERCIAL_COSTS = np.array([
    [ 52.65,  32.40, 198.45,  17.55],
    [ 45.90,  47.25, 214.65,  14.85],
    [143.10,  62.10,  91.80, 108.00],
], dtype=float)

DEFAULT_SOURCES = ["Githuirai", "Marikiti", "Makuyu"]
DEFAULT_DESTS   = ["AGHS", "MHS", "NHS", "PGHS"]
DEFAULT_SUPPLY  = [51, 121, 30]
DEFAULT_DEMAND  = [50, 59, 45, 48]


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

def init_state():
    if "op_df" not in st.session_state:
        df = pd.DataFrame(OPERATING_COSTS, index=DEFAULT_SOURCES, columns=DEFAULT_DESTS)
        df.insert(0, "Supply (bags)", DEFAULT_SUPPLY)
        st.session_state.op_df = df
    if "com_df" not in st.session_state:
        df = pd.DataFrame(COMMERCIAL_COSTS, index=DEFAULT_SOURCES, columns=DEFAULT_DESTS)
        df.insert(0, "Supply (bags)", DEFAULT_SUPPLY)
        st.session_state.com_df = df
    if "demand_df" not in st.session_state:
        st.session_state.demand_df = pd.DataFrame(
            [DEFAULT_DEMAND], columns=DEFAULT_DESTS, index=["Demand (bags)"]
        )

init_state()


# ----------------------------------------------------------------------------
# Global styles + hero header with truck SVG watermark
# ----------------------------------------------------------------------------

st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700&family=Barlow+Condensed:wght@700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Barlow', sans-serif;
  }

  /* ── Page background ── */
  .stApp {
      background-color: #0f2b1a;
  }

  /* ── Hero banner ── */
  .hero {
      position: relative;
      background: linear-gradient(120deg, #0f2b1a 0%, #1a4a28 60%, #0f2b1a 100%);
      border-bottom: 3px solid #2ecc71;
      padding: 2.4rem 2.8rem 2rem;
      overflow: hidden;
      margin-bottom: 2rem;
  }
  .hero-truck {
      position: absolute;
      right: -20px;
      bottom: -10px;
      opacity: 0.10;
      width: 560px;
  }
  .hero-eyebrow {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #2ecc71;
      margin-bottom: 0.4rem;
  }
  .hero-title {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 2.6rem;
      font-weight: 800;
      color: #ffffff;
      line-height: 1.05;
      margin: 0 0 0.6rem;
  }
  .hero-title span { color: #2ecc71; }
  .hero-sub {
      font-size: 0.92rem;
      color: #a8c9b0;
      max-width: 560px;
      line-height: 1.5;
  }

  /* ── Section labels ── */
  .section-label {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #2ecc71;
      margin: 2rem 0 0.4rem;
  }
  .section-title {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.4rem;
      font-weight: 700;
      color: #e8f5ec;
      margin: 0 0 1rem;
      border-left: 4px solid #2ecc71;
      padding-left: 0.7rem;
  }

  /* ── Cards wrapping data editors ── */
  .card {
      background: #162d1e;
      border: 1px solid #254d34;
      border-radius: 8px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1.2rem;
  }
  .card-title {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1rem;
      font-weight: 700;
      color: #2ecc71;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 0.6rem;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
      background: #162d1e;
      border: 1px solid #2ecc71;
      border-radius: 8px;
      padding: 1rem 1.2rem !important;
  }
  [data-testid="stMetricLabel"] { color: #a8c9b0 !important; font-size: 0.8rem !important; }
  [data-testid="stMetricValue"] { color: #2ecc71 !important; font-size: 1.8rem !important; font-family: 'Barlow Condensed', sans-serif !important; font-weight: 700 !important; }

  /* ── Primary button ── */
  .stButton > button[kind="primary"] {
      background: #2ecc71 !important;
      color: #0f2b1a !important;
      font-family: 'Barlow Condensed', sans-serif !important;
      font-weight: 700 !important;
      font-size: 1.1rem !important;
      letter-spacing: 0.06em !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 0.6rem 2.4rem !important;
  }
  .stButton > button[kind="primary"]:hover {
      background: #27ae60 !important;
  }
  .stButton > button:not([kind="primary"]) {
      background: transparent !important;
      border: 1px solid #2ecc71 !important;
      color: #2ecc71 !important;
      font-family: 'Barlow', sans-serif !important;
      border-radius: 6px !important;
  }

  /* ── Radio ── */
  .stRadio label { color: #c8e6c9 !important; }
  .stRadio [data-testid="stMarkdownContainer"] p { color: #a8c9b0 !important; }

  /* ── Dataframe / table ── */
  [data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

  /* ── Info / success banners ── */
  .stAlert { border-radius: 6px !important; }

  /* ── Caption / footnote ── */
  .stCaption p { color: #6a9a78 !important; font-size: 0.78rem !important; }

  /* ── Divider ── */
  hr { border-color: #254d34 !important; margin: 2rem 0 !important; }

  /* ── Sidebar (if used) ── */
  section[data-testid="stSidebar"] { background: #0c2215 !important; }

  /* ── General text ── */
  p, li, span { color: #c8e6c9; }
  h1, h2, h3 { color: #e8f5ec; }
</style>

<!-- ── Hero banner ── -->
<div class="hero">
  <!-- Truck SVG watermark -->
  <svg class="hero-truck" viewBox="0 0 640 320" xmlns="http://www.w3.org/2000/svg" fill="#2ecc71">
    <!-- Trailer box -->
    <rect x="20" y="80" width="400" height="160" rx="6"/>
    <!-- Cab -->
    <path d="M420 140 L420 240 L580 240 L580 160 L540 140 Z"/>
    <!-- Cab window -->
    <rect x="460" y="150" width="80" height="50" rx="4" fill="#0f2b1a" opacity="0.6"/>
    <!-- Exhaust pipe -->
    <rect x="535" y="110" width="12" height="35" rx="3"/>
    <!-- Front bumper -->
    <rect x="570" y="220" width="30" height="20" rx="3"/>
    <!-- Wheels -->
    <circle cx="100" cy="250" r="32" fill="#0f2b1a" stroke="#2ecc71" stroke-width="10"/>
    <circle cx="100" cy="250" r="10" fill="#2ecc71"/>
    <circle cx="280" cy="250" r="32" fill="#0f2b1a" stroke="#2ecc71" stroke-width="10"/>
    <circle cx="280" cy="250" r="10" fill="#2ecc71"/>
    <circle cx="460" cy="250" r="28" fill="#0f2b1a" stroke="#2ecc71" stroke-width="10"/>
    <circle cx="460" cy="250" r="9" fill="#2ecc71"/>
    <circle cx="540" cy="250" r="28" fill="#0f2b1a" stroke="#2ecc71" stroke-width="10"/>
    <circle cx="540" cy="250" r="9" fill="#2ecc71"/>
    <!-- Road line -->
    <rect x="0" y="284" width="640" height="6" rx="2" opacity="0.4"/>
    <rect x="60" y="284" width="60" height="6" rx="2" opacity="0.2"/>
    <rect x="200" y="284" width="60" height="6" rx="2" opacity="0.2"/>
    <rect x="360" y="284" width="60" height="6" rx="2" opacity="0.2"/>
    <rect x="500" y="284" width="60" height="6" rx="2" opacity="0.2"/>
  </svg>

  <div class="hero-eyebrow">🚛 Logistics Company</div>
  <div class="hero-title">Route Cost <span>Optimiser</span></div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Section 1 — Cost matrices & supply/demand
# ----------------------------------------------------------------------------

st.markdown('<div class="section-label">Step 01</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Cost matrices, supply & demand</div>', unsafe_allow_html=True)

# --- Operating Cost ---
st.markdown('<div class="card"><div class="card-title">🔧 Operating cost per bag (KES)</div>', unsafe_allow_html=True)
edited_op = st.data_editor(
    st.session_state.op_df,
    num_rows="fixed",
    use_container_width=True,
    key="op_editor",
)
st.session_state.op_df = edited_op
st.markdown('</div>', unsafe_allow_html=True)

# --- Commercial Freight Cost ---
st.markdown('<div class="card"><div class="card-title">🚛 Commercial freight cost per bag (KES)</div>', unsafe_allow_html=True)
edited_com = st.data_editor(
    st.session_state.com_df,
    num_rows="fixed",
    use_container_width=True,
    key="com_editor",
)
st.session_state.com_df = edited_com
st.markdown('</div>', unsafe_allow_html=True)

# --- Demand ---
st.markdown('<div class="card"><div class="card-title">📦 Weekly demand per delivery point (bags)</div>', unsafe_allow_html=True)
edited_demand = st.data_editor(
    st.session_state.demand_df,
    num_rows="fixed",
    use_container_width=True,
    key="demand_editor",
)
st.session_state.demand_df = edited_demand
st.markdown('</div>', unsafe_allow_html=True)

# Derive arrays
supply    = st.session_state.op_df["Supply (bags)"].tolist()
demand    = st.session_state.demand_df.iloc[0].tolist()
op_costs  = st.session_state.op_df.drop(columns=["Supply (bags)"]).to_numpy(dtype=float)
com_costs = st.session_state.com_df.drop(columns=["Supply (bags)"]).to_numpy(dtype=float)
dests     = list(st.session_state.op_df.drop(columns=["Supply (bags)"]).columns)
sources   = list(st.session_state.op_df.index)

total_supply = sum(supply)
total_demand = sum(demand)
st.caption(f"Total fleet supply: {total_supply:.0f} bags  ·  Total delivery demand: {total_demand:.0f} bags")


# ----------------------------------------------------------------------------
# Section 2 — Solve
# ----------------------------------------------------------------------------

st.markdown('<div class="section-label">Step 02</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Run optimisation</div>', unsafe_allow_html=True)

cost_type = st.radio(
    "Cost basis for this run:",
    ["Operating Cost", "Commercial Freight Cost"],
    horizontal=True,
)

if st.button("▶  Optimise Routes", type="primary"):
    cost = op_costs if cost_type == "Operating Cost" else com_costs
    alloc, total_cost, note = solve_transportation(supply, demand, cost)

    if note:
        st.info(note)
    else:
        st.success(f"Balanced load: total supply = total demand = {total_supply:.0f} bags.")

    names_i = sources[:]
    names_j = dests[:]
    if total_supply > total_demand:
        names_j = names_j + ["Dummy"]
    elif total_demand > total_supply:
        names_i = names_i + ["Dummy"]

    # Metrics row
    routes = []
    for i, s in enumerate(names_i):
        for j, d in enumerate(names_j):
            qty = alloc[i, j]
            if qty > 1e-6:
                unit_cost = cost[i, j] if i < cost.shape[0] and j < cost.shape[1] else 0
                routes.append({
                    "From": s, "To": d,
                    "Bags": round(qty, 2),
                    "Cost/bag (KES)": round(unit_cost, 2),
                    "Route cost (KES)": round(qty * unit_cost, 2),
                })

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total freight cost (KES)", f"{total_cost:,.2f}")
    mc2.metric("Active routes", len(routes))
    mc3.metric("Cost basis", cost_type.split()[0])

    st.markdown('<div class="section-label" style="margin-top:1.4rem">Allocation matrix</div>', unsafe_allow_html=True)
    alloc_df = pd.DataFrame(np.round(alloc, 2), index=names_i, columns=names_j)
    alloc_df["Total allocated"] = alloc_df.sum(axis=1)
    alloc_df.loc["Demand met"] = alloc_df.sum(axis=0)
    st.dataframe(alloc_df, use_container_width=True)

    if routes:
        st.markdown('<div class="section-label" style="margin-top:1.4rem">Route breakdown</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(routes), use_container_width=True)

st.divider()
