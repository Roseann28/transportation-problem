import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Maize Distribution Transportation Model", layout="wide")


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
        if len(values) == 0:
            return -1
        if len(values) == 1:
            return values[0]
        return values[1] - values[0]

    active_rows, active_cols = m, n
    while active_rows > 0 and active_cols > 0:
        best_pen, best_type, best_idx = -1, None, -1
        for i in range(m):
            if row_done[i]:
                continue
            p = penalty([cost[i, j] for j in range(n) if not col_done[j]])
            if p > best_pen:
                best_pen, best_type, best_idx = p, "row", i
        for j in range(n):
            if col_done[j]:
                continue
            p = penalty([cost[i, j] for i in range(m) if not row_done[i]])
            if p > best_pen:
                best_pen, best_type, best_idx = p, "col", j

        if best_type == "row":
            i = best_idx
            j = min((j for j in range(n) if not col_done[j]), key=lambda j: cost[i, j])
        else:
            j = best_idx
            i = min((i for i in range(m) if not row_done[i]), key=lambda i: cost[i, j])

        qty = min(supply[i], demand[j])
        alloc[i, j] += qty
        supply[i] -= qty
        demand[j] -= qty
        if supply[i] == 0 and not row_done[i]:
            row_done[i] = True
            active_rows -= 1
        if demand[j] == 0 and not col_done[j]:
            col_done[j] = True
            active_cols -= 1
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
                    if u[i] is not None and v[j] is None:
                        v[j] = cost[i, j] - u[i]
                        changed = True
                    elif v[j] is not None and u[i] is None:
                        u[i] = cost[i, j] - v[j]
                        changed = True
        if not changed:
            break
    return [0 if x is None else x for x in u], [0 if x is None else x for x in v]


def find_loop(alloc, enter_cell):
    m, n = alloc.shape
    basic_cells = [(i, j) for i in range(m) for j in range(n) if alloc[i, j] > 0]
    basic_cells.append(enter_cell)

    def neighbors(cell):
        res = []
        for c in basic_cells:
            if c[0] == cell[0] and c[1] != cell[1]:
                res.append(c)
            if c[1] == cell[1] and c[0] != cell[0]:
                res.append(c)
        return res

    path = [enter_cell]

    def dfs(cell, direction):
        if len(path) > 3 and cell == enter_cell:
            return True
        for nb in neighbors(cell):
            if nb in path[1:]:
                continue
            next_dir = "row" if nb[0] == cell[0] else "col"
            if next_dir == direction:
                continue
            path.append(nb)
            if dfs(nb, next_dir):
                return True
            path.pop()
        return False

    if not dfs(enter_cell, "col"):
        return None
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
                    if d < min_d - 1e-9:
                        min_d, enter_cell = d, (i, j)
        if enter_cell is None:
            break
        loop = find_loop(alloc, enter_cell)
        if loop is None:
            break
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
# Default cost matrices from Excel
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
# App state
# ----------------------------------------------------------------------------

def init_state():
    if "sources" not in st.session_state:
        st.session_state.sources = DEFAULT_SOURCES.copy()
    if "dests" not in st.session_state:
        st.session_state.dests = DEFAULT_DESTS.copy()
    if "supply" not in st.session_state:
        st.session_state.supply = DEFAULT_SUPPLY.copy()
    if "demand" not in st.session_state:
        st.session_state.demand = DEFAULT_DEMAND.copy()
    if "op_costs" not in st.session_state:
        st.session_state.op_costs = OPERATING_COSTS.copy()
    if "com_costs" not in st.session_state:
        st.session_state.com_costs = COMMERCIAL_COSTS.copy()


init_state()

st.title("Maize Distribution Transportation Model")
st.caption(
    "Enter cost per bag for each market-to-school route under both cost types, "
    "then solve for the least-cost distribution plan using Vogel's Approximation Method + MODI."
)

# ----------------------------------------------------------------------------
# Section 1: Supply, demand & cost matrices
# ----------------------------------------------------------------------------

st.subheader("1. Markets, schools, supply and demand")

bcol1, bcol2, _ = st.columns([1, 1, 6])
with bcol1:
    if st.button("+ Market"):
        st.session_state.sources.append(f"Market {len(st.session_state.sources) + 1}")
        st.session_state.supply.append(0)
        st.session_state.op_costs  = np.vstack([st.session_state.op_costs,  np.zeros((1, len(st.session_state.dests)))])
        st.session_state.com_costs = np.vstack([st.session_state.com_costs, np.zeros((1, len(st.session_state.dests)))])
        st.rerun()
with bcol2:
    if st.button("+ School"):
        st.session_state.dests.append(f"School {len(st.session_state.dests) + 1}")
        st.session_state.demand.append(0)
        st.session_state.op_costs  = np.hstack([st.session_state.op_costs,  np.zeros((len(st.session_state.sources), 1))])
        st.session_state.com_costs = np.hstack([st.session_state.com_costs, np.zeros((len(st.session_state.sources), 1))])
        st.rerun()

n_src = len(st.session_state.sources)
n_dst = len(st.session_state.dests)

# --- Operating Cost matrix ---
st.markdown("**Operating cost per bag (KES)**")
op_table = pd.DataFrame(st.session_state.op_costs, index=st.session_state.sources, columns=st.session_state.dests)
op_table.insert(0, "Supply (bags)", st.session_state.supply)
edited_op = st.data_editor(op_table, num_rows="fixed", use_container_width=True, key="op_editor")

# --- Commercial Freight Cost matrix ---
st.markdown("**Commercial freight cost per bag (KES)**")
com_table = pd.DataFrame(st.session_state.com_costs, index=st.session_state.sources, columns=st.session_state.dests)
com_table.insert(0, "Supply (bags)", st.session_state.supply)
edited_com = st.data_editor(com_table, num_rows="fixed", use_container_width=True, key="com_editor")

# --- Demand row (shared) ---
st.markdown("**Weekly demand per school (bags)**")
demand_row = pd.DataFrame([st.session_state.demand], columns=st.session_state.dests, index=["Demand (bags)"])
edited_demand = st.data_editor(demand_row, num_rows="fixed", use_container_width=True, key="demand_editor")

# Pull edited values back
st.session_state.supply    = edited_op["Supply (bags)"].tolist()
st.session_state.op_costs  = edited_op.drop(columns=["Supply (bags)"]).to_numpy(dtype=float)
st.session_state.com_costs = edited_com.drop(columns=["Supply (bags)"]).to_numpy(dtype=float)
st.session_state.demand    = edited_demand.iloc[0].tolist()

total_supply = sum(st.session_state.supply)
total_demand = sum(st.session_state.demand)
st.caption(f"Total supply: {total_supply:.0f} bags  |  Total demand: {total_demand:.0f} bags")

# ----------------------------------------------------------------------------
# Section 2: Solve
# ----------------------------------------------------------------------------

st.subheader("2. Solve")

cost_type = st.radio(
    "Select cost matrix to optimise:",
    ["Operating Cost", "Commercial Freight Cost"],
    horizontal=True,
)

if st.button("Solve", type="primary"):
    cost = st.session_state.op_costs if cost_type == "Operating Cost" else st.session_state.com_costs

    alloc, total_cost, note = solve_transportation(
        st.session_state.supply, st.session_state.demand, cost
    )

    if note:
        st.info(note)
    else:
        st.success(f"Balanced problem: total supply = total demand = {total_supply:.0f} bags.")

    names_i = list(st.session_state.sources)
    names_j = list(st.session_state.dests)
    if total_supply > total_demand:
        names_j = names_j + ["Dummy"]
    elif total_demand > total_supply:
        names_i = names_i + ["Dummy"]

    st.markdown("**Optimal allocation (bags)**")
    alloc_df = pd.DataFrame(np.round(alloc, 2), index=names_i, columns=names_j)
    alloc_df["Total allocated"] = alloc_df.sum(axis=1)
    alloc_df.loc["Demand met"] = alloc_df.sum(axis=0)
    st.dataframe(alloc_df, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Total cost (KES)", f"{total_cost:,.2f}")

    routes = []
    for i, s in enumerate(names_i):
        for j, d in enumerate(names_j):
            qty = alloc[i, j]
            if qty > 1e-6:
                unit_cost = cost[i, j] if i < cost.shape[0] and j < cost.shape[1] else 0
                routes.append({
                    "From": s,
                    "To": d,
                    "Bags": round(qty, 2),
                    "Cost/bag (KES)": round(unit_cost, 2),
                    "Route cost (KES)": round(qty * unit_cost, 2),
                })
    m2.metric("Routes used", len(routes))

    if routes:
        st.markdown("**Active routes**")
        st.dataframe(pd.DataFrame(routes), use_container_width=True)

st.divider()
st.caption("Solved using Vogel's Approximation Method (VAM) for the initial basic feasible solution, optimised with the Modified Distribution (MODI) method.")
