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
        note = f"Dummy destination added (cost $0): supply exceeds demand by {total_supply - total_demand:.0f} bags."
    elif total_demand > total_supply:
        supply = supply + [total_demand - total_supply]
        cost = np.vstack([cost, np.zeros((1, cost.shape[1]))])
        note = f"Dummy source added (cost $0): demand exceeds supply by {total_demand - total_supply:.0f} bags."
    initial = vogel_initial_solution(supply, demand, cost)
    optimal = modi_optimize(initial, cost)
    total_cost = float((optimal * cost).sum())
    return optimal, total_cost, note


# ----------------------------------------------------------------------------
# App state
# ----------------------------------------------------------------------------

def init_state():
    if "sources" not in st.session_state:
        st.session_state.sources = ["Githurai", "Marikiti", "Mukuyu"]
    if "dests" not in st.session_state:
        st.session_state.dests = ["AGHS", "MHS", "NHS", "PGHS"]
    if "supply" not in st.session_state:
        st.session_state.supply = [51, 121, 30]
    if "demand" not in st.session_state:
        st.session_state.demand = [50, 59, 45, 48]
    if "distances" not in st.session_state:
        st.session_state.distances = np.array([
            [39, 24, 147, 13],
            [34, 35, 159, 11],
            [106, 46, 68, 80],
        ], dtype=float)


init_state()

st.title("Maize distribution transportation model")
st.caption(
    "Set fuel and vehicle assumptions in USD, enter route distances, supply and demand, "
    "then solve for the least-cost distribution plan with Vogel's Approximation Method + MODI."
)

# ----------------------------------------------------------------------------
# Fuel & vehicle assumptions (fixed reference values — do not affect the solver)
# ----------------------------------------------------------------------------

# Fixed constants — for display/reference only
capacity_bags     = 67.0
fuel_consumption  = 8.0    # km/l
diesel_price      = 1.72   # $/l
misc_km           = 8.0    # km
maintenance_trip  = 9.70   # $
cess_trip         = 38.50  # $
staff_trip        = 38.50  # $

fuel_per_km    = (1 / fuel_consumption) * diesel_price
fixed_per_trip = maintenance_trip + cess_trip + staff_trip

st.subheader("1. Fuel and vehicle assumptions (USD) — reference only")
st.caption("These values are fixed parameters used to derive cost per bag from distance. They are not editable here.")

ref_df = pd.DataFrame({
    "Parameter": [
        "Truck capacity (bags)",
        "Fuel consumption (km/l)",
        "Diesel price ($/l)",
        "Misc / detour distance (km)",
        "Maintenance / round trip ($)",
        "Cess / weighbridge ($/trip)",
        "Staff allowances ($/trip)",
        "Derived: Fuel cost ($/km)",
        "Derived: Fixed cost per trip ($)",
    ],
    "Value": [
        capacity_bags,
        fuel_consumption,
        diesel_price,
        misc_km,
        maintenance_trip,
        cess_trip,
        staff_trip,
        round(fuel_per_km, 3),
        round(fixed_per_trip, 2),
    ],
})
st.dataframe(ref_df, use_container_width=False, hide_index=True)

# ----------------------------------------------------------------------------
# Supply, demand, distances
# ----------------------------------------------------------------------------

st.subheader("2. Markets, schools, supply, demand and distance (km)")

bcol1, bcol2, _ = st.columns([1, 1, 6])
with bcol1:
    if st.button("+ Market"):
        st.session_state.sources.append(f"Market {len(st.session_state.sources) + 1}")
        st.session_state.supply.append(0)
        st.session_state.distances = np.vstack(
            [st.session_state.distances, np.zeros((1, len(st.session_state.dests)))]
        )
        st.rerun()
with bcol2:
    if st.button("+ School"):
        st.session_state.dests.append(f"School {len(st.session_state.dests) + 1}")
        st.session_state.demand.append(0)
        st.session_state.distances = np.hstack(
            [st.session_state.distances, np.zeros((len(st.session_state.sources), 1))]
        )
        st.rerun()

n_src = len(st.session_state.sources)
n_dst = len(st.session_state.dests)

table = pd.DataFrame(
    st.session_state.distances,
    index=st.session_state.sources,
    columns=st.session_state.dests,
)
table.insert(0, "Supply", st.session_state.supply)

edited = st.data_editor(
    table,
    num_rows="fixed",
    use_container_width=True,
    key="distance_editor",
)

demand_row = pd.DataFrame(
    [[None] + st.session_state.demand],
    columns=["Supply"] + st.session_state.dests,
    index=["Demand"],
)
st.caption("Weekly demand per school (bags)")
edited_demand = st.data_editor(
    demand_row.drop(columns=["Supply"]),
    num_rows="fixed",
    use_container_width=True,
    key="demand_editor",
)

# Pull edited values back out
st.session_state.supply = edited["Supply"].tolist()
st.session_state.distances = edited.drop(columns=["Supply"]).to_numpy(dtype=float)
st.session_state.demand = edited_demand.iloc[0].tolist()

total_supply = sum(st.session_state.supply)
total_demand = sum(st.session_state.demand)
st.caption(f"Total supply: {total_supply:.0f} bags  |  Total demand: {total_demand:.0f} bags")

# ----------------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------------

st.subheader("3. Solve")

if st.button("Solve", type="primary"):
    distances = st.session_state.distances
    cost = np.zeros((n_src, n_dst))
    for i in range(n_src):
        for j in range(n_dst):
            round_trip_km = 2 * distances[i, j] + misc_km
            running_cost = round_trip_km * fuel_per_km + fixed_per_trip
            cost[i, j] = running_cost / capacity_bags if capacity_bags else 0

    st.markdown("**Derived cost per bag (USD)**")
    cost_df = pd.DataFrame(
        np.round(cost, 2), index=st.session_state.sources, columns=st.session_state.dests
    )
    st.dataframe(cost_df, use_container_width=True)

    alloc, total_cost, note = solve_transportation(
        st.session_state.supply, st.session_state.demand, cost
    )

    if note:
        st.info(note)
    else:
        st.success(f"Balanced problem: total supply = total demand = {total_supply:.0f} bags.")

    names_i = list(st.session_state.sources)
    names_j = list(st.session_state.dests)
    if total_supply != total_demand:
        if total_supply > total_demand:
            names_j = names_j + ["Dummy"]
        else:
            names_i = names_i + ["Dummy"]

    st.markdown("**Optimal allocation (bags)**")
    alloc_df = pd.DataFrame(np.round(alloc, 2), index=names_i, columns=names_j)
    alloc_df["Allocated"] = alloc_df.sum(axis=1)
    alloc_df.loc["Demand met"] = alloc_df.sum(axis=0)
    st.dataframe(alloc_df, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Total cost", f"${total_cost:,.2f}")

    routes = []
    for i, s in enumerate(names_i):
        for j, d in enumerate(names_j):
            qty = alloc[i, j]
            if qty > 1e-6:
                unit_cost = cost[i, j] if i < cost.shape[0] and j < cost.shape[1] else 0
                routes.append({
                    "From": s, "To": d, "Bags": round(qty, 2),
                    "Cost/bag ($)": round(unit_cost, 2),
                    "Route cost ($)": round(qty * unit_cost, 2),
                })
    m2.metric("Routes used", len(routes))

    if routes:
        st.markdown("**Routes used**")
        st.dataframe(pd.DataFrame(routes), use_container_width=True)

st.divider()
st.caption(
    "Cost per bag is derived, not entered directly: "
    "(2 x one-way distance + misc detour) x fuel cost/km, plus maintenance, cess, and staff "
    "allowances per round trip, divided by truck capacity in bags. All figures are in USD."
)
