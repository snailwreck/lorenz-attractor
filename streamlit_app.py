import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.title("Lorenz Attractor")

# --- Sidebar controls ---
st.sidebar.header("Parameters")
sigma = st.sidebar.slider("σ (sigma)",  1.0,  20.0, 10.0, step=0.1)
rho   = st.sidebar.slider("ρ (rho)",    0.0,  60.0, 28.0, step=0.1)
beta  = st.sidebar.slider("β (beta)",   0.1,   8.0,  8/3, step=0.01)

st.sidebar.header("Initial Conditions")
x0 = st.sidebar.number_input("x₀",  value=1.0)
y0 = st.sidebar.number_input("y₀",  value=1.0)
z0 = st.sidebar.number_input("z₀",  value=1.0)

st.sidebar.header("Integration")
t_end = st.sidebar.slider("Time span",   10.0, 200.0,  80.0, step=5.0)
dt    = st.sidebar.slider("Step size dt", 0.001, 0.05, 0.005, step=0.001)

# --- Solve ODE ---
@st.cache_data
def solve_lorenz(sigma, rho, beta, x0, y0, z0, t_end, dt):
    def lorenz(t, state):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ]

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(lorenz, t_span, [x0, y0, z0],
                    t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]

t, x, y, z = solve_lorenz(sigma, rho, beta, x0, y0, z0, t_end, dt)

# --- Colour by time so trajectory direction is visible ---
norm_t = (t - t.min()) / (t.max() - t.min())

# --- Plot ---
view = st.radio("View", ["X–Z", "X–Y", "Y–Z", "3D"], horizontal=True)

fig = plt.figure(figsize=(10, 6))

if view == "3D":
    ax = fig.add_subplot(111, projection="3d")
    # plot in segments so we can colour by time
    n_seg = 500
    idx = np.linspace(0, len(t) - 1, n_seg + 1, dtype=int)
    cmap = plt.cm.plasma
    for i in range(n_seg):
        s, e = idx[i], idx[i + 1]
        ax.plot(x[s:e+1], y[s:e+1], z[s:e+1],
                color=cmap(norm_t[s]), lw=0.4, alpha=0.8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
else:
    ax = fig.add_subplot(111)
    cmap = plt.cm.plasma
    if view == "X–Z (classic)":
        coords = (x, z, "X", "Z")
    elif view == "X–Y":
        coords = (x, y, "X", "Y")
    else:
        coords = (y, z, "Y", "Z")
    a, b, xl, yl = coords
    # scatter with time-based colour
    ax.scatter(a, b, c=norm_t, cmap="plasma", s=0.1, alpha=0.6)
    ax.set_xlabel(xl); ax.set_ylabel(yl)

ax.set_title(f"Lorenz Attractor  (σ={sigma}, ρ={rho}, β={beta:.3f})")
plt.tight_layout()
st.pyplot(fig)

st.sidebar.header("Lorenz Parameters")
sigma = st.sidebar.slider("σ (sigma)", 1.0, 20.0, 10.0, step=0.1)
beta  = st.sidebar.slider("β (beta)",  0.1,  8.0,  8/3, step=0.01)

st.sidebar.header("ρ sweep range")
rho_min = st.sidebar.slider("ρ min",  0.0, 100.0,  50.0, step=0.5)
rho_max = st.sidebar.slider("ρ max",  0.0, 300.0, 320.0, step=0.5)
n_rho   = st.sidebar.slider("Number of ρ values", 50, 1000, 300)

st.sidebar.header("Integration")
t_transient = st.sidebar.slider("Transient time (discarded)", 50, 500, 200)
t_collect   = st.sidebar.slider("Collection time",            20, 200,  80)

# --- Compute ---
@st.cache_data
def compute_lorenz_bifurcation(sigma, beta, rho_min, rho_max, n_rho,
                                t_transient, t_collect):
    rho_values = np.linspace(rho_min, rho_max, n_rho)
    rho_plot, z_plot = [], []

    for rho in rho_values:
        # 1. Run transient to a settled state
        def lorenz(t, s):
            x, y, z = s
            return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

        ic = [1.0, 1.0, 1.0]
        sol_t = solve_ivp(lorenz, (0, t_transient), ic,
                          method="RK45", rtol=1e-8, atol=1e-8,
                          dense_output=False)
        ic2 = sol_t.y[:, -1]          # pick up where transient left off

        # 2. Collect steady-state trajectory
        t_eval = np.linspace(0, t_collect, int(t_collect / 0.01))
        sol = solve_ivp(lorenz, (0, t_collect), ic2,
                        t_eval=t_eval, method="RK45",
                        rtol=1e-9, atol=1e-9)
        z = sol.y[2]

        # 3. Record local maxima of z
        peaks_idx = argrelmax(z, order=3)[0]
        for idx in peaks_idx:
            rho_plot.append(rho)
            z_plot.append(z[idx])

    return np.array(rho_plot), np.array(z_plot)

with st.spinner("Integrating across ρ values… this may take a moment."):
    rho_arr, z_arr = compute_lorenz_bifurcation(
        sigma, beta, rho_min, rho_max, n_rho, t_transient, t_collect)

# --- Plot ---
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(rho_arr, z_arr, ",k", alpha=0.3, markersize=0.8)
ax.set_xlabel("ρ  (rho)")
ax.set_ylabel("Local maxima of z")
ax.set_title(f"Lorenz Bifurcation Diagram  (σ={sigma}, β={beta:.3f})")
ax.set_xlim(rho_min, rho_max)
st.pyplot(fig)

# --- Time series ---
with st.expander("Show time series"):
    fig2, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    for ax2, var, label, color in zip(axes, [x, y, z], ["x", "y", "z"],
                                      ["steelblue", "tomato", "seagreen"]):
        ax2.plot(t, var, lw=0.5, color=color)
        ax2.set_ylabel(label)
    axes[-1].set_xlabel("time")
    plt.tight_layout()
    st.pyplot(fig2)
