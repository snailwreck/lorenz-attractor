import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.title("Lorenz Attractor")

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
