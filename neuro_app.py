"""
SpikeLab - an interactive neuron simulator (Streamlit app)
Run with:  streamlit run neuro_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["savefig.transparent"] = True
plt.rcParams["legend.frameon"] = False

from pathlib import Path
from PIL import Image

icon = Image.open(Path(__file__).parent / "icon.png")
st.set_page_config(page_title="SpikeLab", page_icon=icon, layout="wide")

# ---------------- Simulation ----------------
def simulate(current, v_rest, v_thresh, v_reset, tau, r, refractory, t_total, dt=0.1):
    """Leaky integrate-and-fire neuron with a refractory period."""
    time = np.arange(0, t_total, dt)
    v = np.full(len(time), v_rest, dtype=float)
    spikes = []
    last_spike = -1e9

    for i in range(1, len(time)):
        t = time[i]
        if t - last_spike < refractory:          # neuron is "resting" after a spike
            v[i] = v_reset
            continue
        dv = (-(v[i - 1] - v_rest) + r * current) / tau * dt
        v[i] = v[i - 1] + dv
        if v[i] >= v_thresh:
            spikes.append(t)
            last_spike = t
            v[i] = v_reset
    return time, v, spikes


# ---------------- Sidebar controls ----------------
st.sidebar.title("❀Neuron settings")
current = st.sidebar.slider("Input current (nA)", 0.0, 10.0, 3.0, 0.1)
v_rest = st.sidebar.slider("Resting potential (mV)", -80.0, -60.0, -70.0, 1.0)
v_thresh = st.sidebar.slider("Firing threshold (mV)", -60.0, -40.0, -50.0, 1.0)
tau = st.sidebar.slider("Membrane time constant τ (ms)", 2.0, 50.0, 10.0, 1.0)
r = st.sidebar.slider("Membrane resistance (MΩ)", 1.0, 20.0, 10.0, 0.5)
refractory = st.sidebar.slider("Refractory period (ms)", 0.0, 10.0, 2.0, 0.5)
t_total = st.sidebar.slider("Simulation time (ms)", 100, 1000, 300, 50)

# ---------------- Page ----------------
st.title("➴SpikeLab: Simulating neuronal firing")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Gochi+Hand&display=swap');

    /* pink page */
    .stApp {
        background-color: #fff7f8;
    }
    [data-testid="stSidebar"] {
        background-color: #ffe4e1;
    }
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* dark text */
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li {
        color: #2b2b3a;
    }

    /* title */
    h1, [data-testid="stHeading"] h1 {
        font-family: 'Gochi Hand', cursive !important;
        font-size: 3.0rem !important;
        font-weight: 400 !important;
        color: #c43c78 !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)
st.write("‪‪‬Play with the settings on the left and watch how the neuron respond!🫨")

tab1, tab2, tab3 = st.tabs(["Single neuron✶", "Firing rate curveᝰ", "Learn:)"])

# ----- Tab 1: voltage trace -----
with tab1:
    time, v, spikes = simulate(current, v_rest, v_thresh, v_rest, tau, r, refractory, t_total)
    rate = len(spikes) / (t_total / 1000)

    c1, c2, c3 = st.columns(3)
    c1.metric("Spikes", len(spikes))
    c2.metric("Firing rate", f"{rate:.0f} spikes/s")
    min_current = (v_thresh - v_rest) / r
    c3.metric("Minimum current to fire", f"{min_current:.2f} nA")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, v, color="tab:blue", label="Membrane potential")
    ax.axhline(v_thresh, color="red", linestyle="--", label="Threshold")
    for s in spikes:
        ax.vlines(s, v_thresh, 20, color="black")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend(loc="upper right")
    st.pyplot(fig)

    if len(spikes) == 0:
        st.info("No spikes: the input is too weak to reach threshold. Increase the current.")

# ----- Tab 2: F-I curve -----
with tab2:
    st.write("✦ How does firing rate change as the input gets stronger?")
    currents = np.arange(0, 10.1, 0.25)
    rates = []
    for c in currents:
        _, _, sp = simulate(c, v_rest, v_thresh, v_rest, tau, r, refractory, t_total)
        rates.append(len(sp) / (t_total / 1000))

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(currents, rates, marker="o")
    ax2.axvline(current, color="gray", linestyle=":", label="Current input")
    ax2.set_xlabel("Input current (nA)")
    ax2.set_ylabel("Firing rate (spikes/s)")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# ----- Tab 3: explanation -----
with tab3:
    st.subheader("☆ Leaky integrate-and-fire model")
    st.markdown(
        """
1. **Integrate:** The neuron has an internal voltage (that's the membrane potential) & when it receives electrical inputs from other neurons, it adds them together, causing its internal voltage to rise..
2. **Leak:**  If no new inputs arrive, the voltage decays and drifts back down toward its baseline resting level over time :(
3. **Fire:**  If the inputs arrive fast enough to push the voltage beyond a specific limit (the threshold), then the neuron fires a single electrical output pulse (a spike)!!
4. **Reset:** Immediately after firing, its voltage drops back down to a low baseline value (the reset potential) and the process repeats.

**★ What does each setting really do?**

| Setting | Effect |
|---|---|
| Input current | More current → faster firing |
| Resting potential | Raising it puts the neuron closer to the threshold, so it fires more easily |
| Threshold | Higher threshold makes firing harder (fewer spikes) |
| τ (time constant) | Larger τ → a slow neuron that takes longer to reach threshold |
| Resistance | Larger resistance → the same current produces a bigger voltage change (membrane leaks less) |
| Refractory period | It caps the maximum firing rate |
| Simulation time | A longer run gives more spikes in total and a more accurate firing rate |
        """
    )
