# ➴SpikeLab
An interactive simulation of how a single neuron fires, built with Python and Streamlit. It uses the leaky integrate-and-fire model, a simple description of how a neuron collects input, leaks charge, and fires a spike when it reaches a threshold.

https://spikelab-i11.streamlit.app/

## Features

- Seven sliders to change the neuron's properties and see the membrane potential respond
- Spike count, mean firing rate and the minimum current needed to fire, calculated live
- A graph of firing rate against input current
- A refractory period, which sets a maximum firing rate
- A short Learn tab explaining the model and each setting
- Runs in the browser, with nothing to install
  
# Limitations

This model has no ion channels, so it does not reproduce the true shape of an action potential. Spikes are drawn as a fixed marker.

# Author

Ishita
