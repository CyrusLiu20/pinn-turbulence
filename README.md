# Phyics Informed Neural Network for Turbulence Modelling
Turbulence is one of the most complex and challenging problems in fluid mechanics. Traditional methods for solving turbulence equations involve computationally expensive simulations that can take days or even weeks to complete. Recently, physics-informed neural networks (PINNs) have emerged as a promising alternative for solving turbulent flow problems. PINNs combine the power of deep neural networks with the physical laws that govern fluid mechanics to learn the underlying dynamics of turbulence.

This project aims to use a PINN to model turbulent flow on a flat plate. The PINN is trained on a dataset generated using direct numerical simulation (DNS) data for the flow field. The trained PINN is then used to predict the velocity field of the turbulent flow for a given set of boundary conditions. The project also includes an interactive visualization tool for users to explore the results of the PINN predictions and compare them with DNS data.

The main advantage of using a PINN for turbulent flow problems is that it significantly reduces the computational cost compared to traditional simulation methods. This makes it possible to quickly generate accurate predictions for a range of turbulent flow scenarios, which has important applications in fields such as turbomachinery, oceanography, and climate science.