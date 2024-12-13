# RL-Project

## Overview
This project investigates the role of **memory** in reinforcement learning (RL) by implementing **memory-driven navigation** in dynamic environments. Inspired by natural predator-prey dynamics and foraging behaviors, we design a **2D grid-world environment** where agents must navigate, survive, and gather resources using **Proximal Policy Optimization (PPO)** combined with **Long Short-Term Memory (LSTM)** networks. This is our final project of the class COMPSCI 1840 at Harvard University. 

Unlike traditional RL approaches that often rely on explicit positional awareness, our agents rely solely on **sensory inputs** and **temporal memory** to make decisions. By denying access to absolute positional information, the agents mimic natural behaviors such as remembering resource locations and avoiding predators over time.

Through extensive experimentation and analysis, we demonstrate that:
- Agents with **LSTM-based memory** significantly outperform their non-memory counterparts.
- LSTM hidden states encode **spatial and temporal dependencies**, enabling adaptive behaviors such as revisiting resource locations and evading predators.

This repository contains the complete codebase for training, testing, and visualizing RL agents within these predator-prey environments.

---

## Key Features
- **Custom Grid-World Environment**:  
   A dynamic 2D grid environment with apple trees (resources) and predators simulating natural foraging challenges.
- **Memory-Driven Policies**:  
   Implementation of PPO with LSTM-based policies to analyze memory-driven decision-making.
- **Baseline Comparisons**:  
   Comparison between agents with memory (LSTM) and agents using standard feed-forward architectures.
- **Hidden State Analysis**:  
   Tools for analyzing and visualizing the memory encoded in LSTM hidden states.
- **Parallelized Training**:  
   Support for efficient training across multiple environments using parallel simulation.

---

## Contributions
1. **Novel Framework**: Developed a PPO-LSTM framework for memory-driven navigation in RL tasks.
2. **Hidden State Analysis**: Explored how LSTM hidden states encode information about past positions, enabling advanced spatial behaviors.
3. **Behavior Analysis**: Demonstrated memory's role in adaptive strategies, such as revisiting resources and predator avoidance.

Our results showcase the potential of memory mechanisms to improve RL agent adaptability in dynamic and uncertain environments.


## Project Structure

### Code Directory
The `Code` folder is the main directory and is structured as follows:

#### **Core Scripts**
- **PPO_RNN_Experiment_1.py**:  
   Runs the first experiment using the PPO-RNN model.
- **PPO_RNN_Experiment_2.py**:  
   Runs the second variant of the PPO-RNN experiment with modified configurations and more advanced features
- **PPO_RNN_varying_LSTM_neurons.py**:  
   Tests PPO-RNN models with varying LSTM neuron sizes to analyze performance impact.

#### **Training and Testing**
- **train_multiple.py**:  
   Script for training multiple RL models.
- **test_multiple.py**:  
   Tests multiple models simultaneously to compare their performance.
- **test_trained_model.py**:  
   Tests an already trained PPO-RNN model on evaluation tasks.

#### **Visualization**
- **visualize_states.py**:  
   Visualizes states and hidden layers of the RNN models.
- **visualize_trace.py**:  
   Plots the trace or trajectories of RL agents after training. 
- **visualize_trace_best.py**:  
   A refined version of `visualize_trace.py` for visualizing the best-performing agents.

#### **Utilities**
- **utils.py**:  
   Contains helper functions and utilities used across the project (e.g., logging, data preprocessing).
- **saving_memory.py**:  
   Manages memory-saving techniques during RL training.

#### **Decoding and Results**
- **decode.py**:  
   Decodes saved RL model states or outputs for further analysis.

#### **Data and Results**
- **hidden_states_and_pos.csv**:  
   CSV file containing hidden states and positions of the RL agents for analysis.  
- **Results/**:  
   Directory to store all experimental outputs and logs.

### Other Folders
- **data/**:  
   Contains input datasets or environment configurations.
- **firstCodeForVisualization/**:  
   Initial scripts for visualizing RL outputs.  
- **Parallelized_training/**:  
   Includes code for running RL training processes in a parallelized manner to improve efficiency.

---

## Dependencies
The following libraries are required to run the project:

- PyTorch
- NumPy
- Matplotlib