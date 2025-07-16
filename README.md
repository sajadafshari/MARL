# Learning in Space: Multi-Agent Reinforcement Learning for LEO Satellite Networks

This repository contains MATLAB code and supporting material for the paper:

**Learning in Space: Multi-Agent Reinforcement Learning for LEO Satellite Networks**  
Sajad Saraygord Afshari, Peng Hu, Mana Zandvakili, Philip Ferguson

The code provides simulation frameworks, benchmarks, and reproducible experiments for applying **multi-agent reinforcement learning (MARL)** to three key challenges in Low Earth Orbit (LEO) satellite constellations: resource allocation, dynamic routing, and computation offloading. It accompanies the full manuscript (see `MARL_for_LEO_sats_survey_paper-4.pdf`), which surveys state-of-the-art MARL methods in this rapidly evolving domain[4].

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Case Study Descriptions](#case-study-descriptions)
  - [Resource Allocation](#resource-allocation)
  - [Routing](#routing)
  - [Computation Offloading](#computation-offloading)
- [Dependencies](#dependencies)
- [How to Cite](#how-to-cite)
- [Contact](#contact)

---

## Overview

LEO satellite networks are a fundamental component of future global connectivity. They offer unique advantages such as low-latency coverage, resilience, and scalability—but pose complex challenges due to their highly **dynamic topologies**, **limited onboard resources**, and **stringent quality of service (QoS)** demands. This repository supports the in-depth survey and case studies in our paper, showing how MARL enables more robust and adaptive solutions for:

- **Decentralized resource allocation**
- **Dynamic, delay-aware routing**
- **Computation offloading under mobility and energy constraints**

The open-source codes allow replication and extension of our simulations, enabling researchers and engineers to benchmark MARL algorithms against classical methods using realistic LEO satellite models[4].

---

## Repository Structure

| File / Folder           | Description                                                                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| `Offloading.m`          | MATLAB code for decentralized MARL and centralized DRL approaches to computation offloading             |
| `MARL_final_Routing.m`  | MATLAB implementation for enhanced MARL-based routing vs. shortest-path routing in a large LEO network  |
| `routing.m`             | MATLAB code for MARL-based vs. shortest-path routing (smaller/topologically diverse network)            |
| `LICENSE`               | Licensing information (MIT or equivalent, see file)                                                     |
| (add any other relevant files) |                                                                                                  |

---

## Getting Started

1. System Requirements
   - MATLAB R2019b or later (due to neural network and dlnetwork features required)[2]
   - Standard MATLAB toolboxes: Deep Learning Toolbox is recommended

2. Clone the Repository
   -Here

3. Run Experiments
- For each case study (see below), open the relevant `.m` script in MATLAB and run.
- All simulation parameters are set at the top of each script and can be changed to customize the experiments.

---

## Case Study Descriptions

### Resource Allocation

Demonstrates MARL for dynamic, decentralized allocation of transmission power and communication channels among LEO satellites. Code simulates satellites as autonomous MARL agents learning to meet varying ground terminal demands under severe resource constraints and dynamic topology[4][1].

- Key features: Distributed Q-learning, partial observability, real-time network adaptation, comparison with greedy (non-MARL) allocation.

### Routing

Implements MARL-based next-hop routing with neural network function approximation and experience replay. Satellites learn delay- and congestion-minimizing paths under rapidly evolving network conditions and compete with classical shortest-path routing[4][2][3].

- Scenarios: Grid and more realistic orbital networks.
- Metrics: Total communication delay, hop count, and routing stability.

### Computation Offloading

Models task offloading among satellites, neighbors, and ground stations, balancing processing delays and energy use. Both fully decentralized (independent MARL per satellite) and centralized (CDRL) Q-learning baselines are provided[4][1].

- Key variables and policies: Queue levels, energy weights, communication costs, delay-aware Q-updates.

---

## Dependencies

- MATLAB R2019b or newer
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html) (for neural/dlnetwork functions)
- No external non-MathWorks libraries required

---

## How to Cite

If you use this codebase or reproduce our results in your work, please cite:

> Sajad Saraygord Afshari, Peng Hu, Mana Zandvakili, Philip Ferguson,  
> "Learning in Space: Multi-Agent Reinforcement Learning for LEO Satellite Networks," 2025.

---

## Contact

For questions, bug reports, or collaboration opportunities, please contact:

- Sajad Saraygord Afshari: sajad.afshari@gmail.com

---

## Acknowledgement

This repository is released for academic and research purposes. Please refer to the manuscript for a detailed discussion, benchmarking results, and practical guidelines[4].

---

*Last updated: July 16, 2025*
