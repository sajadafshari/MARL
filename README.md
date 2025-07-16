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
| `MARL_for_LEO_sats_survey_paper-4.pdf` | Full manuscript of the survey and case study paper                                         |
| `Offloading.m`          | MATLAB code for decentralized MARL and centralized DRL approaches to computation offloading             |
| `MARL_final_Routing.m`  | MATLAB implementation for enhanced MARL-based routing vs. shortest-path routing in a large LEO network  |
| `routing.m`             | MATLAB code for MARL-based vs. shortest-path routing (smaller/topologically diverse network)            |
| `LICENSE`               | Licensing information (MIT or equivalent, see file)                                                     |
| (add any other relevant files) |                                                                                                  |

---

## Getting Started

1. **System Requirements**
   - MATLAB R2019b or later (due to neural network and dlnetwork features required)[2]
   - Standard MATLAB toolboxes: Deep Learning Toolbox is recommended

2. **Clone the Repository**
