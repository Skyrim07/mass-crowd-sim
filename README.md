# Scalable Mass Crowd Simulation with Multi-Agent Behavior

**CS348K: Visual Computing Systems**  
**Alex Liu** (alexlpc@stanford.edu)  
**Daniel Song** (djsong@stanford.edu)

---

## Overview

This project builds a **real-time multi-agent crowd simulation in Unity**, focusing on scaling the number of NPCs while maintaining interactive performance.

We explore how **system-level optimizations** can increase agent count without sacrificing believable behavior.

The project starts with a naive per-agent update system and incrementally introduces optimized designs such as:

- Centralized scheduling  
- Spatial partitioning  
- Behavior Level-of-Detail (LOD)  
- (Optional) Parallelization via Unity Jobs/Burst  

---

## Goals

- Maximize the number of simulated NPCs  
- Maintain **real-time performance (≥ 30 FPS)**  
- Preserve **reasonable behavioral realism**  
- Understand tradeoffs between **scalability and quality**  

---

## Problem Definition

### Inputs
- Scene layout  
- Navigation mesh  
- Obstacles  
- NPC count  
- Behavior parameters  

### Outputs
- Real-time NPC trajectories and behaviors  
- Performance metrics (FPS, CPU time)  

### Constraints
- Must maintain **interactive frame rate (≥ 30 FPS)** on a standard desktop  

### Research Question

> How can system-level optimizations enable scalable multi-agent simulation in Unity while preserving acceptable behavior quality?

---

## System Design

### Baseline System

A naive implementation where:

- Each NPC is a Unity `GameObject`  
- Each NPC runs a `MonoBehaviour`  
- All agents update every frame  
- Navigation and decision-making are fully independent  

This serves as the performance reference point.

---

### Optimized System

We introduce multiple system-level improvements:

#### 1. Centralized Scheduling
- Update only a subset of agents per frame  
- Use batching or staggered updates  

#### 2. Spatial Partitioning
- Use grid / spatial hashing / quadtree  
- Limit neighbor queries to local regions  

#### 3. Behavior LOD
- High-frequency, full logic for nearby agents  
- Reduced update frequency or simplified logic for distant agents  

#### 4. Parallelization (Optional)
- Unity Jobs System  
- Burst Compiler for performance acceleration  

---

## Evaluation

### Performance Metrics

- Frames Per Second (FPS)  
- CPU frame time  
- Maximum NPC count at target FPS  

### Behavior Quality Metrics

- Task completion rate  
- Collision / congestion frequency  
- Stagnation or deadlock rate  

---

### Experiments

We perform controlled scaling experiments:

- Gradually increase NPC count  
- Compare:
  - Baseline system  
  - Optimized system variants  

### Visualization

- NPC count vs. FPS  
- NPC count vs. task success rate  

---

## Expected Results

We expect:

- **2×–4× increase** in maximum supported agent count  
- Comparable behavior quality between baseline and optimized systems  
- Clear tradeoff curves between performance and realism  

---

## Risks & Mitigation

**Overly complex behaviors**  
→ Keep agent logic simple and controlled  

**Incorrect bottleneck assumptions**  
→ Profile early (CPU, navigation, rendering)  

**Lack of measurable evaluation**  
→ Implement logging and metrics from the beginning  

---

## Deliverables

- Unity demo with scalable NPC simulation  
- Performance evaluation plots  
- Short demo video  
- Final report documenting system design and findings  

---

## Tech Stack

- Unity (C#)  
- NavMesh / custom navigation  
- Unity Jobs + Burst (optional)  
- Unity Profiler  

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-repo/crowd-simulation.git

# Open the project in Unity Hub