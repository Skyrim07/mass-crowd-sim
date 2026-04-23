# Scalable Mass Crowd Simulation with Multi-Agent Behavior

**CS348K: Visual Computing Systems**  
**Alex Liu** (alexlpc@stanford.edu)  
**Daniel Song** (djsong@stanford.edu)

---

## Overview

We propose building a real-time multi-agent simulation in Unity in which a large number of NPCs navigate and perform simple behaviors within a shared game environment. Our goal is to maximize the number of believable agents that can be simulated while maintaining an interactive frame rate. We will begin with a naive baseline where each NPC independently updates its behavior every frame, and then develop optimized system designs using centralized scheduling, spatial partitioning, and behavior level-of-detail (LOD). We will evaluate performance and behavior quality across increasing agent counts to understand the tradeoffs between scalability and realism.

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
The fundamental constraint of this problem is the wall-clock compute time for each update iteration of the agents. We aim to maintain an interactive fram rate of at least 30 FPS on a typical desktop setup.

---

## Task List

- Create a naive implementation where:
  - Each NPC is a Unity `GameObject`  
  - Each NPC runs a `MonoBehaviour`  
  - All agents update every frame  
  - Navigation and decision-making are fully independent  
*This serves as the performance reference point.*
- Implement Centralized Scheduling
  - Update only a subset of agents per frame  
  - Use batching or staggered updates 

- Implement Spatial Partitioning
  - Use grid / spatial hashing / quadtree  
  - Limit neighbor queries to local regions  

- Create a Behavior LOD System
  - High-frequency, full logic for nearby agents  
  - Reduced update frequency or simplified logic for distant agents  


*Nice to haves (if ahead of schedule):*
- Implement Parallelization
  - Unity Jobs System  
  - Burst Compiler for performance acceleration  

*We plan to mostly work on this project together in-person. We expect to work on the naive implementation together, and partially distribute the focus on implementing the three implementation strategies across the two of us.*

---

## Evaluation

To evaluate the system’s success, we will use performance metrics such as FPS, CPU frame time, and the maximum NPC count that can be maintained at a target frame rate, comparing results against the naive implementation. To assess behavior quality, we will define metrics such as task completion rate, collision or congestion frequency, and stagnation or deadlock rates. This naturally motivates controlled scaling experiments such as gradually increasing the NPC count and comparing the optimized system variants to the baseline.

## Expected Deliverables

To demonstrate the capabilities of the system, we will create a Unity demo showcasing the scalable NPC simulations. Ideally, we will be able to show the effects of applying our strategies to compare it against the naive implementation. To present our evaluations and analysis, we plan to create visualizations of the collected data, such as NPC count vs. FPS and NPC count vs. task success rate, along with similar performance and behavior quality comparisons.

---

## Expected Results

We expect:

- 2×–4× increase in maximum supported agent count  
- Comparable behavior quality between baseline and optimized systems  
- Clear tradeoff curves between performance and realism  

---

## Risks & Mitigation

Overly complex behaviors  
→ Keep agent logic simple and controlled  

Incorrect bottleneck assumptions  
→ Profile early (CPU, navigation, rendering)  

Lack of measurable evaluation  
→ Implement logging and metrics from the beginning  

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-repo/crowd-simulation.git

# Open the project in Unity Hub