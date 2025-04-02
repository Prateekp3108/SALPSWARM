# Optimizing the Salp Swarm Algorithm Using Two-Stage Lévy Flight and Opposition-Based Learning

## Overview
This project focuses on improving the Salp Swarm Algorithm (SSA) by introducing a two-stage Lévy flight mechanism along with Opposition-Based Learning (OBL). The goal is to enhance SSA's exploration and exploitation capabilities, leading to improved optimization performance.

## Objectives
- Modify the standard SSA using a **two-stage Lévy flight** for better global search capability.
- Integrate **Opposition-Based Learning (OBL)** to improve population diversity and convergence speed.
- Evaluate the performance of the modified SSA against benchmark functions and real-world optimization problems.

## Methodology
### **1. Standard Salp Swarm Algorithm (SSA)**
SSA is a nature-inspired metaheuristic algorithm that mimics the swarming behavior of salps in the ocean. It consists of:
- **Leader Salps**: The first salp in the chain, which moves based on an adaptive function.
- **Follower Salps**: Remaining salps that follow the leader using position updates.

### **2. Enhancements Introduced**
#### **Two-Stage Lévy Flight**
- Instead of a single random movement update, a two-stage Lévy flight is incorporated to balance exploration (global search) and exploitation (local refinement).
- This improves SSA’s ability to escape local optima and explore the search space more efficiently.

#### **Opposition-Based Learning (OBL)**
- OBL generates opposite solutions to enhance diversity and prevent premature convergence.
- It ensures better initial population distribution and refines the search in later iterations.

### **3. Benchmarking and Performance Evaluation**
- The modified SSA is tested on standard benchmark functions (Sphere, Rosenbrock, Rastrigin, etc.).
- Performance is compared based on metrics such as convergence speed, solution quality, and computational efficiency.

## Implementation
- The algorithm is implemented in **Python/MATLAB**.
- Libraries such as NumPy, SciPy, and Matplotlib are used for computations and visualization.

## Results and Findings
- The optimized SSA with two-stage Lévy flight and OBL demonstrates improved convergence and accuracy.
- Benchmark comparisons show significant performance gains over standard SSA and other metaheuristic algorithms.

## Future Scope
- Extending this approach to multi-objective optimization problems.
- Applying the optimized SSA to real-world applications such as feature selection, image processing, and engineering design.

## References
- Mirjalili, S., & Lewis, A. (2017). The Salp Swarm Algorithm. *Advances in Engineering Software*.
- Papers on Lévy flight and Opposition-Based Learning in swarm intelligence.

---
**Author:** Suhani Sharma and Prateek Pandey 
**Affiliation:** Manipal University Jaipur  


