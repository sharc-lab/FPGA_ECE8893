# Overview of Labs and Final Projects

## Labs (30 Points Total)
The labs are individual assignments designed to build your skills in HLS and smart architecture design. Each lab will require:
- Submission of source code.
- Performance metrics submitted to a leaderboard for ranking (if applicable).
- A well-written report explaining your implementation and findings.

### Lab Details:
### **Lab 1: Basic Practice of Vitis HLS**
- Objective: Familiarize yourself with the HLS design flow.
- Notes: Simple introductory lab; no leaderboard ranking.
- Points: 10

### **Lab 2: Implementation of an Attention Layer**
- Objective: Implement an attention layer based on a given reference.
- Requirements: Apply advanced HLS techniques such as parallelism and pipelining.
- Points: 10

### **Lab 3: Sparse Matrix Multiplication**
- Objective: Design an efficient sparse matrix multiplication architecture.
- Requirements: Go beyond basic HLS usage by applying smart architecture design principles.
- Points: 10

---

## Final Project (60 Points Total)
The final project involves working in groups (up to 4 members per group) to solve a challenging problem that requires both algorithmic and architectural design. The deliverables include:
- **Midterm Report**: 5 points
- **Final Report**: 25 points
  - Format: IEEE double-column format using LaTeX.
  - Content: Performance data and detailed explanations of the design.
- **Source Code**: 25 points
  - Requirement: Submit to HLSFactory (if no publication is planned).
- **Presentation**: 5 points
  - Clearly identify each group member’s contributions.
  - Length: TBD.

### Final Project Topics:
#### **Topic 1: Flexible Mixed-Precision Matrix Multiplication**
- Objective: Support customized floating-point and fixed-point formats.
- Key Tasks:
  - Design an efficient floating-point multiplier with an adder that can support mixed precision operation.
  - Develop a smart scheduling strategy for matrix multiplication.


#### **Topic 2: Hardware Design for High Energy Physics Track Construction**
- Objective: Create an optimal hardware design for a track construction algorithm, used in high energy physics.
- Key Tasks:
  - Translate the give Python code into synthesizable HLS C.
  - Optimize the end-to-end latency and the overall throughput for processing all graphs


#### **Topic 3: Hardware Design for Cryptography Algorithms (TBA)**
- Objective: Create an optimal hardware design for a cryptographic algorithm primitive.
- Key Tasks:
  - Fully explore the design space.
  - Identify the best algorithm and hardware pair.
- Notes: Detailed introduction will be provided by Prof. Qirun Zhang (SCS).

#### **Topic 4: Accelerator for Neurosymbolic AI (TBA)**
- Objective: Design an accelerator for neurosymbolic AI computation.
- Notes: Topic and details will be introduced by Zishen Wan.

---

## Additional Notes:
- Reports for both labs and the final project should adhere to IEEE double-column format using LaTeX.
- Submission to HLSFactory is mandatory for labs and the final project unless a publication is planned.
- Leaderboard rankings for labs will help assess performance compared to peers.
