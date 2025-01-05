# Final Project Topic 3: Pixel Clustering

## Project Description
In this project, you will develop an algorithm for a pixel clustering problem used in High Energy Physics (HEP) and implement it on FPGA as fast as possible. This is a real problem -- HEP researchers are waiting for you.

### Files provided:

- **Problem Definition**: pixel_clustering.pdf

- **Test benches**: `tb_output_chip_x.txt`, x = 0 to 8

### Goals and Requirements

- Finish processing all files **end-to-end**, including data loading and writing back.

- Data can only be read in one line per clock cycle: real-world limitation.

- Data must be written back to DRAM after processing; no constraints on write-back.

- Ideally, finish each event within **10 us** -- real-world requirement, but the faster, the better. If you can achieve **2 us** or less, even better.

---

## Evaluation Criteria
1. **End-to-End Latency**: Measure the total time required to process all input files (you can merge them into one), from reading input data to writing processed results back to DRAM.
2. **Post-Implementation Resource Utilization**: Report FPGA resource usage, including DSP slices, BRAM, LUTs, and FFs.
3. **Speedup Comparison**: Compare the latency of the FPGA implementation with CPU implementation (if you have one; optional). Report the speedup achieved.
4. **Ranking**: Projects will be ranked based on end-to-end latency and resource efficiency.

---

## Deliverables
1. **HLS Implementation**:
   - Submit well-commented HLS design files with detailed explanations of your design choices.
   - Use **HLSFactory** for submission unless publication is planned.

2. **Simulation Results**:
   - Verify correctness of your computation using simulation.

3. **Performance Report**:
   - **Latency**: Measure end-to-end latency for processing all inputs using **C/RTL Co-Simulation** or **LightningSim**.
   - **Resource Utilization**: Provide detailed post-implementation resource usage data, including the generated bitstream.

4. **Technical Report**:
   - Use LaTeX in IEEE double-column format. Include:
     - **Design Overview**:
       - Algorithm-level details, including any optimizations made to improve performance.
       - Hardware component details, focusing on the core processing elements.
     - **Experimental Results**:
       - Performance metrics (latency, throughput, etc.) and resource usage.
       - (Optional) Comparisons with baseline implementation on a CPU.
     - **Insights and Takeaways**:
       - Lessons learned, challenges encountered, and potential areas for improvement.

---

## Submission Instructions
- Submit all required files (code, simulation results, and reports) as a **zipped folder**.
- Ensure all files are clearly labeled and easy to navigate.

---

**Good luck with your project!**
