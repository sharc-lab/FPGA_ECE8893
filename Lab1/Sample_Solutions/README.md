<h3>Solution1: Sample Code and Report</h3>

Submission by: Parima Mehta <br>
Latency obtained: 1.14ms &nbsp;&nbsp;&nbsp;&nbsp;BRAM:426&nbsp;&nbsp;&nbsp;&nbsp;DSP:64&nbsp;&nbsp;&nbsp;&nbsp;FF:20580&nbsp;&nbsp;&nbsp;&nbsp;LUT:48251<br>
Some good points to note:<br>
-Does not completely unroll any loop, instead lowers latency by making best use of loop tiling.<br>
-Does not aggressively partition the arrays.<br>


<h3>Solution2: Sample Code</h3>

Latency obtained: 1.73ms &nbsp;&nbsp;&nbsp;&nbsp;BRAM:268&nbsp;&nbsp;&nbsp;&nbsp;DSP:47&nbsp;&nbsp;&nbsp;&nbsp;FF:3239&nbsp;&nbsp;&nbsp;&nbsp;LUT:13524<br>
A good point to note:<br>
-Low latency with focus on minimal resource usage <br>
