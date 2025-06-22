#Day 12 CUDA Debugging: 

Whether to use Nsight compute or Systems ? Which one would help see the perf blockers across the system ? 
![image](https://github.com/user-attachments/assets/81140cc9-256c-4155-9885-c1d323d4b9bd)
                        Overall tools

![image](https://github.com/user-attachments/assets/fc864c36-474f-48f5-b0a9-e8cdce994b35)

Nsight Systems: how data is moving, how tasks ae parllized,  how optimixing would most likely be succsessful ? Need more analysis on Kernel ? zoom in & navigate to Nsight compute (low level GPU processes). 

Nsight compute: Guided analysis, GPU throughput, acompute and mmeory worload analayis. line by line access. 

CUDA GDB: 

Compute Sanitizer: Functional correctness checking suite

CUPTI: For trace and profile CUDA tools. 

![image](https://github.com/user-attachments/assets/d0c5c82c-1748-431d-98ee-e8fe35407ea8)



Ref: 

1. https://developer.nvidia.com/nsight-compute#:~:text=NVIDIA%20Nsight%20Systems%20is%20a,size%20of%20CPUs%20and%20GPUs.
2. 


