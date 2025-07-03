In convolution,  Each output data element can be calculated independently of each other, a desirable trait for parallel computing.
On the other hand, there is substantial level of input data sharing among output data elements with somewhat challenging boundary 
conditions. This makes convolution an important use case of sophisticated tiling methods and input data staging methods.

Why ?
  Using the kernel and doing conv to a signal, we can try to extract a feature. 

What ?   
  Conv: 
  Sliding dot prod of a flipped kernel wiht its signal. 

How ? 
  
  Y & X: 0 to n. 
  Kernel: W-p to Wp (Making it odd numbered and symmetric around 0) 
  
  1D Conv: 
  Last element in a 1D output y = sliding kernel (w) middle element meets the last element of input x (i.e W-p to W0) & vice-versa (i.e W0 to Wp). 
  
![image](https://github.com/user-attachments/assets/f5bf9eea-2486-424f-8106-50cd8975bf4e)
Each element of X being multiplied by a sliding / overall kernel ranging from - p to p. 

