# DDP Advanced / Practical:

More GPUs - more throrughput: 

Fashion_Mnist.py - Just observing the throughput when changing the number of GPUs - will help you observe the inc in throughput - images/sec. 
!python3 fashion_mnist.py --epochs 3 --num-gpus 1 --batch-size 128
1 GPU: Images/sec = 622

Multi GPU: 
!python3 fashion_mnist.py --epochs 5 --num-gpus 4 --batch-size 128
4 GPU: Images/sec = ___

Clearly the Multi GPU Speedup factor: 

Note: It stands to reason that each GPU would add a significant boost to the training process.  We don't get perfectly linear scaling. A significant component of this is due to communication between the GPUs when updating weights. There can also be other factors, such as waiting for slower GPUs to finish processing before the weights are averaged. But still, this is pretty good.
Production DL training at scale is usually benchmarked against the ideal case of linear scaling (N GPUs should be N times faster than 1 GPU). DDP, and the NCCL library, do a good job of maintaining high throughput, but it's worth mentioning that performance is also intricately tied to the hardware in use. As you scale to more GPUs, multi-node training is required, and further hardware considerations are needed to effectively scale.

Multi GPU Effect on Validation accuracy: 
! pending

Playing with Optimizer: 
Recap:In the stochastic gradient descent optimizer that we've used so far, weights are updated based on their gradient with respect to the loss function of a mini-batch. In other words, we determine how altering a weight will affect the loss, and move a small step in the direction that minimizes that loss for the mini-batch.
* Selecting and tuning the optimizer that we use. 




