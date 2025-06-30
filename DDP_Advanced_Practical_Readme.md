# DDP Advanced / Practical:

More GPUs - more throrughput: 

Fashion_Mnist.py - Just observing the throughput when changing the number of GPUs - will help you observe the inc in throughput - images/sec. 
!python3 fashion_mnist.py --epochs 3 --num-gpus 1 --batch-size 128
1 GPU: Images/sec = 622

Multi GPU: 
!python3 fashion_mnist.py --epochs 5 --num-gpus 4 --batch-size 128
4 GPU: Images/sec = 1835

Clearly the Multi GPU Speedup factor: 
1 run: 1835/620 = 3x
2nd run: 2400/620 = 4x

Note: It stands to reason that each GPU would add a significant boost to the training process.  We don't get perfectly linear scaling. A significant component of this is due to communication between the GPUs when updating weights. There can also be other factors, such as waiting for slower GPUs to finish processing before the weights are averaged. But still, this is pretty good.
Production DL training at scale is usually benchmarked against the ideal case of linear scaling (N GPUs should be N times faster than 1 GPU). DDP, and the NCCL library, do a good job of maintaining high throughput, but it's worth mentioning that performance is also intricately tied to the hardware in use. As you scale to more GPUs, multi-node training is required, and further hardware considerations are needed to effectively scale.

Multi GPU Effect on Validation accuracy: 
! pending

Playing with Optimizer: 
Recap:In the stochastic gradient descent optimizer that we've used so far, weights are updated based on their gradient with respect to the loss function of a mini-batch. In other words, we determine how altering a weight will affect the loss, and move a small step in the direction that minimizes that loss for the mini-batch.
* Selecting and tuning the optimizer that we use. 
  
  Though this process works well, it can be improved upon. One downside is that if the network gets near a local minimum or saddle point, the gradient can be quite small, and the training will slow down in that area. The noise introduced by using minibatches might help the model find its way out, but it might take some time. 
  
  Additionally, there may be areas where the algorithm keeps taking steps in roughly the same direction. It would be advantageous in those areas if the optimizer helped us take larger steps to move toward the global minimum faster.
  
  A good solution to these issues is to use momentum. Instead of the algorithm taking a small independent step each time, adding momentum to our optimizer allows the process to retain a memory of the last several steps. If the weights have been moving in a certain direction on average, momentum will help continue to propel the updates in the same direction. If the training is also fluctuating back and forth, it can smooth out this movement. A decent analogy is a ball rolling down a hill, which will pick up and retain momentum.

This momentum is a form of normalization to vanilla SGD to prevent gradients from overshooting. For more bigger datasets, bigger batches and higher base_lr we may shift to new optimizers thoguh ! 

Note: Below experiments are done with Fashion MNIST Dataset. 
Adding momentum:

  Step 1: Add momentum arg
  
  ```Python
  parser.add_argument('--momentum', type=float, default=0.9,
                      help='SGD momentum')
  ```
  
  Step 2: Pass the momentum into the optimizer:
  
  ```Python
  optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum)
  ```
Results: 

Epoch =  1: Cumulative Time = 32.635, Epoch Time = 32.635, Images/sec = 1835.55322265625, Validation Loss = 0.278, Validation Accuracy = 0.903
Epoch =  2: Cumulative Time = 64.444, Epoch Time = 31.809, Images/sec = 1881.526611328125, Validation Loss = 0.259, Validation Accuracy = 0.909

![image](https://github.com/user-attachments/assets/bf2bb89c-5f97-4b8b-899f-fa5ac0bde8bf)

Until now the batch size = 32 and base_lr = 0.01 - good right ? 

## Scaling Learning Rate with Batch Size

* Lets recall that inc batch size (we loose the variance in small batches). So, we need to follow the rule that we need to inc the learning rate to put back the variance in these large batches. But ours is a large batch ? Yes ! by doing torch.All_reduce - weights are updated after the backprop results are synched across all the GPUs or all GPUs mini batch now effectively form a bigger batch (Big batch = num_GPU*Large_Mini_Batch). But yeah we get the benefit of throughput increment right. See the _Multi GPU Speedup factor_ !!

![image](https://github.com/user-attachments/assets/7411a600-df54-492e-9e23-9ce905f2da33)


* Now try separate runs with Batch size: 128 to 256 and base_lr = 0.08 & higher. Only increasing both at the same time will help converge and we see val_acc > 0.85. Otherwise we will see Val_acc = 0.1

* !python3 fashion_mnist.py --num-gpus 4 --base-lr 0.08 --batch-size 128 

![image](https://github.com/user-attachments/assets/6cac71b3-ddaf-4715-9a1b-c345f9cb9df4)

*  !python3 fashion_mnist.py --num-gpus 4 --base-lr 0.08 --batch-size 256 

** The below image shows that whatever the batch size and base_lr used - doesnt help with inc in val_accuracy. 
![image](https://github.com/user-attachments/assets/2ec1e0c2-c1ea-4af8-8db4-6392ece92205)


**IMP Observation: 
**
* Now the validation accuracy still stays around 0.1 - Guess we should have stayed with batch size = 32 and base_lr = 0.01 ? But to test with higher batch_szies - coz we need to exploit the Multi GPU scalablity - we added momentum to SGD. But we should have stayed with lesser base_lr and less_batch_size& then the working code at val_acc = 0.8. Maybe we can fix the learning rate and batch issues ? Yes !! Read on !!

# Learning Rate Warm up: 

A high enough learning rate caused the network to never converge in your previous stage. In this scenario, the weights are updated with a high enough magnitude that they overshoot, and never end up finding the slope toward a minimum. 

With this approach, the learning rate will start at a fraction of the target value, and slowly scale up over a series of epochs. This allows the network to move slowly at first, taking "careful steps" as it finds a slope toward the minimum. As the learning rate increases to the target value, the benefits of the larger learning rate will take effect.

Pytorch code: 
torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/num_gpus, total_iters=5). With the parameters as prescribed, over the first 5 epochs, the learning rate will linearly increase, starting from the learning rate / the number of GPUs.

Why learning rate / the number of GPUs ?  Since here we are defining for a single GPU, the learning rate is 1/all GPUs. LR for N GPUs = LR for 1 GPU × N. So if your 1-GPU learning rate is 0.01, and you run on 8 GPUs, you set 0.08

![image](https://github.com/user-attachments/assets/8cf0f43d-81e8-480f-a71a-e50a9db80d57)

We see that for 32 Batch size and base_lr of 0.06 - we got a good result. Is it all good ? 

Next problem: For even complex/large datasets, we need to use higher batch_size and higher learning rates - where this convergence may not be observed. 

!python3 fashion_mnist.py --num-gpus 4 --base-lr 0.25 --batch-size 256

![image](https://github.com/user-attachments/assets/c98013fa-c766-485d-ace1-3e9ed2d27d6b)
Above image showes a bit longer time / epochs but did converge finally. But the degraded perf in timing is seen or sometimes this is called 

Hence researchers have done their job to give us the next set of optimizers (as mentioned in the theory section of DDP - one day older GIT repo here). Using NovoGrad Optimizer should help !

Novograd optimizer: 
It updates weights by - Δ𝐰=−𝜆𝐦
The grad_averaging parameter, which weights the momentum using a mix of the current and previous steps (like Adam), is empirically helpful for this problem.

    

