# Why do we need Pytorch DDP: 
(Note: All the images used are for educational purpose and I dont claim authroship to any) 

To accomadate large data size trainings, we can distribute the trainings across multiple GPUs. 

# DDP Defintions: 
  Node: One physical CPU-GPU/Multiple GPU machine. We may use one or more of such nodes for DDP. 
  Global and Local Rank: In the context of a single node (say with 4 GPUs), local rank is from 0 to 3 & Global rank is also the same. In case of a dual node system (i.e 8 GPUs), now the Global rank across this whole system is 0 to 7 and local rank in the respective node will still be 0 to 3. 
    
  **  Singel Node: 
  **  
  ![image](https://github.com/user-attachments/assets/974c4659-419a-44cc-8f81-7d5337ce6e12)
  **  Dual Node: 
  **  
  ![image](https://github.com/user-attachments/assets/1399dea6-cd36-4522-a26e-dce394e72e84)
  
  Download only once per node: How ? - Download for the first GPU i.e Local rank = 0 
  Logging only once across system: How ? - Just log prints only for the total system i.e Gloabl Rank = 0

DDP Flow: 
* INITIALIZE THE PROCESS
  def setup(global_rank, world_size):
  dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

PIN GPU TO BE USED
  device = torch.device("cuda:" + str(local_rank))
  model = Net().to(device)

ENCAPSULATE MODEL WITH DDP
  model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

DATA PARTIONING: Shuffle the dataset Partition records among workers. Train by sequentially reading the partition. After epoch is done, reshuffle and partition again. 
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=global_rank)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)

* Data vs Model Parallelism  
![image](https://github.com/user-attachments/assets/62e334fb-c387-4297-bc57-7d5b30eaaccd)

# Main Tips and Tricks for Multi GPU DDP Trainings:

# Batch Size and mitigating Noise: 
Batch Size: In one forward or backward pass, how many samples do we pass. 
* But we get noise / variance in the output (more deviation from our output), if we use smaller batches. So, smaller batches are completely useless ? Read on !!!
![image](https://github.com/user-attachments/assets/bc2819c3-1263-443e-bd0c-2a4b5b4aa106)

![image](https://github.com/user-attachments/assets/8a34fc33-b493-44a7-be56-47f0856a97c4)


So, can we increase the batch size and use Multi GPUs effectively for this case ?? After a certain batch size - the benefit is not seen. There are limits. 
Also the large batch size will have sharp minima in testing stage and leads to poor genaralization and sometimes we also need small batch size & its noise which helps in having variance to explore all over and not quickly overhsoot the space / by not exploring well / generalizing.  (see more in the below section)
![image](https://github.com/user-attachments/assets/47a8145e-1a25-4c31-8c2a-490315d3c438)



## LARGE MINIBATCH AND ITS IMPACT ON ACCURACY: 
A large minibatch - so put a higher learning rate to explore all of that big batch ? As it helps lessen the training time ? Coz going by the rule: “Theory suggests that when multiplying the batch size by k, one
should multiply the learning rate by √(k) to keep the variance in the
gradient expectation constant.

![image](https://github.com/user-attachments/assets/d53c99d0-ee65-402c-931f-acd13ee3c900)
![image](https://github.com/user-attachments/assets/9c02c6e0-b072-4693-be27-6377daf3f0d0)

We can see flat (good in general) and sharp minima(bad) for inc batch sizes in the same learning rate category. 
Adding weight decay (L2 regularization) helps mitigate this but does not fully eliminate the issue for very large batches.

![image](https://github.com/user-attachments/assets/0eea9559-2fa1-4373-9b6d-f7270fdaaa24)


![image](https://github.com/user-attachments/assets/b7765789-cd0c-4e72-a072-2f25498ace6a)

![image](https://github.com/user-attachments/assets/b36efe51-7c9d-4f9e-965a-a5a5da807d09)


What can we do this for problem: 
* Manipulate the learning rate - like increase the learning rate and use batch normalization (which allows for higher learning rate) ?
• Add noise to the gradient?
• Manipulate the batch size?
• Change the learning algorithm?

### Optimization Strategies / Solutions: 

Optimizers like SGD and Adam were designed for typical batch sizes (e.g., 32–512). When scaling up to large batch sizes (8192+, especially across GPUs), issues arise:

❌ Problems:
Vanishing updates: When batch size increases, gradient variance decreases → smaller updates → training stagnates.

Poor generalization: Large batches can lead to sharp minima (as discussed).

Imbalanced learning: Some layers (especially in deep nets) update too little or too much. This is IMPORTANT !!!!!

![image](https://github.com/user-attachments/assets/4fc24a38-fc58-4e49-b8e0-b286238a8de3)

When to use each of this new optimizer in a Multi GPU - Large Batch Case: 
![image](https://github.com/user-attachments/assets/d16af335-d3a4-4e8c-b74a-8bcc5869610a)

**Conclusion:** Using large mini-batches improves hardware efficiency but poses risks to generalization performance due to converging to sharp minima. 
Proper learning rate adjustments, warmup, and normalization techniques are essential to mitigate these issues and maintain accuracy.


# Neural Network Misceleanous: 

* NN defintions: 
   Loss – A scalar value representing the error between predicted and true outputs - resulting from a typical forward pass.
   Forward Pass – Computation of activations layer by layer using current weights.
   Activations – Outputs of neurons after applying weights, biases, and activation functions.
   Activation functions - Which introduce non linearity like RELU, sigmoid. 
   Weights – Trainable parameters of the neural network that connect neurons across layers.

   Optimization – Uses a Optimizer algorithm that updates weights using gradients / the results of back propagation(e.g., SGD, Adam) to reach global minima and not any local minima.
   Backpropagation – Algorithm to compute gradients of the loss with respect to each weight.  
   Gradient – Partial derivative of the loss with respect to a weight, used for updates.
   Global minima vs local minima & Sharp vs Flat Minima:
    ![image](https://github.com/user-attachments/assets/d9115025-0c65-4717-96e2-6d4839ec1b34)
 

* Non Convex Loss:
![image](https://github.com/user-attachments/assets/71d0440e-d012-4e8a-8214-0cdda70c9994)
 
* With Skip Connections easier to see global minima:

![image](https://github.com/user-attachments/assets/d8d80773-0cdb-4857-9409-8cc56a05e8f2)


Ref: 

Nvidia & Pytorch DDP 
