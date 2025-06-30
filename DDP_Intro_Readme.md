Here's a cleaned up and properly formatted version of your GitHub README in Markdown syntax:

---

# ⚡ Why Do We Need PyTorch DDP?

To accommodate large dataset training workloads, we distribute the training across multiple GPUs using **PyTorch Distributed Data Parallel (DDP)**.

> **Note**: All images used here are for educational purposes only. I do not claim authorship.

---

## 🔧 DDP Definitions

* **Node**: One physical machine with a CPU and one or more GPUs. DDP can span across single or multiple such nodes.
* **Global Rank vs Local Rank**:

  * **Local Rank**: GPU ID within a single node (e.g., 0–3 for 4 GPUs).
  * **Global Rank**: Unique ID across all GPUs in all nodes (e.g., 0–7 for 2 nodes × 4 GPUs each).

### 🖼️ Single Node

![Single Node](https://github.com/user-attachments/assets/974c4659-419a-44cc-8f81-7d5337ce6e12)

### 🖼️ Dual Node

![Dual Node](https://github.com/user-attachments/assets/1399dea6-cd36-4522-a26e-dce394e72e84)

---

## 📥 Downloading & Logging Strategy

* **Download Dataset Once per Node**: Use `local_rank == 0` to handle downloads.
* **Log Only Once per System**: Use `global_rank == 0` to print logs across distributed setup.

---

## 🔁 DDP Flow

```python
# Step 1: Initialize process group
def setup(global_rank, world_size):
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

# Step 2: Pin GPU to process
device = torch.device("cuda:" + str(local_rank))
model = Net().to(device)

# Step 3: Wrap model with DDP
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# Step 4: Partition data
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set, num_replicas=world_size, rank=global_rank
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, sampler=train_sampler
)
```

---

## ⚖️ Data vs Model Parallelism

![Data vs Model Parallelism](https://github.com/user-attachments/assets/62e334fb-c387-4297-bc57-7d5b30eaaccd)

---

# 💡 Main Tips and Tricks for Multi-GPU DDP Training

---

## 🎯 Batch Size & Gradient Noise

* **Batch Size**: Number of samples processed in one forward/backward pass.
* **Smaller Batches**: Add gradient noise (variance) → help in generalization.
* **Larger Batches**: Reduce noise → better convergence but risk overfitting.

![Small vs Large Batch](https://github.com/user-attachments/assets/bc2819c3-1263-443e-bd0c-2a4b5b4aa106)
![Noise Illustration](https://github.com/user-attachments/assets/8a34fc33-b493-44a7-be56-47f0856a97c4)

### ⚠️ Tradeoff

> After a certain batch size, the benefits plateau. Larger batch size can lead to **sharp minima**, poor generalization.

![Problem Visualization](https://github.com/user-attachments/assets/b7765789-cd0c-4e72-a072-2f25498ace6a)

But lets see the imapct of batch size in different cases
![Sharp vs Flat](https://github.com/user-attachments/assets/47a8145e-1a25-4c31-8c2a-490315d3c438)

---

## 📉 Large Minibatch and Its Impact on Accuracy

* Larger batches reduce training time, but need learning rate adjustments:

  $$
  \eta' = \eta \cdot \sqrt{k} \quad \text{(theory)}
  $$

![Minima Impact](https://github.com/user-attachments/assets/d53c99d0-ee65-402c-931f-acd13ee3c900)
![Loss Curves](https://github.com/user-attachments/assets/9c02c6e0-b072-4693-be27-6377daf3f0d0)

* Weight decay (L2 regularization) helps but can't fully offset large batch effects:

![Weight Decay Effect](https://github.com/user-attachments/assets/0eea9559-2fa1-4373-9b6d-f7270fdaaa24)

---

## 🛠️ Optimization Strategies / Solutions

### Problems in Large-Batch Multi-GPU Training:

* **Vanishing updates**: Gradients exploding due to higher training rates. 
* **Poor generalization**: Sharp minima 
* **Imbalanced layer updates**: Some deeper layers may learn too less or too much. 

---

## 🔄 When to Use Which Optimizer

![Optimizers Summary](https://github.com/user-attachments/assets/d16af335-d3a4-4e8c-b74a-8bcc5869610a)

---

## ✅ Conclusion

> Using **large mini-batches** improves hardware utilization but may lead to **sharp minima** and **poorer generalization**.
> To mitigate:

* Use **learning rate warmup**
* **Normalize** (BN, Ghost BN)
* Consider optimizers like **LARS, LAMB, NovoGrad**
* **Inject gradient noise** if needed
![image](https://github.com/user-attachments/assets/e55b9096-4e94-44e7-930a-07423d335320)

Why warmup ? 

![image](https://github.com/user-attachments/assets/265d523b-da58-4249-af60-9fbb66cfd22b)

---

# 🧠 Neural Network Miscellaneous

### 🔤 Common Definitions

| Term             | Description                                     |
| ---------------- | ----------------------------------------------- |
| Loss             | Scalar error between predicted and true outputs |
| Forward Pass     | Computation from input to output                |
| Activations      | Output of neurons                               |
| Activation Funcs | Introduce non-linearity (ReLU, Sigmoid)         |
| Weights          | Learnable parameters                            |
| Backpropagation  | Computes gradients                              |
| Gradient         | Derivative of loss w\.r.t. weights              |
| Optimizer        | Updates weights (SGD, Adam)                     |

---

## 🗻 Global vs Local Minima / Sharp vs Flat

![Minima Types](https://github.com/user-attachments/assets/d9115025-0c65-4717-96e2-6d4839ec1b34)

---

## 🚧 Non-Convex Loss

![Non-Convex](https://github.com/user-attachments/assets/71d0440e-d012-4e8a-8214-0cdda70c9994)

### ➕ Skip Connections Ease Optimization

![Skip Connections](https://github.com/user-attachments/assets/d8d80773-0cdb-4857-9409-8cc56a05e8f2)

---

## 📚 References

* NVIDIA DLI: Data Parallelism Workshop
* Goyal et al., 2017. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
* Li et al., 2017. *Visualizing the Loss Landscape of Neural Nets* [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)

---

Let me know if you'd like a version with collapsible sections, or rendered diagrams locally instead of image links.
