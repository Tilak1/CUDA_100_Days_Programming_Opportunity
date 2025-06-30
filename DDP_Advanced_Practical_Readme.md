---

# ⚡ DDP Advanced / Practical

This guide explores practical aspects of using PyTorch DistributedDataParallel (DDP) for training models like Fashion MNIST at scale using multiple GPUs.

---

## 🚀 More GPUs = More Throughput

### Single GPU:

```bash
!python3 fashion_mnist.py --epochs 3 --num-gpus 1 --batch-size 128
```

Output:

> 1 GPU: Images/sec = 622

### Multi-GPU:

```bash
!python3 fashion_mnist.py --epochs 5 --num-gpus 4 --batch-size 128
```

Output:

> 4 GPU: Images/sec = 1835

**Clearly the Multi GPU Speedup factor:**

* 1st run: 1835 / 620 ≈ **3x**
* 2nd run: 2400 / 620 ≈ **4x**

> Note: It stands to reason that each GPU would add a significant boost to the training process.
> We don’t get perfectly linear scaling. A significant component of this is due to communication between the GPUs when updating weights.
> There can also be other factors like waiting for slower GPUs to finish.
> Still, this is **pretty good**.

> Production DL training at scale is usually benchmarked against the ideal case of linear scaling (N GPUs should be N times faster than 1 GPU).
> DDP and the NCCL library do a good job of maintaining high throughput. But performance is **tied to the hardware in use**, and as we scale to **multi-node**, further hardware-level considerations are needed.

---

## 🎯 Multi-GPU Effect on Validation Accuracy

> 🧪 *Pending additional experiments*

---

## ⚙️ Playing with Optimizer

### Recap:

In SGD, weights are updated based on their gradient with respect to the loss of a mini-batch. This process works well, but can **slow down near local minima** or **saddle points**.

One fix: **Momentum**

It allows updates to retain memory of previous gradients. If weights are consistently moving in one direction, momentum helps push further in that same direction.

> Think of a ball rolling downhill: it builds momentum and smooths out small bumps.

---

### Adding Momentum

Step 1: Add `momentum` argument

```python
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
```

Step 2: Use it in the optimizer

```python
optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum)
```

**Results:**

```
Epoch 1: Cumulative Time = 32.635, Images/sec = 1835.55, Val Loss = 0.278, Val Acc = 0.903  
Epoch 2: Cumulative Time = 64.444, Images/sec = 1881.52, Val Loss = 0.259, Val Acc = 0.909
```

![image](https://github.com/user-attachments/assets/bf2bb89c-5f97-4b8b-899f-fa5ac0bde8bf)

Until now: `batch_size = 32`, `base_lr = 0.01` – **Good, right?**

---

## 📈 Scaling Learning Rate with Batch Size

Larger batch = less gradient noise. So we must increase learning rate to maintain convergence behavior.

> In DDP, using `torch.all_reduce`, all GPUs’ gradients are averaged, so their combined mini-batches act like a **larger batch**.

**Yes! We get throughput gains!**
See: *Multi GPU Speedup factor*

![image](https://github.com/user-attachments/assets/7411a600-df54-492e-9e23-9ce905f2da33)

---

### Run Experiments:

```bash
# Higher base_lr and batch-size
!python3 fashion_mnist.py --num-gpus 4 --base-lr 0.08 --batch-size 128
!python3 fashion_mnist.py --num-gpus 4 --base-lr 0.08 --batch-size 256
```

**But what happens?**
Sometimes accuracy doesn't improve. See below.

![image](https://github.com/user-attachments/assets/6cac71b3-ddaf-4715-9a1b-c345f9cb9df4)
![image](https://github.com/user-attachments/assets/2ec1e0c2-c1ea-4af8-8db4-6392ece92205)

---

## 🔎 Important Observation:

Despite trying momentum and higher base\_lr, **validation accuracy is still stuck around 0.1**.

Maybe we should’ve stuck to:

* `batch_size = 32`
* `base_lr = 0.01`

But to leverage DDP scalability, we **must** scale batch size and LR. Let’s fix it!

---

## 🔥 Learning Rate Warm-Up

High LR from the start can prevent convergence (weights overshoot).
Solution: **start small and ramp up LR** over a few epochs.

```python
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1/args.num_gpus,
    total_iters=5
)
```

> Why divide LR by num GPUs?
> Because:
> `LR_N_GPU = LR_1_GPU × N`

If 1-GPU LR = 0.01 → 8-GPU LR = 0.08

![image](https://github.com/user-attachments/assets/8cf0f43d-81e8-480f-a71a-e50a9db80d57)

### Result:

With `batch_size = 32`, `base_lr = 0.06`, and warmup → **good results** ✅

---

### Trying Higher Again:

```bash
!python3 fashion_mnist.py --num-gpus 4 --base-lr 0.25 --batch-size 256
```

![image](https://github.com/user-attachments/assets/c98013fa-c766-485d-ace1-3e9ed2d27d6b)

We see it **did converge eventually** – but took longer, with some degradation in timing.

---

## 🧠 Enter: NovoGrad Optimizer

To overcome large batch training challenges, use a better optimizer.

```python
import torch_optimizer as opt
optimizer = opt.NovoGrad(model.parameters(), lr=args.base_lr, grad_averaging=True)
```

> `Δw = -λ * m` (momentum-like update)

`grad_averaging=True` helps smooth updates (like Adam).

### Results with NovoGrad:

![image](https://github.com/user-attachments/assets/a9ea670d-313e-4783-8a21-35593512a612)

**✅ Best in class for now.**
Where `SGD` failed with `batch_size=256` and `base_lr=0.25`, `NovoGrad` succeeded.

---

## ✅ Final Takeaways

| Config                   | Batch Size | Base LR | Warm-Up | Optimizer    | Result      |
| ------------------------ | ---------- | ------- | ------- | ------------ | ----------- |
| Vanilla SGD              | 32         | 0.01    | ❌       | SGD          | ✅ Good      |
| SGD + Momentum           | 32         | 0.01    | ❌       | SGD+Momentum | ✅ Better    |
| High BS & LR (No Warmup) | 256        | 0.25    | ❌       | SGD          | ❌ Failed    |
| High BS & LR + Warmup    | 256        | 0.25    | ✅       | SGD          | ✅ Converged |
| NovoGrad + High BS & LR  | 256        | 0.25    | ✅       | NovoGrad     | ✅ Best      |

---
