# 🧠 Day 13: Optimizing a DL Model

## 🚫 Why GPUs Not Go Brrr?

Despite having massive compute potential, GPUs often **don’t go brrr** due to:

- **Memory Bottlenecks** – Limiting throughput due to excessive global memory access.
- **Overhead** – Time wasted in non-compute activities (Python interpreter, kernel launch delays, etc.).

Our goal: ✅ Relieve these bottlenecks and make sure **GPUs go brrr** 💨.

---

## 🧵 Memory Bound?

To reduce global memory accesses, **operator fusion** is key. This combines multiple operations into one GPU kernel, minimizing memory I/O.

### ✅ Operator Fusion Example:
![Operator Fusion](https://github.com/user-attachments/assets/d19c1dce-6869-4d5c-b260-62fea339ebbd)

### How?

- **PyTorch Eager Mode** already applies some fusion.
- **Compilers like `nvFuser` and `XLA`** can do more advanced fusions.
- But for peak performance, **handwriting custom CUDA kernels** is king 👑.

> 🎯 Motivation unlocked: Time to learn GPU kernels! 🔥

---

## 🔢 Compute Bound?

One way to identify if you're compute-bound:
- Measure **achieved FLOPS** as a % of your GPU’s **peak theoretical FLOPS**.

If this number is low, you're either:
- Limited by memory access (again), or
- Not launching enough parallel work to saturate GPU compute resources.

---

## ⚙️ Overhead?

Overhead = Time spent **not doing actual compute or tensor transfers**.


-- PENDING ------

CPU vs GPU grapha nd the nvidia-smi giving chance to see this - ?


------


Examples:
- Python interpreter time 🐍
- PyTorch framework internal time 🧱
- CUDA kernel *launch* delays (not execution!) 🚀

Torch Dynamo - using best of both worlds ? 
https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361

PyTorch Dynamo: This is a Python-level Just-In-Time (JIT) compiler that works as a tracer within PyTorch's function. Its purpose is to make PyTorch programs faster by capturing the model's operations into an optimized, executable graph. It does this by analyzing bytecode and mixing Python execution with compiled backends.


PyTorch Dynamo Graphs (FxGraphs): When PyTorch Dynamo compiles your code, it traces the Python operations and builds a representation of the model's computation as an FxGraph. This FxGraph is essentially a high-level Intermediate Representation (IR) that captures the sequence of operations in your PyTorch model. It's a graph of PyTorch operations, not a graph of CUDA kernels.



### Visualization:
![Overhead Breakdown](https://github.com/user-attachments/assets/e288d349-5d27-4754-8b82-dd7031096b75)

---

## 🔍 Application: Profiling U-Net

Let’s now apply all the concepts above and **profile a U-Net** model:
- Identify memory bottlenecks
- Look at kernel launch vs execution
- Measure utilization
- Use fusion and/or custom kernels where needed


Included the folloowing code: 

    model.eval()  # this enables model for production level inference setting - no norm / zeroing optimzations 
    # grab exactly one sample from your test loader
    sample = next(iter(test_loader))
    inputs = sample['input'].to(self.device)
  
    with torch.no_grad(), profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(exp_dir/ "ppnet_profiler"))
        ) as prof:
        # this is the single forward pass we’ll trace
        _ = model(inputs)
  
    # (optional) dump a Chrome‐compatible JSON for chrome://tracing
    prof.export_chrome_trace(str(exp_dir / "ppnet_trace.json"))
  






---

## 📚 References

- 📝 [Horace.io: GPU Go Brrr!](https://horace.io/brrr_intro.html)
- 📝 [Pytorch Dynamo] (https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361)

- 💡 [Inference Optimization Blog Series](https://github.com/vdesai2014/inference-optimization-blog-post/tree/main/part-3)
