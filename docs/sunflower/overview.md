# 🌻 Sunflower Quantized Inference

## Overview
The **Sunflower** models are available in **14B** and **32B** sizes and support **8-bit** and **4-bit quantized inference** for efficient performance on GPUs with limited memory.  

Quantization reduces memory requirements while keeping inference quality high, enabling large models to run on consumer-grade GPUs or hardware with limited VRAM.

| Feature          | 8-bit                    | 4-bit                      |
|-----------------|--------------------------|----------------------------|
| Memory Usage     |  Higher (~16GB for 14B) |  Lower (~10GB for 14B) |
| Speed            | ⚡ Fast                   | ⚡⚡ Faster                |
| Accuracy         |  Very Good              |  Slightly Lower         |
| VRAM Efficiency  |  Moderate               |  High                    |

> ⚠️ **Important:** Do not set both 8-bit and 4-bit modes at the same time.

---

## 🌻 Sunflower 14B Models

- **14B 8-bit:** Balanced memory and accuracy, suitable for most GPUs.  
- **14B 4-bit:** Optimized for memory-limited GPUs and faster inference, with minimal accuracy trade-off.  

---

## 🌻 Sunflower 32B Models

- **32B 8-bit:** High accuracy, requires more GPU memory.  
- **32B 4-bit:** Reduced memory usage, faster inference, slightly lower accuracy.  

> The usage process is identical for 14B and 32B; only model size and quantization type differ.


## Tips & Best Practices

- Use 4-bit models when **GPU memory is limited** or faster inference is needed.  
- 8-bit models offer a **good balance of memory usage and accuracy**.  
- Always choose **either 8-bit or 4-bit** for a model.  
- For large inputs or batch processing, monitor GPU memory to avoid out-of-memory errors.  
- Adjust inference parameters (like sequence length or tokens) for optimal performance based on your hardware.

---


