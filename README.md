# Flash-Attention-Triton-kernel-Bloc-Sliding-Window
A Triton implementation of flash attention with a sliding window block attention pattern (SWBA). Very useful for processing videos with a transformer in an autoregressive manner.


## Attention pattern

This kernel is defined by two parameters : the bloc size (ex : the number of tokens per frames) and the window size (the contexte size).


<div style="text-align: center;">
  <img src="./att_pattern.png" alt="attention pattern" width="300">
</div>


## Performance vs torch implementation

Coming Soon

## Usage

This kernel is designed for fp16 precision.
The bloc size need to be divisible by 64 and 128 (You can modify BLOC_M and BLOC_N parameters to avoid this constraint).

```python
import torch
from kernel.SWBA import attention

B = 4 # batch size
NH = 12  # num heads
N = 4096 # num tokens 
DH = 64 # head dim

BS = 256 # bloc size
WS = 4 # window size

# BS % BLOCK_M == 128/64 and BS % BLOCK_N == 128/64
assert BS % 128 == 0 
assert BS % 64 == 0 

sm_scale = 1/(D_HEAD**0.5)

q = torch.randn((B,NH,N,DH), dtype = torch.float16).cuda().requires_grad_()
k = torch.randn((B,NH,N,DH), dtype = torch.float16).cuda().requires_grad_()
v = torch.randn((B,NH,N,DH), dtype = torch.float16).cuda().requires_grad_()

do = torch.randn_like(q)

out = attention(q, k, v, BS, WS, sm_scale) # forward

out.backward(do) # backward
```



