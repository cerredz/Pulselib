# PyTorch Codebase Best Practices

## 1. Limiting CPU to GPU Switches
**Description**  
Keep tensors on the device they were created on for as long as possible. Aggregate with torch ops on-device, and only call `.item()` or move to CPU at the final logging, printing, or serialization boundary.

**Do**
```python
episode_reward = rewards.sum()
td_errors = torch.stack(step_td_errors)
mean_td_error = td_errors.mean()
logger.log("episode complete", {"td_error": mean_td_error})
print(f"TD Error: {mean_td_error.item():.4f}")
```

**Do Not**
```python
episode_reward = float(rewards.sum().item())
mean_td_error = np.mean([float(td.item()) for td in step_td_errors])
```

## 2. Prefer Batched Torch Reductions Over Python Loops
**Description**  
When computing benchmark aggregates, stack scalar tensors and reduce them with `torch.mean`, `torch.min`, `torch.max`, and `torch.std` instead of using Python lists plus NumPy.

**Do**
```python
q_values = torch.stack(step_q_values)
summary = {
    "q_mean": q_values.mean(),
    "q_min": q_values.min(),
    "q_max": q_values.max(),
}
```

**Do Not**
```python
summary = {
    "q_mean": np.mean(step_q_values),
    "q_min": np.min(step_q_values),
    "q_max": np.max(step_q_values),
}
```

## 3. Keep Optimizer-Side Metrics in the Training Device Path
**Description**  
Losses, gradient norms, TD errors, and clipping indicators should stay as scalar tensors during the training step so follow-on computations can remain on the accelerator.

**Do**
```python
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
clip_rate = torch.where(
    grad_norm > torch.tensor(1.0, device=grad_norm.device, dtype=grad_norm.dtype),
    torch.ones((), device=grad_norm.device, dtype=grad_norm.dtype),
    torch.zeros((), device=grad_norm.device, dtype=grad_norm.dtype),
)
```

**Do Not**
```python
loss.backward()
grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
clip_rate = 1.0 if grad_norm > 1.0 else 0.0
```

## 4. Use Tensor-Native Trend and Regression Calculations
**Description**  
If a training benchmark needs a slope, trend, or simple fit, compute it with torch tensor math instead of converting to NumPy for `polyfit`.

**Do**
```python
x = torch.arange(values.shape[0], device=values.device, dtype=values.dtype)
x_centered = x - x.mean()
y_centered = values - values.mean()
slope = (x_centered * y_centered).sum() / (x_centered * x_centered).sum()
```

**Do Not**
```python
slope = np.polyfit(np.arange(len(values)), values, 1)[0]
```

## 5. Convert to Python Scalars Only at the Output Boundary
**Description**  
Printing, JSON serialization, and CLI summaries are the right place to convert scalar tensors to Python values. Keep the rest of the pipeline tensor-native.

**Do**
```python
final_metrics = {"reward_std": reward_std_tensor}
print(f"Reward Std Dev: {final_metrics['reward_std'].item():.2f}")
```

**Do Not**
```python
final_metrics = {"reward_std": float(reward_std_tensor.item())}
```
