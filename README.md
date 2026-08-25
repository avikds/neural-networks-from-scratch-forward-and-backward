# Neural Networks From Scratch: Forward and Backward

Implement a complete neural network stack from scratch in NumPy: finite-difference gradient checks, dense and activation layers, loss, sequential composition, optimizers, and a training loop that overfits then generalizes on a nonlinear dataset.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** numerical_gradient
- [x] **2.** gradient_check
- [x] **3.** make_dense
- [x] **4.** make_activation
- [x] **5.** initialize_weights
- [x] **6.** make_loss
- [x] **7.** make_sequential
- [x] **8.** forward_backward
- [x] **9.** make_optimizer
- [x] **10.** train_step
- [x] **11.** train
- [x] **12.** design_network
- [x] **13.** improve_generalization

## Results

```
design_network_accuracy: 0.9962
initial_batch_loss: 0.687444810499734
train_step_loss: 0.687444810499734
after_train_step_loss: 0.6378195833985174
overfit_loss_start: 0.62087684457059
overfit_loss_end: 0.017222366745095224
baseline_val_accuracy: 1.0
improved_val_accuracy: 0.9844
```
