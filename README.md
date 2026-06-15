# Introduction and Problem description :

This repository implements and benchmarks an optimized Physics-Informed Neural Network (PINN). 
It introduces an Accelerated Random Projection PINN architecture, inspired by Extreme Learning Machines / Random Feature Networks, to solve a 1D Partial Differential Equation (PDE) modeling malware propagation across peer-to-peer (P2P) computer networks.

I wanted to find a way to make Physics-Informed Neural Networks (PINNs) run faster and use less memory without destroying their accuracy. To test this, I built a simulation that models how malware spreads across a peer-to-peer (P2P) network over time. 

# Theory :

The physics of this malware movement is governed by a 1D Reaction Diffusion partial differential equation:
∂u/∂t = (D * (∂²u/∂x²)) + (R * u * (1 - u))

u = u(x,t).

# Procedure :

To solve this, I wrote and compared two distinct neural network architectures in PyTorch:

i. The Baseline Model: I set up a standard, 4-layer Deep Multi-Layer Perceptron. I left every single layer unfrozen so that backpropagation would actively update all 8,577 parameters.

ii. The Proposed Model: I designed an Accelerated Random PINN inspired by Extreme Learning Machines. I scaled the hidden dimension up to 120 nodes but deliberately froze the input-to-hidden layer weights using requires_grad=False. This created a strict Optimization Barrier, meaning my Adam optimizer only had to train the final 120 output parameters.

iii. I then built a custom benchmarking engine using PyTorch and Matplotlib to run both models through 250 epochs over 5,000 spatial-temporal points, printing a clean technical report and generating an architectural comparison diagram at the end.

# Results and Observations :

Running this experiment taught me a massive lesson about the realities of training physics-informed AI models:

i. The Speed-Accuracy Trade-off is Real: My proposed model cut trainable parameters down by an incredible 98.60%. Because of this, it ran 2.38 times faster than the baseline. However, because it relies on random, static shapes rather than actively learning new ones, its final convergence loss was slightly higher, 0.000067 vs the baseline's 0.000016. For a real-time cyber defense system on edge hardware, this tiny loss in accuracy is a totally acceptable tradeoff for cutting training time in half.

ii. The "Autograd" Bottleneck: My biggest breakthrough in understanding was realizing why a 98.6% reduction in parameters didn't result in a 98% reduction in runtime. In a PINN, the real performance killer isn't updating the weights—it is calculating the physics derivatives (∂²u/∂x² and ∂u/∂t). Even though my hidden layer weights were frozen, PyTorch's autograd engine still had to travel all the way back through those layers to calculate how the output changed relative to the input coordinates (x, t).

iii. Finally, I proved that freezing random projection layers is a highly viable way to strip away optimizer memory overhead. This approach makes PINNs practical for resource-constrained environments, even if the automatic differentiation step keeps a hard cap on raw execution speed.
