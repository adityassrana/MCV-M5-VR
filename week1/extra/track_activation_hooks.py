import matplotlib.pyplot as plt
from functools import partial
"""
These experiments were performed adding these lines to the main training script.
Only the required modules are presented here to reduce code redundany and increase
visibility of this code.

In short, instead of having to always manually grab activations, PyTorch offers us 
a quick way to do this by registering hooks with network layers. PyTorch hooks can 
be added to any nn.Module. When created, a hook must be registered to a specific layer. 
Then during training, the hook will be executed during the forward pass (using forward hooks) 
or during the backward pass (backward hooks).
"""

def plot_layer_stat(layer_outputs, metric, title, num_layers_shown, cutoff=None):
    for l in layer_outputs[:num_layers_shown]: 
        plt.plot(l) if cutoff==None else plt.plot(l[:cutoff])
    plt.xlabel('Iterations')
    plt.ylabel(f'Activation Output {metric}')
    plt.title(title)
    plt.legend([f'Convolutional Layer {i}' for i in range(num_layers_shown)]);

def children(l):
    return list(l.children())

class ForwardHook():
    def __init__(self, l, f): self.hook = l.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()

# for tracking means and standard deviations
def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, 'stats'): hook.stats = ([], [])
    means, stds = hook.stats
    if mod.training:
        means.append(outp.data.mean())
        stds .append(outp.data.std())

model = get_model()

# attach hooks to layers
hooks = [ForwardHook(l, append_stats) for l in children(model[:5])]

# run training loop for 1 epoch
# model.fit(1)

plot_layer_stat([h.stats[0] for h in hooks], 'Mean', 'Layer Average Activation', 4)
plot_layer_stat(model.act_stds, 'Standard Deviation', 'Layer Standard Deviation', 4)

# to clear up GPU memory
for h in hooks:
    h.remove()
