import torch
device=("cuda" if torch.cuda.is_available() else "cpu")
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        self.labels = torch.Tensor([[0,1]]) #useless
        labels = self.labels
        self.toa = torch.Tensor([[45]]) #useless
        toa = self.toa
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compitability with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x, labels, toa):
        self.gradients = []
        self.activations = []
        x = torch.unsqueeze(x,0)
        with torch.backends.cudnn.flags(enabled=False):
            output = self.model(x, torch.Tensor([[1,0]]).to(device),torch.Tensor([[45]]).to(device))
        return output

    def release(self):
        for handle in self.handles:
            handle.remove()
