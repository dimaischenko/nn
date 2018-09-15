from nn.modules import Module


class Sequential(Module):
    """Sequantial container with list of modules"""

    def __init__(self, modules=[]):
        super(Sequential, self).__init__()
        self.modules = modules

    def add(self, module):
        self.modules.append(module)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def get_params(self):
        return [param for module in self.modules
                for param in module.get_params()]

    def get_grad_params(self):
        return [grad_param for module in self.modules
                for grad_param in module.get_grad_params()]

    def forward(self, input_):
        self.output = input_

        for module in self.modules:
            self.output = module.forward(self.output)

        return self.output

    def backward(self, grad_output):
        if not len(self.modules):
            # TODO: is it right?
            self.grad_input = grad_output
        else:
            self.grad_input = self.modules[-1].backward(grad_output)

            for module in self.modules[:-1][::-1]:
                self.grad_input = module.backward(self.grad_input)

        return self.grad_input
