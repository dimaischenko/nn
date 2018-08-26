from nn.modules import Module


class Sequential(Module):
    """Sequantial container with list of modules"""

    def __init__(self, modules=[]):
        super(Sequential, self).__init__()

        self.modules = modules

    def add(self, module):
        self.modules.append(module)

    def forward(self, input_):
        output = input_

        for module in self.modules:
            output = module.forward(output)

        self.output = output
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
