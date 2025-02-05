"""
reproduce based on https://github.com/karpathy/micrograd
"""

import random
import math

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # forward pass
        out = Value(self.data + other.data, (self, other), _op='+')
        # backward pass
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        out = self + (-other)
        out._op = '-'
        return out

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # forward pass
        out = Value(self.data * other.data, (self, other), _op='*')
        # backward pass
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self, other):
        return self * -1

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = self * other.__pow__(-1)
        out._op = '/'
        return out

    def __rtruediv__(self, other):
        return self / other

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only supporting int/float powers for now"
        # forward pass
        out = Value(self.data ** p, (self,), _op='**')
        # backward pass
        def _backward():
            self.grad += p * (self.data**(p-1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        # forward pass
        out = Value(math.exp(self.data), (self,), _op='exp')
        # backward pass
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        # forward pass
        out = Value(0 if self.data < 0 else self.data, (self,), 'relu')
        # backward pass
        def _backward():
            self.grad += 0.0 if out.data < 0 else out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # forward pass
        out = Value((math.exp(2*self.data)-1)/(math.exp(2*self.data)+1), (self,), 'tanh')
        # backward pass
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def _get_topo(v):
            if v not in visited:
                visited.add(v)
                for pre in v._prev:
                    _get_topo(pre)
                topo.append(v)
        _get_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

class Module():

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weight = [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
        self.bias = Value(0.0)

    def __call__(self, x):
        return sum(((self.weight[i]*x[i]) for i in range(self.input_dim)), self.bias).tanh()

    def parameters(self):
        return self.weight + [self.bias]

class Linear(Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons = [Neuron(input_dim) for _ in range(output_dim)]
    
    def __call__(self, x):
        out = [self.neurons[i](x) for i in range(self.output_dim)] 
        return out if self.output_dim > 1 else out[0]
    
    def parameters(self):
        params = []
        for i in range(self.output_dim):
            params.extend(self.neurons[i].parameters())
        return params

class MLP(Module):

    def __init__(self, dims):
        self.n_layers = len(dims) - 1
        self.layers = [Linear(dims[i], dims[i+1]) for i in range(self.n_layers)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for i in range(self.n_layers):
            params.extend(self.layers[i].parameters())
        return params

if __name__ == "__main__":
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    lr = 0.05
    epochs = 100
    net = MLP([3, 4, 4, 1])
    for epoch in range(epochs):
        # zero_grad
        net.zero_grad()
        # forward pass
        ys_pred = [net(x) for x in xs]
        # mse loss
        loss = sum((y_pred-y)**2 for y_pred, y in zip(ys_pred, ys))
        # backward pass
        loss.backward()
        # update parameter
        for p in net.parameters():
            p.data -= lr * p.grad
        print(f"epoch: {epoch}, loss: {loss.data:.4f}")
    print("Results:")
    print([net(x).data for x in xs])
    print(ys)
