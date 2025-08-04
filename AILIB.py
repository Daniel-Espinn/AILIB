import math
import random
import numpy as np
import time
import sys
import queue
import threading

class Value:    
    _CACHE = {}  # Memoria caché para operaciones costosas
    
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        cache_key = ('pow', self.data, other)
        if cache_key in Value._CACHE:
            out = Value._CACHE[cache_key]
            out._prev = (self,)
            return out
            
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            if self.requires_grad:
                self.grad += other * (self.data**(other - 1)) * out.grad
        out._backward = _backward
        
        Value._CACHE[cache_key] = out
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            if self.requires_grad:
                self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out

    def tanh(self):
        cache_key = ('tanh', self.data)
        if cache_key in Value._CACHE:
            out = Value._CACHE[cache_key]
            out._prev = (self,)
            return out
            
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        Value._CACHE[cache_key] = out
        return out

    def log(self):
        x = self.data
        if x <= 0:
            raise ValueError("Math domain error (log of <= 0)")
            
        cache_key = ('log', x)
        if cache_key in Value._CACHE:
            out = Value._CACHE[cache_key]
            out._prev = (self,)
            return out
            
        t = math.log(x)
        out = Value(t, (self,), 'log')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        
        Value._CACHE[cache_key] = out
        return out

    def exp(self):
        cache_key = ('exp', self.data)
        if cache_key in Value._CACHE:
            out = Value._CACHE[cache_key]
            out._prev = (self,)
            return out
            
        x = self.data
        t = math.exp(x)
        out = Value(t, (self,), 'exp')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        
        Value._CACHE[cache_key] = out
        return out

    def sigmoid(self):
        if self.data >= 0:
            out = Value(1) / (Value(1) + (-self).exp())
        else:
            e = self.exp()
            out = e / (Value(1) + e)
        out._op = 'sigmoid'
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
    def train(self):
        self.training = True
        for p in self.parameters():
            p.requires_grad = True
            
    def eval(self):
        self.training = False
        for p in self.parameters():
            p.requires_grad = False

class Neuron(Module):
    def __init__(self, nin, nonlin='relu', dropout=0.0, use_batchnorm=False):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        self.dropout = dropout
        self.training = True
        self.use_batchnorm = use_batchnorm
        
        if use_batchnorm:
            self.bn = BatchNorm1d(1)
    
    def __call__(self, x):
        # Vectorización de operaciones
        act = self.b + sum((wi * xi for wi, xi in zip(self.w, x)), Value(0))
        
        # Batch normalization
        if self.use_batchnorm and self.training:
            act = self.bn([act])[0]
        
        if self.training and self.dropout > 0:
            if random.random() < self.dropout:
                act = Value(0)
            else:
                act = act / (1 - self.dropout)
        
        if self.nonlin == 'relu':
            return act.relu()
        elif self.nonlin == 'tanh':
            return act.tanh()
        elif self.nonlin == 'sigmoid':
            return act.sigmoid()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]
    
    def set_dropout(self, p):
        self.dropout = p

    def __repr__(self):
        return f"{self.nonlin.upper() if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        self.use_batchnorm = kwargs.get('use_batchnorm', False)
        self.nonlin = kwargs.get('nonlin', None)
        
        # Inicializar BatchNorm solo si se requiere y hay activación
        if self.use_batchnorm and self.nonlin:
            self.bn = BatchNorm1d(nout)
        else:
            self.bn = None  # Asegurarse de que el atributo exista siempre
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        
        if self.bn is not None and self.neurons[0].training:  # Verificar que bn existe
            out = self.bn(out)
            
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = [p for n in self.neurons for p in n.parameters()]
        if self.bn is not None:
            params += self.bn.parameters()
        return params
    
    def set_dropout(self, p):
        for neuron in self.neurons:
            neuron.set_dropout(p)

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class BatchNorm1d(Module):
    """Capa de Batch Normalization para redes densas"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = Value(1.0)
        self.beta = Value(0.0)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def __call__(self, x):
        if self.training:
            batch_data = np.array([v.data for v in x])
            mean = batch_data.mean()
            var = batch_data.var()
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            x_normalized = [(v - mean) / np.sqrt(var + self.eps) for v in x]
        else:
            x_normalized = [(v - self.running_mean) / np.sqrt(self.running_var + self.eps) for v in x]
        
        return [self.gamma * v + self.beta for v in x_normalized]

    def parameters(self):
        return [self.gamma, self.beta]

class MLP(Module):
    def __init__(self, nin, nouts, activation='relu', dropout=0.0, 
                 weight_decay=0.0, use_batchnorm=False):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            nonlin = activation if i < len(nouts)-1 else None
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=nonlin, 
                                   dropout=dropout, use_batchnorm=use_batchnorm))
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.set_dropout(dropout)
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def set_dropout(self, p):
        for layer in self.layers:
            layer.set_dropout(p)

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# Loss Functions
class Loss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        
    def __call__(self, y_pred, y_true):
        raise NotImplementedError("Loss function must implement __call__")
        
    def reduce(self, losses):
        if self.reduction == 'mean':
            return sum(losses) / len(losses)
        elif self.reduction == 'sum':
            return sum(losses)
        return losses

class MSE(Loss):
    """Mean Squared Error Loss"""
    def __call__(self, y_pred, y_true):
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
            y_true = [y_true]
            
        losses = [(pred - true)**2 for pred, true in zip(y_pred, y_true)]
        return self.reduce(losses)

class BCE(Loss):
    """Binary Cross Entropy Loss (stable)"""
    def __init__(self, reduction='mean', epsilon=1e-8):
        super().__init__(reduction)
        self.epsilon = epsilon
        
    def __call__(self, y_pred, y_true):
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
            y_true = [y_true]
            
        losses = []
        for pred, true in zip(y_pred, y_true):
            p = pred.sigmoid()
            loss = - (true * (p + self.epsilon).log() + (Value(1) - true) * (Value(1) - p + self.epsilon).log())
            losses.append(loss)
            
        return self.reduce(losses)

class CCE(Loss):
    """Categorical Cross Entropy Loss"""
    def __init__(self, reduction='mean', epsilon=1e-8):
        super().__init__(reduction)
        self.epsilon = epsilon
        
    def __call__(self, y_pred, y_true):
        if isinstance(y_true, int) or isinstance(y_true, float):
            y_true = int(y_true)
            if len(y_pred) > 1:
                true_vector = [Value(1) if i == y_true else Value(0) for i in range(len(y_pred))]
            else:
                true_vector = [Value(y_true)]
        else:
            true_vector = y_true
            
        max_val = max(pred.data for pred in y_pred)
        exps = [(pred - max_val).exp() for pred in y_pred]
        sum_exps = sum(exps)
        probs = [exp / sum_exps for exp in exps]
        
        losses = []
        for prob, true_val in zip(probs, true_vector):
            losses.append(-true_val * (prob + self.epsilon).log())
            
        return self.reduce(losses)

# Optimizers
class Optimizer:
    def __init__(self, params, lr=0.01, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.t = 0
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0
            
    def step(self):
        self.t += 1
        self._step()
        
    def _step(self):
        raise NotImplementedError("Optimizer must implement _step")
        
    def set_lr(self, lr):
        self.lr = lr

class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum and weight decay"""
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = [0] * len(self.params)
        
    def _step(self):
        for i, p in enumerate(self.params):
            grad = p.grad + self.weight_decay * p.data
            self.velocity[i] = self.momentum * self.velocity[i] + grad
            if self.nesterov:
                grad = grad + self.momentum * self.velocity[i]
            else:
                grad = self.velocity[i]
            p.data -= self.lr * grad

class Adam(Optimizer):
    """Adam optimizer with weight decay"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [0] * len(self.params)
        self.v = [0] * len(self.params)
        
    def _step(self):
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.alpha = alpha
        self.eps = eps
        self.sq_grad = [0] * len(self.params)
        
    def _step(self):
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self.sq_grad[i] = self.alpha * self.sq_grad[i] + (1 - self.alpha) * (g**2)
            p.data -= self.lr * g / (self.sq_grad[i]**0.5 + self.eps)

# Schedulers
class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        
    def step(self):
        self.update_lr()
        
    def update_lr(self):
        raise NotImplementedError("Scheduler must implement update_lr")

class StepLR(LRScheduler):
    """Step learning rate scheduler"""
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_step = 0
        
    def update_lr(self):
        self.last_step += 1
        if self.last_step % self.step_size == 0:
            self.optimizer.lr *= self.gamma

class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving"""
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4):
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best = None
        self.num_bad_epochs = 0
        
    def step(self, metric):
        if self.best is None:
            self.best = metric
            return
            
        if self.mode == 'min':
            improve = (metric - self.best) < -self.threshold
        else:
            improve = (metric - self.best) > self.threshold
            
        if improve:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            self.optimizer.lr *= self.factor
            self.num_bad_epochs = 0
            print(f"Reducing learning rate to {self.optimizer.lr:.6f}")

# Regularization
def l2_regularization(model, alpha=0.01):
    """L2 Regularization (Weight Decay)"""
    reg_loss = Value(0)
    for p in model.parameters():
        if p.requires_grad:
            reg_loss += alpha * (p ** 2)
    return reg_loss

def l1_regularization(model, alpha=0.01):
    """L1 Regularization"""
    reg_loss = Value(0)
    for p in model.parameters():
        if p.requires_grad:
            reg_loss += alpha * abs(p)
    return reg_loss

# Model Utilities
def accuracy(y_pred, y_true):
    """Calculate accuracy for classification"""
    correct = 0
    for pred, true in zip(y_pred, y_true):
        pred_class = 1 if pred.data > 0.5 else 0
        if pred_class == true.data:
            correct += 1
    return correct / len(y_true)

def save_model(model, path):
    """Save model parameters to file"""
    params = [p.data for p in model.parameters()]
    np.save(path, np.array(params))

def load_model(model, path):
    """Load model parameters from file"""
    params = np.load(path, allow_pickle=True)
    for i, p in enumerate(model.parameters()):
        p.data = params[i]

# Training Tools
class ProgressMonitor:
    """Sistema avanzado de monitoreo de progreso para entrenamiento"""
    def __init__(self, total_epochs, total_batches):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.start_time = time.time()
        self.epoch_times = []
        self.batch_times = []
        
    def start_epoch(self, epoch):
        self.epoch_start = time.time()
        self.current_epoch = epoch
        self.batch_times = []
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{self.total_epochs}")
        print(f"{'='*50}")
        
    def update_batch(self, batch_idx, batch_loss, batch_acc):
        """Actualiza el progreso dentro de una época"""
        batch_time = time.time()
        if len(self.batch_times) > 0:
            avg_batch_time = np.mean(self.batch_times)
        else:
            avg_batch_time = 0
            
        completed = (batch_idx + 1) / self.total_batches
        remaining_batches = self.total_batches - (batch_idx + 1)
        remaining_time = avg_batch_time * remaining_batches
        
        mins, secs = divmod(remaining_time, 60)
        
        bar_length = 30
        filled_length = int(bar_length * completed)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        sys.stdout.write(
            f"\rBatch {batch_idx+1}/{self.total_batches} | "
            f"Loss: {batch_loss:.4f} | Acc: {batch_acc*100:.1f}% | "
            f"ETA: {mins:.0f}m {secs:.0f}s | "
            f"[{bar}] {completed*100:.1f}%"
        )
        sys.stdout.flush()
        
        self.batch_times.append(time.time() - batch_time)
        
    def end_epoch(self, epoch_loss, epoch_acc, val_loss=None, val_acc=None):
        """Finaliza una época y muestra resumen"""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - (self.current_epoch + 1)
        remaining_time = avg_epoch_time * remaining_epochs
        
        mins, secs = divmod(remaining_time, 60)
        hours, mins = divmod(mins, 60)
        
        print("\n" + "-"*50)
        print(f"EPOCH SUMMARY")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%")
        
        if val_loss is not None:
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        print(f"Time: {epoch_time:.1f}s | Total: {time.time()-self.start_time:.1f}s")
        print(f"ETA: {hours:.0f}h {mins:.0f}m {secs:.0f}s")
        print("-"*50)
        
    def training_summary(self):
        """Resumen final del entrenamiento"""
        total_time = time.time() - self.start_time
        mins, secs = divmod(total_time, 60)
        hours, mins = divmod(mins, 60)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print(f"Total Time: {hours:.0f}h {mins:.0f}m {secs:.0f}s")
        print(f"Avg Epoch Time: {np.mean(self.epoch_times):.2f}s")
        print("="*50)

class SimpleDataLoader:
    """DataLoader básico sin multiprocessing para cuando no se necesita carga avanzada"""
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.dataset)
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
            
        batch_samples = self.dataset[self.index:self.index+self.batch_size]
        self.index += len(batch_samples)
        
        X_batch = np.array([s[0] for s in batch_samples])
        y_batch = np.array([s[1] for s in batch_samples])
        return X_batch, y_batch

class DataLoader:
    """DataLoader con multiprocessing y prefetching"""
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.workers = []
        self.stop_event = threading.Event()
        self.idx = 0
        
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                batch = self._get_next_batch()
                if batch is not None:
                    self.queue.put(batch)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error en worker: {e}")
    
    def _get_next_batch(self):
        if self.idx >= len(self.dataset):
            return None
            
        batch_samples = self.dataset[self.idx:self.idx+self.batch_size]
        self.idx += len(batch_samples)
        
        X_batch = np.array([s[0] for s in batch_samples])
        y_batch = np.array([s[1] for s in batch_samples])
        return X_batch, y_batch
    
    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            np.random.shuffle(self.dataset)
        return self
    
    def __next__(self):
        if self.idx >= len(self.dataset) and self.queue.empty():
            raise StopIteration
        return self.queue.get(timeout=30.0)
    
    def shutdown(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=1.0)

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.gradient_clip = 1.0
        self.best_val_acc = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def fit(self, dataset_train, dataset_val, epochs=100, batch_size=32,
            verbose=True, num_workers=4, use_advanced_loader=True):
        
        if use_advanced_loader:
            train_loader = DataLoader(dataset_train, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(dataset_val, batch_size=batch_size,
                                  shuffle=False, num_workers=1)
        else:
            train_loader = SimpleDataLoader(dataset_train, batch_size, shuffle=True)
            val_loader = SimpleDataLoader(dataset_val, batch_size, shuffle=False)
        
        n_batches = int(np.ceil(len(dataset_train) / batch_size))
        progress = ProgressMonitor(total_epochs=epochs, total_batches=n_batches)
        
        for epoch in range(epochs):
            if verbose:
                progress.start_epoch(epoch)
            
            self.model.train()
            total_loss = 0
            total_correct = 0
            batch_counter = 0
            
            for batch_x, batch_y in train_loader:
                batch_loss, batch_correct = self._process_batch(batch_x, batch_y)
                
                total_loss += batch_loss
                total_correct += batch_correct
                batch_acc = batch_correct / len(batch_y)
                
                if verbose:
                    progress.update_batch(batch_counter, batch_loss / len(batch_y), batch_acc)
                batch_counter += 1
            
            avg_loss = total_loss / len(dataset_train)
            accuracy = total_correct / len(dataset_train)
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(accuracy)
            
            val_loss, val_acc = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            if verbose:
                progress.end_epoch(avg_loss, accuracy, val_loss, val_acc)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_params = [p.data for p in self.model.parameters()]
        
        if verbose:
            progress.training_summary()
        
        if hasattr(self, 'best_model_params'):
            for param, best_value in zip(self.model.parameters(), self.best_model_params):
                param.data = best_value
        
        if use_advanced_loader:
            train_loader.shutdown()
            val_loader.shutdown()
        
        return self.history
    
    def _process_batch(self, batch_x, batch_y):
        batch_loss = 0
        batch_correct = 0
        
        # Vectorización del forward pass
        inputs = [[Value(x_i) for x_i in x] for x in batch_x]
        y_preds = [self.model(x) for x in inputs]
        
        # Calcular pérdida
        loss = self.criterion(y_preds, batch_y)
        batch_loss += loss.data * len(batch_y)
        
        # Calcular precisión
        batch_correct = sum(1 for pred, true in zip(y_preds, batch_y) 
                           if (pred.data > 0.5) == (true > 0.5))
        
        # Backpropagation
        self.model.zero_grad()
        loss.backward()
        
        # Recorte de gradientes
        for p in self.model.parameters():
            if abs(p.grad) > self.gradient_clip:
                p.grad = self.gradient_clip * np.sign(p.grad)
        
        # Paso del optimizador
        self.optimizer.step()
        
        return batch_loss, batch_correct
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_x, batch_y in data_loader:
            inputs = [[Value(x_i) for x_i in x] for x in batch_x]
            y_preds = [self.model(x) for x in inputs]
            
            loss = self.criterion(y_preds, batch_y)
            total_loss += loss.data * len(batch_y)
            
            total_correct += sum(1 for pred, true in zip(y_preds, batch_y) 
                            if (pred.data > 0.5) == (true > 0.5))
            total_samples += len(batch_y)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def predict(self, X, batch_size=32):
        self.model.eval()
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size]
            inputs = [[Value(x_i) for x_i in x] for x in batch_x]
            y_preds = [self.model(x) for x in inputs]
            predictions.extend([p.data for p in y_preds])
        
        return predictions