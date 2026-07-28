

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


SEMILLA = 42
np.random.seed(SEMILLA)

class LogsisticRegressionSGD:
    """Regresión Logística con SGD estándar"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        
    def softmax(self, z):
        """Función softmax numéricamente estable"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred_proba):
        """Cálculo de pérdida de entropía cruzada"""
        n_samples = y_true.shape[0]
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        y_onehot = np.eye(self.n_classes)[y_true]
        loss = -np.sum(y_onehot * np.log(y_pred_proba)) / n_samples
        return loss
    
    def compute_gradient(self, X, y_true, y_pred_proba):
        """Gradiente de la entropía cruzada con softmax"""
        n_samples = X.shape[0]
        y_onehot = np.eye(self.n_classes)[y_true]
        gradient = (1 / n_samples) * X.T @ (y_pred_proba - y_onehot)
        return gradient
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Entrenamiento con SGD"""
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        # Inicialización de pesos
        np.random.seed(42)
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        
        print("\nEntrenando SGD...")
        for iteration in tqdm(range(self.n_iterations)):
            # Forward pass
            logits = X @ self.weights
            y_pred_proba = self.softmax(logits)
            
            # Calcular pérdida en entrenamiento
            loss = self.cross_entropy_loss(y, y_pred_proba)
            self.loss_history.append(loss)
            
            # Calcular accuracy en entrenamiento
            y_pred = np.argmax(y_pred_proba, axis=1)
            acc = accuracy_score(y, y_pred)
            self.accuracy_history.append(acc)
            
            # Validación (si se proporciona)
            if X_val is not None and y_val is not None:
                val_logits = X_val @ self.weights
                val_pred_proba = self.softmax(val_logits)
                val_loss = self.cross_entropy_loss(y_val, val_pred_proba)
                val_pred = np.argmax(val_pred_proba, axis=1)
                val_acc = accuracy_score(y_val, val_pred)
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_acc)
            
            # Calcular gradiente
            gradient = self.compute_gradient(X, y, y_pred_proba)
            
            # Actualización SGD: w = w - lr * grad
            self.weights -= self.lr * gradient
        
        return self
    
    def predict(self, X):
        """Predecir clases"""
        logits = X @ self.weights
        y_pred_proba = self.softmax(logits)
        return np.argmax(y_pred_proba, axis=1)
    
    def predict_proba(self, X):
        """Predecir probabilidades"""
        logits = X @ self.weights
        return self.softmax(logits)


class LogisticRegressionAdaGrad:
    """Regresión Logística con AdaGrad"""
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, epsilon=1e-8):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.weights = None
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        
    def softmax(self, z):
        """Función softmax numéricamente estable"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred_proba):
        """Cálculo de pérdida de entropía cruzada"""
        n_samples = y_true.shape[0]
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        y_onehot = np.eye(self.n_classes)[y_true]
        loss = -np.sum(y_onehot * np.log(y_pred_proba)) / n_samples
        return loss
    
    def compute_gradient(self, X, y_true, y_pred_proba):
        """Gradiente de la entropía cruzada con softmax"""
        n_samples = X.shape[0]
        y_onehot = np.eye(self.n_classes)[y_true]
        gradient = (1 / n_samples) * X.T @ (y_pred_proba - y_onehot)
        return gradient
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Entrenamiento con AdaGrad"""
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        # Inicialización de pesos
        np.random.seed(42)
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        
        # AdaGrad: acumulador de gradientes al cuadrado
        G = np.zeros_like(self.weights)
        
        print("\nEntrenando AdaGrad...")
        for iteration in tqdm(range(self.n_iterations)):
            # Forward pass
            logits = X @ self.weights
            y_pred_proba = self.softmax(logits)
            
            # Calcular pérdida en entrenamiento
            loss = self.cross_entropy_loss(y, y_pred_proba)
            self.loss_history.append(loss)
            
            # Calcular accuracy en entrenamiento
            y_pred = np.argmax(y_pred_proba, axis=1)
            acc = accuracy_score(y, y_pred)
            self.accuracy_history.append(acc)
            
            # Validación (si se proporciona)
            if X_val is not None and y_val is not None:
                val_logits = X_val @ self.weights
                val_pred_proba = self.softmax(val_logits)
                val_loss = self.cross_entropy_loss(y_val, val_pred_proba)
                val_pred = np.argmax(val_pred_proba, axis=1)
                val_acc = accuracy_score(y_val, val_pred)
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_acc)
            
            # Calcular gradiente
            gradient = self.compute_gradient(X, y, y_pred_proba)
            
            # AdaGrad: acumular gradientes al cuadrado
            G += gradient ** 2
            
            # Actualización AdaGrad: w = w - (lr / sqrt(G + epsilon)) * grad
            self.weights -= (self.lr / (np.sqrt(G) + self.epsilon)) * gradient
        
        return self
    
    def predict(self, X):
        """Predecir clases"""
        logits = X @ self.weights
        y_pred_proba = self.softmax(logits)
        return np.argmax(y_pred_proba, axis=1)
    
    def predict_proba(self, X):
        """Predecir probabilidades"""
        logits = X @ self.weights
        return self.softmax(logits)


class MLP:
    """Red multicapa 8 -> 64 -> 32 -> 1 (ReLU, salida lineal) con pérdida MSE."""

    def __init__(self, n_entradas, n_oculta1=64, n_oculta2=32, semilla=SEMILLA):
        rng = np.random.default_rng(semilla)
        # Inicialización de He: N(0, sqrt(2 / n_entradas_de_la_capa))
        self.params = {
            'W1': rng.normal(0, np.sqrt(2 / n_entradas), (n_entradas, n_oculta1)),
            'b1': np.zeros(n_oculta1),
            'W2': rng.normal(0, np.sqrt(2 / n_oculta1), (n_oculta1, n_oculta2)),
            'b2': np.zeros(n_oculta2),
            'W3': rng.normal(0, np.sqrt(2 / n_oculta2), (n_oculta2, 1)),
            'b3': np.zeros(1),
        }

    def forward(self, X):
        p = self.params
        self.X = X
        self.Z1 = X @ p['W1'] + p['b1']
        self.A1 = np.maximum(0, self.Z1)          # ReLU
        self.Z2 = self.A1 @ p['W2'] + p['b2']
        self.A2 = np.maximum(0, self.Z2)          # ReLU
        self.salida = (self.A2 @ p['W3'] + p['b3']).ravel()
        return self.salida

    def backward(self, y):
        """Gradiente de la pérdida MSE respecto de cada parámetro (retropropagación)."""
        p = self.params
        m = len(y)
        grads = {}
        d_salida = (2.0 / m) * (self.salida - y)[:, None]     # dL/d(salida)
        grads['W3'] = self.A2.T @ d_salida
        grads['b3'] = d_salida.sum(axis=0)
        dA2 = d_salida @ p['W3'].T
        dZ2 = dA2 * (self.Z2 > 0)                             # derivada de ReLU
        grads['W2'] = self.A1.T @ dZ2
        grads['b2'] = dZ2.sum(axis=0)
        dA1 = dZ2 @ p['W2'].T
        dZ1 = dA1 * (self.Z1 > 0)
        grads['W1'] = self.X.T @ dZ1
        grads['b1'] = dZ1.sum(axis=0)
        return grads

    def perdida(self, X, y):
        return np.mean((self.forward(X) - y) ** 2)


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, {}
    def update(self, params, grads):
        for k in params:
            self.v[k] = self.beta * self.v.get(k, 0.0) + grads[k]
            params[k] -= self.lr * self.v[k]


class AdaGrad:
    def __init__(self, lr=0.05, epsilon=1e-8):
        self.lr, self.epsilon, self.G = lr, epsilon, {}
    def update(self, params, grads):
        for k in params:
            self.G[k] = self.G.get(k, 0.0) + grads[k] ** 2
            params[k] -= self.lr * grads[k] / (np.sqrt(self.G[k]) + self.epsilon)


class RMSProp:
    def __init__(self, lr=0.005, rho=0.9, epsilon=1e-8):
        self.lr, self.rho, self.epsilon, self.G = lr, rho, epsilon, {}
    def update(self, params, grads):
        for k in params:
            self.G[k] = self.rho * self.G.get(k, 0.0) + (1 - self.rho) * grads[k] ** 2
            params[k] -= self.lr * grads[k] / (np.sqrt(self.G[k]) + self.epsilon)


class Adam:
    def __init__(self, lr=0.005, rho1=0.9, rho2=0.999, epsilon=1e-8):
        self.lr, self.rho1, self.rho2, self.epsilon = lr, rho1, rho2, epsilon
        self.m, self.v, self.t = {}, {}, 0
    def update(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.rho1 * self.m.get(k, 0.0) + (1 - self.rho1) * grads[k]
            self.v[k] = self.rho2 * self.v.get(k, 0.0) + (1 - self.rho2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.rho1 ** self.t)   # corrección de sesgo
            v_hat = self.v[k] / (1 - self.rho2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdamW(Adam):
    """Adam con weight decay desacoplado (Loshchilov & Hutter, 2019)."""
    def __init__(self, lr=0.005, rho1=0.9, rho2=0.999, epsilon=1e-8, wd=1e-4):
        super().__init__(lr, rho1, rho2, epsilon)
        self.wd = wd
    def update(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.rho1 * self.m.get(k, 0.0) + (1 - self.rho1) * grads[k]
            self.v[k] = self.rho2 * self.v.get(k, 0.0) + (1 - self.rho2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.rho1 ** self.t)
            v_hat = self.v[k] / (1 - self.rho2 ** self.t)
            params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.wd * params[k])