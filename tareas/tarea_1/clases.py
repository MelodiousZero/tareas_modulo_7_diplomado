

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
