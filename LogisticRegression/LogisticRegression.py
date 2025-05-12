import math
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float =0.01, max_iterations: int=1000, tol: float=1e-4):
        """Initialize the logistic regression model.
        Args:
            learning_rate (float): Learning rate for gradient descent.
            max_iterations (int): Maximum number of iterations for training.
            tol (float): Tolerance for convergence.
        """
        if not isinstance(learning_rate,(int,float)) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number")
        if not isinstance(max_iterations,(int)) or max_iterations <= 0:
            raise ValueError("Max Iterations must be a positive number")
        if not isinstance(tol,(int,float)) or tol <= 0:
            raise ValueError("tol must be a positive number")
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z: float) -> float:
        """Compute sigmoid function.
           Input and should be a float 
        """
        if not isinstance(z,(int,float)):
            raise TypeError("z value must be int or a float")    
        #Clipping z to avoid overflowing
        z = max(min(z,500.0),-500.0)
        return 1.0 / (1.0 + math.exp(-z))   
    
    def fit(self, X: list, y: list):
        """
        Train the model using gradient descent.
        INITIALIZE weights to zeros (size = number of features)
        INITIALIZE bias to zero
        FOR iteration = 1 to max_iterations:
        COMPUTE z = X * weights + bias
        COMPUTE predictions = sigmoid(z)
        COMPUTE loss = binary cross-entropy(y, predictions)
        COMPUTE gradients for weights and bias
        UPDATE weights = weights - learning_rate * weight_gradients
        UPDATE bias = bias - learning_rate * bias_gradient
        IF absolute(loss - previous_loss) < tol:
            BREAK
        STORE weights and bias
        """
        if not isinstance(X,np.ndarray()) or not isinstance(y,np.ndarray):
            raise ValueError("X and y must be Numpy Array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        if not np.all((y==0) | (y==1)):
            raise ValueError("y must contain only 1 or 0")
        
        num_features = len(X[0]) if X else 0
        w = [0.0] * num_features
        b = 0.0
        prev_loss = 0
        for i in range(0,self.max_iterations):
            z = 
            predictions = sigmoid(z)
            loss = _compute_cost(y,predictions)
            w1,b1 = _compute_gradients(X,y,predictions)
            w = w - self.learning_rate*w1
            b = b - self.learning_rate*b1
            self.weights = w
            self.bias = b
            if math.modf(loss-prev_loss) < tol:
                break


        
    
    def predict_proba(self, X):
        """Predict probability of class 1."""
        pass
    
    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        pass
    
    def _compute_gradient(self, X, y, y_pred):
        """Compute gradients for weight update."""
        pass
    
    def _compute_cost(self, y, y_pred):
        """Compute binary cross-entropy loss."""
        pass