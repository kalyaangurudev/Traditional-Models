import math
import numpy as np # type: ignore

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
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid function.
           Input and should be a float 
        """
        if not isinstance(z,(int,float,np.ndarray)):
            raise TypeError("z value must be int or a float or a numpy array")    
        #Clipping z to avoid overflowing
        # z = max(min(z,500.0),-500.0)
        z = np.clip(z,-500.0,500.0)
        return 1.0 / (1.0 + np.exp(-z))   
    
    def _compute_gradient(self, X, y, y_pred):
        """Compute gradients for weight update."""
        error = y_pred - y
        w_gradients = np.dot(X.T, error) / X.shape[0]
        b_gradient = np.mean(error)
        return  w_gradients,b_gradient                                            
    
    def _compute_cost(self, y, y_pred):
        """Compute binary cross-entropy loss."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred,epsilon,1-epsilon)
        loss = -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
        return loss

    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Train the model using gradient descent.
            Args:
            X (np.ndarray): Feature matrix of shape (m, n).
            y (np.ndarray): Binary labels of shape (m,).
            Raises:
            ValueError: If X or y are invalid.
        """
        if not isinstance(X,np.ndarray) or not isinstance(y,np.ndarray):
            raise ValueError("X and y must be Numpy Array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        if not np.all((y==0) | (y==1)):
            raise ValueError("y must contain only 1 or 0")
        
        num_features = X.shape[1]
        w = np.zeros(num_features)
        b = 0.0
        prev_loss = float('inf')

        for i in range(0,self.max_iterations):
            z = np.dot(X,w) + b
            predictions = self.sigmoid(z)
            loss = self._compute_cost(y,predictions)
            w1,b1 = self._compute_gradient(X,y,predictions)
            w = w - self.learning_rate*w1
            b = b - self.learning_rate*b1
            if i>0 and  np.abs(loss - prev_loss) < self.tol:
                break
            prev_loss = loss
        self.weights = w
        self.bias = b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of class 1."""
        if not isinstance(X,np.ndarray):
            raise ValueError("X must be Numpy Array")
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before predicting")
        if X.shape[1]!=self.weights.shape[0]:
            raise ValueError("X must have the same number of features as training data")
        
        z = np.dot(X,self.weights) + self.bias
        y_predicted = self.sigmoid(z)
        return y_predicted

    
    def predict(self, X: np.ndarray, threshold: float =0.5) -> np.ndarray:
        """Predict class labels."""
        if not 0< threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        y_predicted = self.predict_proba(X)
        # y_class = [1 if i>= threshold else 0 for i in y_predicted]
        # faster method is boolean indexing than list comprehension
        y_class = (y_predicted >= threshold).astype(int)
        return y_class