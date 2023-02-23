import torch
def kernel_ridge(X,y,jit=1e-3):
    K_xx = 2 * (X @ X.T) + 0.01
    K_xx = K_xx + (jit * torch.eye(1 * X.shape[0], device = X.device) * torch.trace(K_xx)/X.shape[0])
    solved = torch.linalg.solve(K_xx.double(), y.double())
    return solved

def svm_forward(X,Xs,solved):
    K_xx = 2 * (X @ Xs.T) + 0.01
    return (K_xx.double() @ solved).float()

