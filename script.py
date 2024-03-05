import argparse
import csv
import torch
import torch.nn as nn

# File paths for data
INP_DATA_PATH = 'word_word_correlation'
PROCESSED_DATA_PATH = 'processed_input.pt'
PARAMETERS = 'parameters.pt'
SOLUTION = 'nmf_ans'

# Hyperparameters
LEARNING_RATE = 0.0001
NUM_ITER = 100
TRAINING_ITERATIONS = 31999


def process_data():
    """Read input matrix from file and save as tensor"""
    with open(INP_DATA_PATH, "r") as file:
        # Read the first line to get dimensions of the matrix
        m, r = map(int, file.readline().split())
        M = []
        for _ in range(m):
            row = list(map(float, file.readline().split()))
            M.append(row)
        M_tensor = torch.tensor(M)
        assert list(M_tensor.shape) == [m, m]
        torch.save(torch.tensor([m, r]), PARAMETERS)
        torch.save(M_tensor, PROCESSED_DATA_PATH)


def load_data():
    """Load processed data"""
    m, r = torch.load(PARAMETERS)
    m, r = m.item(), r.item()
    M = torch.load(PROCESSED_DATA_PATH)
    return m, r, M


def initialize_matrices(M, r):
    """Initialize matrices for NMF using SVD"""
    # Get mean value of M. Will use this wherever there is a negative value
    m_avg = torch.mean(M)
    U, S, Vh = torch.linalg.svd(M)
    U_reduced = U[:, :r]
    S_reduced = torch.diag(torch.sqrt(S[:r]))
    Vh_reduced = Vh[:r, :]
    X = (U_reduced @ S_reduced)
    Y = (S_reduced @ Vh_reduced)
    return torch.where(X < 0, m_avg, X), torch.where(Y < 0, m_avg, Y)

def optimize_loss_asymmetrical(M, A, W, loss_func, index):
    """Optimize loss of A@W wrt M using loss_func wrt only one matrix"""
    if index == 0: # If index is 0 optimize wrt A else optimize wrt W
        A.requires_grad = True
        W.requires_grad = False
        optimizer = torch.optim.Adam([A], lr=LEARNING_RATE, weight_decay=1e-5)
    else:
        A.requires_grad = False
        W.requires_grad = True
        optimizer = torch.optim.Adam([W], lr=LEARNING_RATE, weight_decay=1e-5)
    for i in range(NUM_ITER):
        optimizer.zero_grad()
        loss = loss_func(M, A @ W)
        loss.backward()
        optimizer.step()
        with torch.no_grad(): # set all negative values to 0
            A = A.clamp(min=0)
            W = W.clamp(min=0)
        return A, W

def verify_solutions(A, W, m, r):
    """Make sure the solutions are of the specified format"""
    assert torch.min(A) >= 0
    assert torch.min(W) >= 0
    assert list(A.shape) == [m, r]
    assert list(W.shape) == [r, m]


def write_predictions(A, W):
    """write predictions to file"""
    with open(SOLUTION, "w") as file:
        for row in A:
            row_str = " ".join(map(str, row.tolist()))
            file.write(row_str + "\n")
        for row in W:
            row_str = " ".join(map(str, row.tolist()))
            file.write(row_str + "\n")




def train():
    """Main training function following the following steps:
    1. Load training data
    2. Initialize matrices
    3. Train model using alternating gradient descent
    """
    m, r, M = load_data()
    A, W = initialize_matrices(M, r)
    loss_func = torch.nn.MSELoss(reduction='sum')
    for i in range(TRAINING_ITERATIONS):
        A, W = optimize_loss_asymmetrical(M, A, W, loss_func, 0)
        A, W = optimize_loss_asymmetrical(M, A, W, loss_func, 1)
        print(f"Iteration {i}: {loss_func(M, A@W)}")
    verify_solutions(A, W, m, r)
    write_predictions(A, W)
    return A, W


def main(mode):
    """Main function to process data or train the model based on the input mode."""
    if mode == "process_data":
        process_data()
    elif mode == "train":
        train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        help='Must be process_data or train')
    args = parser.parse_args()
    main(args.mode)
