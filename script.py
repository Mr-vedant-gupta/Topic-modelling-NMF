import argparse
import csv
import torch
import torch.nn as nn

# File paths for input, validation, and test datasets, as well as model parameters and results
INP_DATA_PATH = 'word_word_correlation'
PROCESSED_DATA_PATH = 'processed_input.pt'
PARAMETERS = 'parameters.pt'


def process_data():
    """Read input matrix from file and save as tensor"""
    with open(INP_DATA_PATH, "r") as file:
        # Read the first line to get dimensions of the matrix
        m, r = map(int, file.readline().split())
        print(m, r)
        M = []
        for _ in range(m):
            row = list(map(float, file.readline().split()))
            M.append(row)
        m_tensor = torch.tensor(M)
        torch.save(torch.tensor([m, r]), PARAMETERS)
        torch.save(m_tensor, PROCESSED_DATA_PATH)


def load_data():
    """Load processed data"""
    m, r = torch.load(PARAMETERS)
    m, r = m.item(), r.item()
    M = torch.load(PROCESSED_DATA_PATH)
    return m, r, M


def initialize_matrices(M, r):
    U, S, Vh = torch.linalg.svd(M)
    U_reduced = U[:, :r]
    S_reduced = torch.diag(torch.sqrt(S[:r]))
    Vh_reduced = Vh[:r, :]
    return (U_reduced @ S_reduced).clamp(min=0), (S_reduced @ Vh_reduced).clamp(min=0)


def optimize_loss_asymmetrical(M, A, W, loss_func, num_iter, index):
    if index == 0:
        A.requires_grad = True
        W.requires_grad = False
        optimizer = torch.optim.Adam([A], lr=0.01)
    else:
        A.requires_grad = False
        W.requires_grad = True
        optimizer = torch.optim.Adam([W], lr=0.01)
    for i in range(num_iter):
        optimizer.zero_grad()
        loss = loss_func(M, A @ W)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            A.clamp(min=0)
            W.clamp(min=0)
        return A, W


def train():
    # TODO: make sure results are indeed non-negative
    """
    -load data
    -initialize matrices
    in a loop:
        optimize wrt M
        optimize wrt W
    :return:
    """
    m, r, M = load_data()
    A, W = initialize_matrices(M, r)
    loss_func = torch.nn.MSELoss(reduction='sum')
    print(loss_func(M, A @ W))
    for i in range(1000):
        A, W = optimize_loss_asymmetrical(M, A, W, loss_func, 500, 0)
        A, W = optimize_loss_asymmetrical(M, A, W, loss_func, 500, 1)
        print(i, loss_func(M, A@W))

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
