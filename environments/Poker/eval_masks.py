import torch
# class that contains the mask logic for the type of hands in poker
def pair_mask(board, hands, device):
    board_ranks=board%13
    ranks=hands%13,
    matches1 = (board_ranks == ranks[:, 0:1]).sum(dim=1)
    matches2 = (board_ranks == ranks[:, 1:2]).sum(dim=1)
    return (matches1 + matches2 == 1).to(torch.int32, device=device)

def two_pair_mask(board, hands, device):
    board_ranks=board%13
    ranks=hands%13
    matches1 = (board_ranks == ranks[:, 0:1]).sum(dim=1)
    matches2 = (board_ranks == ranks[:, 1:2]).sum(dim=1)
    
    return ((matches1+matches2==2 ) & (matches1 < 2) & (matches2 < 2)).to(torch.int32, device=device)

