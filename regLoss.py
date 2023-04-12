import torch

def cal_reg(model, criterion_l1, coefficient):
    reg_loss = 0
    np = 0
    for param in model.parameters():
        reg_loss += criterion_l1(param, torch.zeros_like(param))
        np += param.nelement()
    reg_loss = coefficient * (reg_loss / np)
    return reg_loss