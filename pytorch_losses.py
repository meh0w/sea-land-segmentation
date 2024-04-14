import torch

class CEDice(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.ce = torch.nn.BCELoss()
      self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, real):
        epsilon = 1e-9
        pred = self.softmax(pred)
        ce = self.ce(pred, real)
        dice=torch.mean(1 - ((2*torch.sum(real*pred, dim=(1,2,3))+epsilon)/(torch.sum(real+pred, dim=(1,2,3))+epsilon)))
        

        return ce + 0.7*dice