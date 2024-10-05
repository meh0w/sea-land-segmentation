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


class SeNetLoss(torch.nn.Module):
  def __init__(self, Lambda=35):
    super().__init__()
    self.softmax = torch.nn.Softmax(dim=1)
    self.ce = torch.nn.BCELoss(reduction='none')
    self.pad = torch.nn.ReplicationPad2d(1)
    self.Lambda = Lambda

  def view_as_windows_torch(self, image, shape, stride=None):
    """View tensor as overlapping rectangular windows, with a given stride.

    Parameters
    ----------
    image : `~torch.Tensor`
        4D image tensor, with the last two dimensions
        being the image dimensions
    shape : tuple of int
        Shape of the window.
    stride : tuple of int
        Stride of the windows. By default it is half of the window size.

    Returns
    -------
    windows : `~torch.Tensor`
        Tensor of overlapping windows

    """
    if stride is None:
        stride = shape[0] // 2, shape[1] // 2

    windows = image.unfold(2, shape[0], stride[0])
    return windows.unfold(3, shape[1], stride[1])

  def forward(self, pred, real, img):
      epsilon = 1e-9
      pred_seg = self.softmax(pred[0])
      pred_edge = self.softmax(pred[1])
      
      ce = self.ce(pred_seg, real)
      padded_p = self.pad(pred_seg)
      padded_im = self.pad(img)
      neighbours = self.view_as_windows_torch(image=padded_p, shape=(3, 3)).reshape(pred_seg.size()+(9,))
      im_neighbours = self.view_as_windows_torch(image=padded_im, shape=(3, 3)).reshape(img.size()+(9,))
      p = pred_seg.unsqueeze(-1).expand(pred_seg.size()+(9,))
      im = img.unsqueeze(-1).expand(img.size()+(9,))

      temp = torch.sum((p - neighbours)**2, dim=1, keepdim=True)
      temp2 = torch.sum((im-im_neighbours)**2, dim=1, keepdim=True)
      sigma = torch.mean(temp2, dim=(1,2,3,4), keepdim=True).expand(temp2.size())
      l_seg = torch.mean(ce + (self.Lambda / 2) * torch.sum(temp * torch.exp(-(temp2+epsilon)/(sigma+epsilon)), dim=-1))

      padded_p = self.pad(pred_edge)
      padded_r = self.pad(real)
      neighbours = self.view_as_windows_torch(image=padded_p, shape=(3, 3)).reshape(pred_edge.size()+(9,))[:,1,:,:,[0,1,2,3,5,6,7,8]]
      real_neighbours = self.view_as_windows_torch(image=padded_r, shape=(3, 3)).reshape(real.size()+(9,))[:,:,:,:,[0,1,2,3,5,6,7,8]]
      p = pred_edge.unsqueeze(-1).expand(pred_edge.size()+(8,))[:,1,:,:,:]
      r = real.unsqueeze(-1).expand(real.size()+(8,))
      SN = torch.abs(r-real_neighbours)
      SN0 = SN[:,0,:,:,:]
      SN1 = SN[:,1,:,:,:]
      temp4 = torch.ceil(torch.sum(SN, dim=-1)).unsqueeze(-1).expand(r.size())/8 # if 0 => not edge
      temp4_0 = temp4[:,0,:,:,:]
      temp4_1 = temp4[:,1,:,:,:]
      temp5 = torch.sum(temp4[:,:,:,:,0], dim=1, keepdim=True)
      N = torch.flatten(real).size()[0]
      # l_edge = -(torch.sum(r * ((SN*temp4)*(torch.log(p)+torch.log(1-neighbours)))/(SN*temp4+(1-temp4))) + torch.sum(temp5 * torch.log(1-torch.sum(torch.abs(p-neighbours)/8,dim=-1))))/N
      
      l_edge_L = torch.sum(((SN0*temp4_0)*(torch.log(p+epsilon)+torch.log(1-neighbours+epsilon)))/(SN0*temp4_0+(1-temp4_0)))
      l_edge_S = torch.sum(((SN1*temp4_1)*(torch.log(1-p+epsilon)+torch.log(neighbours+epsilon)))/(SN1*temp4_1+(1-temp4_1)))
      l_edge = -(l_edge_L + l_edge_S + torch.sum(temp5 * torch.log(1-torch.sum(torch.abs(p-neighbours)/8,dim=-1))))/N

      # loss = torch.mean(ce + (self.Lambda / 2) * torch.sum(temp * torch.exp(-(temp2+epsilon)/(sigma+epsilon)) + l_edge, dim=-1))
      loss = torch.mean(l_seg + l_edge, dim=-1)


      return loss

  
