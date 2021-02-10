import torch


class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize


class HeatmapAcc(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapAcc, self).__init__()

    def forward(self, pred_heatmap, heatmap):

        gt_ind = heatmap.view(heatmap.shape[0], heatmap.shape[1], -1).argmax(dim=2)
        pred_ind = pred_heatmap.view(heatmap.shape[0], heatmap.shape[1], -1).argmax(dim=2)
        eq = torch.eq(gt_ind, pred_ind)
        return eq.float().mean(dim=0) ## l of dim bsize