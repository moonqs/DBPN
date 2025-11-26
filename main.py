
class purificationReconstructionMoudle(nn.Module):
    def __init__(self, channels, groups=32, thresholds=(0.7, 0.3)):
        super().__init__()
        self.pad_channels = (2 - (channels % 2)) % 2
        self.true_channels = channels + self.pad_channels
        effective_groups = min(groups, self.true_channels)
        self.gn = nn.GroupNorm(effective_groups, self.true_channels)
        self.weight_gen = nn.Sequential(
            nn.Conv2d(self.true_channels, self.true_channels//16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.true_channels//16, 1, 1),
            nn.Sigmoid()
        )
        
        self.register_buffer('thresholds', torch.tensor(thresholds))
        self.split_size = self.true_channels // 2 

    def forward(self, x):
        x_pad = F.pad(x, (0,0,0,0,0,self.pad_channels))
        x_norm = self.gn(x_pad)

        reweigts = self.weight_gen(x_norm)
        w1 = torch.where(reweigts > self.thresholds[0], 1.0, reweigts)
        w2 = torch.where((reweigts > self.thresholds[1]) & 
                        (reweigts <= self.thresholds[0]), reweigts, 0.0)
        x1, x2 = x_pad * w1, x_pad * w2
        x1_parts = torch.split(x1, self.split_size, dim=1)  # [x1_0, x1_1]
        x2_parts = torch.split(x2, self.split_size, dim=1)  # [x2_0, x2_1]

        y1 = x1_parts[0] + x2_parts[1]
        y2 = x1_parts[1] + x2_parts[0]
        out = torch.cat([y1, y2], dim=1)
        
        return out[:, :x.size(1)]



