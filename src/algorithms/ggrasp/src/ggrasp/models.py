import torch.nn as nn
import torch.nn.functional as F

class GGCNN2(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super(GGCNN2, self).__init__()

        if filter_sizes is None:
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            16]  # Transpose Convs

        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            # 4 conv layers.
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
            nn.ReLU(inplace=True),

            # Output layers
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.features(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

class GRConvNet(nn.Module):
    def __init__(self, input_channels=1, channel_size=32, dropout_prob=0.1):
        super(GRConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1,padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size // 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size // 2)

        self.conv3 = nn.Conv2d(channel_size // 2, channel_size // 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size // 4)

        self.res1 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res2 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res3 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res4 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res5 = ResidualBlock(channel_size // 4, channel_size // 4)

        self.conv4 = nn.ConvTranspose2d(channel_size // 4, channel_size // 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size // 2)

        self.conv5 = nn.ConvTranspose2d(channel_size // 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=2)

        self.dropout_pos = nn.Dropout(p=dropout_prob)
        self.dropout_cos = nn.Dropout(p=dropout_prob)
        self.dropout_sin = nn.Dropout(p=dropout_prob)
        self.dropout_wid = nn.Dropout(p=dropout_prob)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        pos_output = self.pos_output(self.dropout_pos(x))
        cos_output = self.cos_output(self.dropout_cos(x))
        sin_output = self.sin_output(self.dropout_sin(x))
        width_output = self.width_output(self.dropout_wid(x))
        
        return pos_output, cos_output, sin_output, width_output
