import torch
import torch.nn.functional as F
import torch.nn as nn 
from torchinfo import summary

# For the SRU block
class SRU(nn.Module):
    # separate and reconstruct
    def __init__(self, in_channels, group_num, gate_threshold):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=group_num, num_channels=in_channels)
        self.gate_threshhold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # group normalize X -> new_weight = sigmoid(normalized(gn(X))) -> separate by threshhold
        gn_x = self.gn(x)
        gamma = self.gn.weight
        w_gamma = gamma / (sum(gamma) + 1e-5)  # add a small eps for division
        w_gamma = w_gamma.view(1, -1, 1, 1)  # match the input x dimension
        new_weight = self.sigmoid(gn_x * w_gamma) # 以 normalize * gamma

        # set complementary masks
        mask_info = (new_weight >= self.gate_threshhold).float()
        mask_non_info = (new_weight < self.gate_threshhold).float()

        # separate
        x_w_1 = mask_info * x
        x_w_2 = mask_non_info * x

        # reconstruct
        # torch.chunk(input: Tensor, chunks: int, dim: int = 0)
        # 交叉中和掉特徵，比起直接把不重要的特徵全部丟掉好
        # chunk=2 等於 把原本的 channel 分成 2 部分
        # dim=1 等於把 feature map 從厚度上分成兩半
        # dim=2 等於把 feature map 分成上下兩半
        # example:
        # chunk=2 dim=1 -> 不動原來的圖，分成兩份
        # chunk=2 dim=2 -> 把原來的圖上下對切，分成兩份
        # chunk=4 dim = 2 -> 把原來的圖上下對切，分成 4 份

        # 結構
        # Tensor -> Channel(feature map)(切片) -> Feature vector(長棍)
        # Tensor: C*H*W, Channel: H*W (in pixels),  Feature vec: C
        # dim = 1 to split chennal dimension
        x_w_11, x_w_12 = torch.chunk(x_w_1, 2, dim=1)
        x_w_21, x_w_22 = torch.chunk(x_w_2, 2, dim=1)

        x_w1 = x_w_11 + x_w_22
        x_w2 = x_w_12 + x_w_21

        # torch.cat(tensors, dim=0, *, out=None) → Tensor
        # torch.cat() can be seen as an inverse operation for torch.split() and torch.chunk().
        # 所以確保 dim 一樣就可以直接使用 cat 還原剛剛被 chunk 的圖
        out = torch.cat([x_w1, x_w2], dim=1) # dim=1 確保最後厚度一樣
        return out
    
# For the CRU block
class CRU(nn.Module):
    def __init__(self, in_channels, out_channels, alpha, r,
                group_size, kernel_size):  # alpha = split ratio, r = sqeeze ratio
        super().__init__()

        # separate
        self.upper_channels = int(alpha * in_channels)
        self.lower_channels = in_channels - self.upper_channels

        self.upper_squeez_channels = int(self.upper_channels / r)
        self.lower_squeez_channels = int(self.lower_channels / r)

        # CHECK if the channel size is applicable
        if out_channels - self.lower_squeez_channels < 0:
            print("Please Adjust the Alpha and Squeeze Ratio in CRU to satisfy channel constraint in lower part")
            exit()

        # upper conv
        self.upper_conv1 = nn.Conv2d(self.upper_channels, self.upper_squeez_channels, kernel_size=1) # for getting X_up
        self.upper_gwc = nn.Conv2d(self.upper_squeez_channels, out_channels, kernel_size=kernel_size,
                                   padding=kernel_size//2, groups=group_size)
        self.upper_pwc = nn.Conv2d(self.upper_squeez_channels, out_channels, kernel_size=1)

        # lower conv
        self.lower_conv1 = nn.Conv2d(self.lower_channels, self.lower_squeez_channels, kernel_size=1) # for getting X_low
        # NOTE!!! WE HAVE TO HANDLE THE CASE OF C2 <= (1-alpha)C1/r
        # Which will happen in DenseNet because the bottleneck layer is 128 --> 32
        # if alpha is 0.5 and r is 2, we have lower_squeez_channels = 32, which makes the lower_pwc have 0 channel output
        if out_channels - self.lower_squeez_channels == 0:
            self.lower_pwc = None
        else:
            self.lower_pwc = nn.Conv2d(self.lower_squeez_channels, out_channels - self.lower_squeez_channels, kernel_size=1)

        # element-wise summation
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # basically means AvgPool2d(out_channels)

    def forward(self, x):
        # split channels
        x_upper, x_lower = torch.split(x, [self.upper_channels, self.lower_channels], dim=1)

        # 1*1 conv to squeeze
        x_upper_squeeze = self.upper_conv1(x_upper)
        x_lower_squeeze = self.lower_conv1(x_lower)

        # transform
        # For upper part
        y_upper = self.upper_gwc(x_upper_squeeze) + self.upper_pwc(x_upper_squeeze)
        # For lower part
        # HANDLE THE C2 <= (1-alpha)C1/r CASE
        if self.lower_pwc is None:
            y_lower = x_lower_squeeze
        else:
            y_lower = torch.cat([self.lower_pwc(x_lower_squeeze), x_lower_squeeze], dim=1)

        # Fuse
        s1 = self.global_avg_pool(y_upper)
        s2 = self.global_avg_pool(y_lower)

        # s1 ans s2 are in dimension of [N, C, 1, 1]
        # in order to do pair wise siftmax between s1 and s2
        # we first stacj them into [N, C, 1, 1, 2] tensor
        s_stacked = torch.stack([s1, s2], dim=-1)
        # apply softmax pairwise
        s_final = torch.softmax(s_stacked, dim=-1)
        # separate it back into 2 [N, C, 1, 1] tensors
        beta1 = s_final[... , 0]
        beta2 = s_final[... , 1]

        final_y_upper = y_upper * beta1
        final_y_lower = y_lower * beta2

        final_y = final_y_upper + final_y_lower

        return final_y
    
# The main SCConv class
class SCConv(nn.Module):
    def __init__(self, in_channels, out_channels, target_stride=1):
        super().__init__()
        # define paper's Hyper params
        self.SRU_NUM_GROUP = in_channels // 2  # Unsure...
        self.SRU_THRESHOLD = 0.5
        self.CRU_ALPHA = 0.5
        self.CRU_SQUEEZE_R = 2
        self.CRU_NUM_GROUP = 2
        self.CRU_KERNEL_SIZE = 3

        self.sru = SRU(in_channels, group_num=self.SRU_NUM_GROUP, gate_threshold=self.SRU_THRESHOLD)
        self.cru = CRU(in_channels, out_channels, alpha=self.CRU_ALPHA, r=self.CRU_SQUEEZE_R,
                       group_size=self.CRU_NUM_GROUP, kernel_size=self.CRU_KERNEL_SIZE)

        # We use AvgPool2d with kernel=2, stride=2 to halve width/height
        if target_stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity() # Do nothing

    def forward(self, x):
        x = self.downsample(x)
        x = self.sru(x)
        x = self.cru(x)
        return x

# To test out the SCConv blocks
def test_scconv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [Batch, Channel, Height, Width], batch 等於圖量
    input_tensor = torch.randn(1, 64, 32, 32).to(device)
    scconv = SCConv(64, 64).to(device)
    output_tensor = scconv(input_tensor)
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output_tensor.shape}")

    # Straightly simulate with the input data
    summary(scconv, (1, 64, 32, 32))

    # print out structure
    print(scconv)
    
if __name__ == "__main__":
    test_scconv()