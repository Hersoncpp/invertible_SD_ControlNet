# 新增模块使用说明

本文档介绍新增的三个模块：Pixel Shuffling、HAAR 小波变换和 Invertible 1x1 Conv。

## 1. Pixel Shuffling 模块

### 功能
用于可逆的上采样和下采样操作，常用于超分辨率任务。

### 使用方法

```python
from models.modules import PixelShuffle, InvertiblePixelShuffle, ConditionalPixelShuffle

# 基础 Pixel Shuffle
pixel_shuffle = PixelShuffle(upscale_factor=2)
# 上采样: (B, C*4, H, W) -> (B, C, 2*H, 2*W)
upsampled = pixel_shuffle(x, rev=False)
# 下采样: (B, C, 2*H, 2*W) -> (B, C*4, H, W)
downsampled = pixel_shuffle(upsampled, rev=True)

# 可逆 Pixel Shuffle（带可学习通道混洗）
inv_pixel_shuffle = InvertiblePixelShuffle(in_channels=64, upscale_factor=2)

# 条件 Pixel Shuffle（支持文本嵌入）
cond_pixel_shuffle = ConditionalPixelShuffle(
    in_channels=64, 
    upscale_factor=2, 
    embedding_dim=768
)
output = cond_pixel_shuffle(x, rev=False, text_embedding=text_emb)
```

## 2. HAAR 小波变换模块

### 功能
使用 HAAR 小波变换实现可逆的多尺度分解，常用于图像压缩和可逆下采样。

### 使用方法

```python
from models.modules import HaarTransform, LearnableHaarTransform, ConditionalHaarTransform

# 标准 HAAR 变换
haar = HaarTransform()
# 下采样: (B, C, H, W) -> (B, 4*C, H/2, W/2) [LL, LH, HL, HH 四个子带]
decomposed = haar(x, rev=False)
# 上采样: (B, 4*C, H/2, W/2) -> (B, C, H, W)
reconstructed = haar(decomposed, rev=True)

# 可学习的 HAAR 变换
learnable_haar = LearnableHaarTransform(in_channels=64)

# 条件 HAAR 变换（支持文本嵌入）
cond_haar = ConditionalHaarTransform(
    in_channels=64, 
    embedding_dim=768
)
output = cond_haar(x, rev=False, text_embedding=text_emb)
```

### HAAR 变换说明
- **LL**: 低频子带（近似信息）
- **LH**: 水平高频子带（垂直边缘）
- **HL**: 垂直高频子带（水平边缘）
- **HH**: 对角高频子带（对角边缘）

## 3. Invertible 1x1 Conv 模块

### 功能
用于可逆网络的通道混洗操作，基于 GLOW 和 RealNVP 中的可逆 1x1 卷积。

### 使用方法

```python
from models.modules import (
    Invertible1x1Conv, 
    Invertible1x1ConvSimple, 
    ConditionalInvertible1x1Conv
)

# 使用 LU 分解的高效实现（推荐）
inv_conv = Invertible1x1Conv(num_channels=64, use_lu=True)
# 正向: 通道混洗
output = inv_conv(x, rev=False)
# 反向: 逆通道混洗
reconstructed = inv_conv(output, rev=True)

# 简单实现（适合小通道数）
simple_conv = Invertible1x1ConvSimple(num_channels=64)

# 条件可逆 1x1 卷积（支持文本嵌入）
cond_conv = ConditionalInvertible1x1Conv(
    num_channels=64, 
    embedding_dim=768, 
    use_lu=True
)
output = cond_conv(x, rev=False, text_embedding=text_emb)

# 计算雅可比行列式（用于损失函数）
jacobian = inv_conv.jacobian(x, rev=False)
```

## 集成到现有架构

### 在 Inv_arch.py 中使用

```python
from models.modules import HaarTransform, Invertible1x1Conv, PixelShuffle

class CustomInvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 使用 HAAR 进行下采样
        self.haar = HaarTransform()
        # 使用可逆 1x1 卷积进行通道混洗
        self.inv_conv = Invertible1x1Conv(channels)
        # 使用 Pixel Shuffle 进行上采样
        self.pixel_shuffle = PixelShuffle(upscale_factor=2)
    
    def forward(self, x, rev=False):
        if not rev:
            # 下采样
            x = self.haar(x, rev=False)
            # 通道混洗
            x = self.inv_conv(x, rev=False)
            # 上采样
            x = self.pixel_shuffle(x, rev=False)
        else:
            # 反向操作
            x = self.pixel_shuffle(x, rev=True)
            x = self.inv_conv(x, rev=True)
            x = self.haar(x, rev=True)
        return x
```

## 注意事项

1. **HAAR 变换**: 输入的高度和宽度必须是偶数，如果不是会自动填充。
2. **Invertible 1x1 Conv**: 使用 LU 分解版本更高效，但需要确保通道数不太大。
3. **Pixel Shuffle**: 上采样因子必须是整数，且输出通道数 = 输入通道数 / (upscale_factor^2)。
4. **条件版本**: 所有模块都提供了支持文本嵌入的条件版本，可以用于文本引导的图像生成。

## 性能建议

- **小通道数 (< 128)**: 使用 `Invertible1x1ConvSimple`
- **大通道数 (>= 128)**: 使用 `Invertible1x1Conv` with `use_lu=True`
- **需要文本条件**: 使用对应的 `Conditional*` 版本
- **需要可学习变换**: 使用 `LearnableHaarTransform` 替代标准 HAAR

