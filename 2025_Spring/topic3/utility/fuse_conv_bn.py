import torch
import torch.nn as nn
import torchvision
import copy

# ------------------------------------------------------------------
# 1. Fuse a single Conv2d + BatchNorm2d pair
#    (Same fuse_conv_and_bn function as before)
# ------------------------------------------------------------------
def fuse_conv_and_bn(conv, bn):
    """
    Takes a conv layer and a batch norm layer, and returns
    a new conv layer with BN 'baked in'.
    """
    # Initialize fused Conv2d (must have bias because BN modifies bias)
    fusedconv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    )

    # Clone conv weight
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    # BN scale factor for each channel: gamma / sqrt(var + eps)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.running_var + bn.eps)))
    # Fused weights
    fused_weight = torch.mm(w_bn, w_conv).view(conv.weight.size())
    fusedconv.weight = nn.Parameter(fused_weight)

    # Construct fused bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.out_channels, dtype=conv.weight.dtype, device=conv.weight.device)

    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias = nn.Parameter(b_conv + b_bn)

    return fusedconv


# ------------------------------------------------------------------
# 2. Recursively fuse all Conv+BN pairs in a ResNet18 model
#    This function will modify the model in-place.
# ------------------------------------------------------------------
def fuse_resnet_conv_bn_inplace(module):
    """
    Recursively fuses all Conv2d+BatchNorm2d pairs in a (sub)module
    of a ResNet. Typical use: fuse_resnet_conv_bn_inplace(resnet18_model).
    """
    # If the module has children (like BasicBlock, etc.), we recurse down
    for name, child in list(module.named_children()):
        # 1) Recurse first so you go down into submodules (e.g., BasicBlock)
        fuse_resnet_conv_bn_inplace(child)
        
        # 2) Then check if child is exactly a BasicBlock or Bottleneck
        #    Because typical ResNet blocks have conv1+bn1, conv2+bn2, ...
        if isinstance(child, nn.Sequential):
            # E.g. the downsample in BasicBlock is often a Sequential of {Conv2d, BN2d}
            # but let's fuse inside that if it's just a two-layer sequence
            # We'll handle downsample as well
            if len(child) == 2:
                if isinstance(child[0], nn.Conv2d) and isinstance(child[1], nn.BatchNorm2d):
                    fused = fuse_conv_and_bn(child[0], child[1])
                    setattr(module, name, fused)  # replace child with fused conv
        elif isinstance(child, nn.BatchNorm2d):
            # If you see a BN without an immediate preceding conv in the same submodule,
            # it's typically because the ResNet BasicBlock has separate attributes (conv1,bn1), (conv2,bn2).
            # We'll try to fuse with the conv that appears just before it in the parent.
            # That logic requires we also look at the parent's attributes in order, or the block's structure.

            # Example: In BasicBlock:
            #   self.conv1 = nn.Conv2d(...)
            #   self.bn1   = nn.BatchNorm2d(...)
            #   self.conv2 = nn.Conv2d(...)
            #   self.bn2   = nn.BatchNorm2d(...)
            # So if "child" is "bn1", the sibling "conv1" might exist in the same module's __dict__.

            # Because we just have the child, let's see if there's a "conv" with the same suffix in the name:
            #   e.g. bn1 -> conv1, bn2 -> conv2
            # This is a bit hacky, but it's how the basic blocks are typically named.
            bn_name = name
            suffix = bn_name[-1]   # '1' or '2'
            conv_name = "conv" + suffix

            # If the sibling conv exists, fuse them
            if hasattr(module, conv_name):
                conv_sibling = getattr(module, conv_name)
                if isinstance(conv_sibling, nn.Conv2d):
                    # fuse them
                    fused = fuse_conv_and_bn(conv_sibling, child)
                    # replace the conv
                    setattr(module, conv_name, fused)
                    # The BN is now baked-in, so let's set BN to Identity
                    identity = nn.Identity()
                    setattr(module, bn_name, identity)

    return module


# ------------------------------------------------------------------
# 3. Main code to demonstrate fusing ResNet18
# ------------------------------------------------------------------
if __name__ == "__main__":
    # (A) Create a ResNet18 model from torchvision
    original_model = torchvision.models.resnet18(pretrained=False)
    original_model.eval()

    # Make a copy so we can compare original vs. fused
    fused_model = copy.deepcopy(original_model)
    
    # (B) Fuse in place
    fuse_resnet_conv_bn_inplace(fused_model)
    fused_model.eval()

    # (C) Validate: same output on random input
    # Create random input
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    
    
    with torch.no_grad():
        out_original = original_model(x)
        out_fused = fused_model(x)

    # Compare
    diff = (out_original - out_fused).abs().max().item()
    print(f"Max difference between original ResNet18 and fused ResNet18: {diff:.6g}")
    # Expect the difference to be on the order of 1e-6 ~ 1e-7 
    # depending on floating point precision.
