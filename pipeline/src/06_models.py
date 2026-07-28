import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        attention_map = self.sigmoid(out)
        return attention_map

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
        # Identity node specifically designed for Grad-CAM to attach a hook
        self.output_node = nn.Identity()

    def forward(self, x):
        x_out = x * self.ca(x)
        spatial_att = self.sa(x_out)
        x_out = x_out * spatial_att
        
        # Pass through identity node for Grad-CAM extraction
        return self.output_node(x_out), spatial_att

class ClassificationModel(nn.Module):
    def __init__(self, architecture: str, freeze_percent: float, use_cbam: bool = False):
        super().__init__()
        self.architecture = architecture
        self.use_cbam = use_cbam
        
        if architecture == "DenseNet121":
            # Load pretrained weights
            self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier.in_features
            
            if self.use_cbam:
                self.cbam = CBAM(in_features)
                
            # Replace classifier head for binary classification
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 1)
            )
            self._freeze_layers_densenet(freeze_percent)
            
        elif architecture == "ResNet50":
            # Load pretrained weights
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            
            if self.use_cbam:
                self.cbam = CBAM(in_features)
                
            # Replace classifier head
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, 1)
            )
            self._freeze_layers_resnet(freeze_percent)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x, return_attention=False):
        if self.architecture == "DenseNet121":
            features = self.backbone.features(x)
            features = F.relu(features, inplace=True)
            
            spatial_att = None
            if self.use_cbam:
                features, spatial_att = self.cbam(features)
                
            out = F.adaptive_avg_pool2d(features, (1, 1))
            out = torch.flatten(out, 1)
            logits = self.backbone.classifier(out)
            
        elif self.architecture == "ResNet50":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)
            
            spatial_att = None
            if self.use_cbam:
                features, spatial_att = self.cbam(features)
                
            out = self.backbone.avgpool(features)
            out = torch.flatten(out, 1)
            logits = self.backbone.fc(out)
            
        if return_attention:
            return logits, spatial_att
        return logits
        
    def _freeze_layers_densenet(self, freeze_percent):
        """Freezes the specified percentage of layers from the bottom up."""
        total_layers = len(list(self.backbone.features.children()))
        freeze_up_to = int(total_layers * freeze_percent)
        
        for i, child in enumerate(self.backbone.features.children()):
            if i < freeze_up_to:
                for param in child.parameters():
                    param.requires_grad = False
                    
    def _freeze_layers_resnet(self, freeze_percent):
        """Freezes the specified percentage of layers from the bottom up for ResNet."""
        layers = [
            self.backbone.conv1, 
            self.backbone.bn1, 
            self.backbone.relu, 
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        ]
        
        total_layers = len(layers)
        freeze_up_to = int(total_layers * freeze_percent)
        
        for i, layer in enumerate(layers):
            if i < freeze_up_to:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes all layers for Stage 2 training."""
        for param in self.parameters():
            param.requires_grad = True
