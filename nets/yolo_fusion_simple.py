import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny


#---------------------------------------------------#
#   Fusion YoloBody
#---------------------------------------------------#
class YoloBodyFusion(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBodyFusion, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        
        self.backbone_name  = backbone
        self.base_channels  = base_channels
        self.base_depth     = base_depth
        
        #-----------------------------------------------#
        #   创建两个相同的backbone，一个处理RGB，一个处理Event
        #-----------------------------------------------#
        if backbone == "cspdarknet":
            # RGB backbone
            self.rgb_backbone = CSPDarknet(base_channels, base_depth, phi, pretrained)
            # Event backbone (不使用预训练权重)
            self.event_backbone = CSPDarknet(base_channels, base_depth, phi, False)
        else:
            # RGB backbone
            self.rgb_backbone = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            
            # Event backbone (不使用预训练权重)
            self.event_backbone = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=False, input_shape=input_shape)
            
            in_channels = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels 
            
            # 1x1卷积调整通道数
            self.rgb_conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.rgb_conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.rgb_conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)
            
            self.event_conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.event_conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.event_conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)
        
        #-----------------------------------------------#
        #   特征融合层：拼接后用1x1卷积压缩维度
        #-----------------------------------------------#
        # 对三个特征层分别进行融合
        if backbone == "cspdarknet":
            # CSPDarknet的输出通道已经是标准格式
            self.fusion_feat1 = nn.Conv2d(base_channels * 8, base_channels * 4, 1, 1)  # 256*2 -> 256
            self.fusion_feat2 = nn.Conv2d(base_channels * 16, base_channels * 8, 1, 1)  # 512*2 -> 512
            self.fusion_feat3 = nn.Conv2d(base_channels * 32, base_channels * 16, 1, 1)  # 1024*2 -> 1024
        else:
            # 其他backbone的输出通道调整后也已经是标准格式
            self.fusion_feat1 = nn.Conv2d(base_channels * 8, base_channels * 4, 1, 1)
            self.fusion_feat2 = nn.Conv2d(base_channels * 16, base_channels * 8, 1, 1)
            self.fusion_feat3 = nn.Conv2d(base_channels * 32, base_channels * 16, 1, 1)
        
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, rgb_input, event_input):
        #  分别提取RGB和Event特征
        rgb_feat1, rgb_feat2, rgb_feat3 = self.rgb_backbone(rgb_input)
        event_feat1, event_feat2, event_feat3 = self.event_backbone(event_input)
        
        #  调整非CSPDarknet的通道数
        if self.backbone_name != "cspdarknet":
            rgb_feat1 = self.rgb_conv_1x1_feat1(rgb_feat1)
            rgb_feat2 = self.rgb_conv_1x1_feat2(rgb_feat2)
            rgb_feat3 = self.rgb_conv_1x1_feat3(rgb_feat3)
            
            event_feat1 = self.event_conv_1x1_feat1(event_feat1)
            event_feat2 = self.event_conv_1x1_feat2(event_feat2)
            event_feat3 = self.event_conv_1x1_feat3(event_feat3)
        
        #  特征融合：拼接 + 1x1卷积压缩
        # feat1: 80,80,256 -> concat -> 80,80,512 -> 1x1conv -> 80,80,256
        feat1 = torch.cat([rgb_feat1, event_feat1], dim=1)
        feat1 = self.fusion_feat1(feat1)
        
        # feat2: 40,40,512 -> concat -> 40,40,1024 -> 1x1conv -> 40,40,512
        feat2 = torch.cat([rgb_feat2, event_feat2], dim=1)
        feat2 = self.fusion_feat2(feat2)
        
        # feat3: 20,20,1024 -> concat -> 20,20,2048 -> 1x1conv -> 20,20,1024
        feat3 = torch.cat([rgb_feat3, event_feat3], dim=1)
        feat3 = self.fusion_feat3(feat3)

        #  以下是原始的YOLOv5 FPN+PAN结构，保持不变
        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2


# 测试函数
def test_fusion_model():
    """测试融合模型"""
    print("Testing YOLOv5 Fusion Model...")
    
    # 创建模型
    model = YoloBodyFusion(
        anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        num_classes=1,
        phi='s',
        backbone='cspdarknet',
        pretrained=False
    )
    
    # 测试前向传播
    rgb_input = torch.randn(2, 3, 640, 640)
    event_input = torch.randn(2, 3, 640, 640)
    
    with torch.no_grad():
        outputs = model(rgb_input, event_input)
    
    print(f"RGB input shape: {rgb_input.shape}")
    print(f"Event input shape: {event_input.shape}")
    print(f"Number of detection outputs: {len(outputs)}")
    
    for i, output in enumerate(outputs):
        print(f"Detection {i} shape: {output.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Fusion model test completed successfully!")


if __name__ == "__main__":
    test_fusion_model()