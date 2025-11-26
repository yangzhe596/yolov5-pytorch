import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny


#---------------------------------------------------#
#   融合YOLO网络：同时处理RGB和Event两种模态
#   使用相同的backbone提取特征，直接拼接两种模态的特征，
#   然后通过1*1卷积压缩维度
#---------------------------------------------------#
class YoloFusionBody(nn.Module):
    """
    融合YOLO网络
    输入:
        - rgb_image: RGB图像 [B, 3, H, W]
        - event_image: Event图像 [B, 3, H, W] (已转换为伪彩色)
    
    处理流程:
        1. 使用相同的backbone分别提取RGB和Event特征
        2. 拼接两种模态的特征
        3. 使用1*1卷积压缩维度
        4. 后续处理与单模态保持一致
    """
    
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', 
                 pretrained=False, input_shape=[640, 640], high_res=False, 
                 four_features=False, fusion_mode='concat', compression_ratio=1.0):
        """
        初始化融合网络
        
        Args:
            anchors_mask: 先验框掩码
            num_classes: 类别数量
            phi: 模型尺寸 ('s', 'm', 'l', 'x')
            backbone: 主干网络类型
            pretrained: 是否使用预训练权重
            input_shape: 输入尺寸 [height, width]
            high_res: 是否使用高分辨率模式
            four_features: 是否使用四特征层
            fusion_mode: 融合模式 ('concat', 'add', 'channel_attn')
            compression_ratio: 压缩比率（0.5 表示压缩到一半）
        """
        super(YoloFusionBody, self).__init__()
        
        depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]
        
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        
        self.backbone_name = backbone
        self.high_res = high_res
        self.four_features = four_features
        self.base_channels = base_channels
        self.fusion_mode = fusion_mode
        self.compression_ratio = compression_ratio
        
        # 1. 创建两个相同的backbone（共享权重）
        if backbone == "cspdarknet":
            self.backbone_rgb = CSPDarknet(base_channels, base_depth, phi, pretrained, 
                                         high_res=high_res, four_features=four_features)
            self.backbone_event = CSPDarknet(base_channels, base_depth, phi, pretrained, 
                                           high_res=high_res, four_features=four_features)
        else:
            backbone_class = {
                'convnext_tiny': ConvNeXt_Tiny,
                'convnext_small': ConvNeXt_Small,
                'swin_transfomer_tiny': Swin_transformer_Tiny,
            }[backbone]
            
            self.backbone_rgb = backbone_class(pretrained=pretrained, input_shape=input_shape, 
                                             high_res=high_res, four_features=four_features)
            self.backbone_event = backbone_class(pretrained=pretrained, input_shape=input_shape, 
                                               high_res=high_res, four_features=four_features)
        
        # 2. 确定特征通道数
        if high_res:
            if four_features:
                self.feat_channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
            else:
                self.feat_channels = [base_channels * 2, base_channels * 4, base_channels * 8]
        else:
            self.feat_channels = [base_channels * 4, base_channels * 8, base_channels * 16]
        
        # 3. 创建1*1卷积用于特征压缩
        self.compression_convs = nn.ModuleList([
            Conv(int(c * 2), int(c * compression_ratio), 1, 1)  # 拼接后通道数翻倍
            for c in self.feat_channels
        ])
        self.out_channels = [int(c * compression_ratio) for c in self.feat_channels]
        
        # 3. FPN和PANet结构（简化版，直接使用压缩后的通道数）
        # 注意：为了简化实现，我们直接使用压缩后的特征图，不进行额外的FPN/PANet变换
        # 这样可以保持与原始YOLOv5相同的架构，只是输入特征图来自融合后的结果
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        if high_res:
            if four_features:
                # 四特征层模式使用原有的FPN结构
                # 参考原代码，我们需要重建FPN
                # 40, 40, compressed -> 40, 40, compressed/2
                self.conv_for_feat2 = Conv(self.out_channels[2], self.out_channels[1], 1, 1)
                # 40, 40, compressed/2 -> 80, 80, compressed/2
                self.conv3_for_upsample1 = C3(self.out_channels[1] + self.out_channels[1], 
                                             self.out_channels[1], base_depth, shortcut=False)
                
                # 80, 80, compressed/2 -> 80, 80, compressed/3
                self.conv_for_feat1 = Conv(self.out_channels[1], self.out_channels[0], 1, 1)
                # 80, 80, compressed/3 -> 160, 160, compressed/4
                self.conv3_for_upsample2 = C3(self.out_channels[0] + self.out_channels[0], 
                                             self.out_channels[0], base_depth, shortcut=False)
                
                # 下采样路径
                # 160, 160, compressed/4 -> 80, 80, compressed/4
                self.down_sample1 = Conv(self.out_channels[0], self.out_channels[0], 3, 2)
                # 80, 80, compressed/4 cat 80, 80, compressed/3 -> 80, 80, compressed/3
                self.conv3_for_downsample1 = C3(self.out_channels[0] + self.out_channels[0],
                                               self.out_channels[1], base_depth, shortcut=False)
                
                # 80, 80, compressed/3 -> 40, 40, compressed/3
                self.down_sample2 = Conv(self.out_channels[1], self.out_channels[1], 3, 2)
                # 40, 40, compressed/3 cat 40, 40, compressed/2 -> 40, 40, compressed/2
                self.conv3_for_downsample2 = C3(self.out_channels[1] + self.out_channels[1],
                                               self.out_channels[2], base_depth, shortcut=False)
                
                # 40, 40, compressed/2 -> 20, 20, compressed
                self.down_sample3 = Conv(self.out_channels[2], self.out_channels[3], 3, 2)
                self.conv_for_P5 = Conv(self.out_channels[3], self.out_channels[3], 1, 1)
                
                # 检测头
                self.yolo_head_P2 = nn.Conv2d(self.out_channels[0], 
                                             len(anchors_mask[3]) * (5 + num_classes), 1)
                self.yolo_head_P3 = nn.Conv2d(self.out_channels[1], 
                                             len(anchors_mask[2]) * (5 + num_classes), 1)
                self.yolo_head_P4 = nn.Conv2d(self.out_channels[2], 
                                             len(anchors_mask[1]) * (5 + num_classes), 1)
                self.yolo_head_P5 = nn.Conv2d(self.out_channels[3], 
                                             len(anchors_mask[0]) * (5 + num_classes), 1)
            else:
                # 三特征层高分辨率模式
                self.conv_for_feat2 = Conv(self.out_channels[2], self.out_channels[1], 1, 1)
                self.conv3_for_upsample1 = C3(self.out_channels[1] + self.out_channels[1], 
                                             self.out_channels[1], base_depth, shortcut=False)
                
                self.conv_for_feat1 = Conv(self.out_channels[1], self.out_channels[0], 1, 1)
                self.conv3_for_upsample2 = C3(self.out_channels[0] + self.out_channels[0], 
                                             self.out_channels[0], base_depth, shortcut=False)
                
                self.down_sample1 = Conv(self.out_channels[0], self.out_channels[0], 3, 2)
                self.conv3_for_downsample1 = C3(self.out_channels[0] + self.out_channels[0],
                                               self.out_channels[1], base_depth, shortcut=False)
                
                self.down_sample2 = Conv(self.out_channels[1], self.out_channels[1], 3, 2)
                self.conv3_for_downsample2 = C3(self.out_channels[1] + self.out_channels[1],
                                               self.out_channels[2], base_depth, shortcut=False)
                
                # 检测头
                self.yolo_head_P2 = nn.Conv2d(self.out_channels[0], 
                                             len(anchors_mask[2]) * (5 + num_classes), 1)
                self.yolo_head_P3 = nn.Conv2d(self.out_channels[1], 
                                             len(anchors_mask[1]) * (5 + num_classes), 1)
                self.yolo_head_P4 = nn.Conv2d(self.out_channels[2], 
                                             len(anchors_mask[0]) * (5 + num_classes), 1)
        else:
            # 原始模式
            self.conv_for_feat3 = Conv(self.out_channels[2], self.out_channels[1], 1, 1)
            self.conv3_for_upsample1 = C3(self.out_channels[1] + self.out_channels[1],
                                         self.out_channels[1], base_depth, shortcut=False)
            
            self.conv_for_feat2 = Conv(self.out_channels[1], self.out_channels[0], 1, 1)
            self.conv3_for_upsample2 = C3(self.out_channels[0] + self.out_channels[0],
                                         self.out_channels[0], base_depth, shortcut=False)
            
            self.down_sample1 = Conv(self.out_channels[0], self.out_channels[0], 3, 2)
            self.conv3_for_downsample1 = C3(self.out_channels[0] + self.out_channels[0],
                                           self.out_channels[1], base_depth, shortcut=False)
            
            self.down_sample2 = Conv(self.out_channels[1], self.out_channels[1], 3, 2)
            self.conv3_for_downsample2 = C3(self.out_channels[1] + self.out_channels[1],
                                           self.out_channels[2], base_depth, shortcut=False)
            
            # 检测头
            self.yolo_head_P3 = nn.Conv2d(self.out_channels[0], 
                                         len(anchors_mask[2]) * (5 + num_classes), 1)
            self.yolo_head_P4 = nn.Conv2d(self.out_channels[1], 
                                         len(anchors_mask[1]) * (5 + num_classes), 1)
            self.yolo_head_P5 = nn.Conv2d(self.out_channels[2], 
                                         len(anchors_mask[0]) * (5 + num_classes), 1)
    
    def forward(self, rgb_image, event_image=None):
        """
        前向传播
        
        Args:
            rgb_image: RGB图像 [B, 3, H, W] 或者 (rgb_image, event_image)
            event_image: Event图像 [B, 3, H, W] 或 None
            
        Returns:
            多层特征图列表
        """
        # 处理输入格式兼容性
        # 如果只传入一个参数，则假设是 rgb_image
        # 如果传入两个参数，则分别是 rgb_image 和 event_image
        if event_image is None:
            # 单模态模式：rgb_image 实际上是图像输入
            # 这种情况不应该发生，除非是单模态推理
            raise ValueError("Fusion模型需要传入两个图像：(rgb_image, event_image)")
        
        # 1. 使用相同的backbone提取特征
        rgb_features = self.backbone_rgb(rgb_image)
        event_features = self.backbone_event(event_image)
        
        # 2. 拼接两种模态的特征
        fused_features = []
        for rgb_feat, event_feat in zip(rgb_features, event_features):
            fused_feat = torch.cat([rgb_feat, event_feat], dim=1)  # [B, C1+C2, H, W]
            fused_features.append(fused_feat)
        
        # 3. 使用1*1卷积压缩维度
        compressed_features = []
        for i, fused_feat in enumerate(fused_features):
            compressed_feat = self.compression_convs[i](fused_feat)  # [B, C_out, H, W]
            compressed_features.append(compressed_feat)
        
        feat0, feat1, feat2 = compressed_features
        
        if self.high_res:
            # 高分辨率模式的FPN和PANet
            # 上采样路径
            P5 = self.conv_for_feat2(feat2)
            P5_upsample = self.upsample(P5)
            P4 = self.conv3_for_upsample1(torch.cat([feat1, P5_upsample], dim=1))
            
            P4 = self.conv_for_feat1(P4)
            P4_upsample = self.upsample(P4)
            P3 = self.conv3_for_upsample2(torch.cat([feat0, P4_upsample], dim=1))
            
            # 下采样路径
            P3_downsample = self.down_sample1(P3)
            P4 = self.conv3_for_downsample1(torch.cat([P4, P3_downsample], dim=1))
            
            P4_downsample = self.down_sample2(P4)
            P5 = self.conv3_for_downsample2(torch.cat([P5, P4_downsample], dim=1))
            
            if self.four_features:
                P5_downsample = self.down_sample3(P5)
                P5_final = self.conv_for_P5(torch.cat([feat2, P5_downsample], dim=1))
                
                out2 = self.yolo_head_P2(P3)
                out1 = self.yolo_head_P3(P4)
                out0 = self.yolo_head_P4(P5)
                out_p5 = self.yolo_head_P5(feat2)
                
                return out0, out1, out2, out_p5
            else:
                out2 = self.yolo_head_P2(P3)
                out1 = self.yolo_head_P3(P4)
                out0 = self.yolo_head_P4(P5)
                
                return out0, out1, out2
        else:
            # 原始模式的FPN和PANet
            # 上采样路径
            P5 = self.conv_for_feat3(feat2)
            P5_upsample = self.upsample(P5)
            P4 = self.conv3_for_upsample1(torch.cat([feat1, P5_upsample], dim=1))
            
            P4 = self.conv_for_feat2(P4)
            P4_upsample = self.upsample(P4)
            P3 = self.conv3_for_upsample2(torch.cat([feat0, P4_upsample], dim=1))
            
            # 下采样路径
            P3_downsample = self.down_sample1(P3)
            P4 = self.conv3_for_downsample1(torch.cat([P4, P3_downsample], dim=1))
            
            P4_downsample = self.down_sample2(P4)
            P5 = self.conv3_for_downsample2(torch.cat([P5, P4_downsample], dim=1))
            
            out2 = self.yolo_head_P3(P3)
            out1 = self.yolo_head_P4(P4)
            out0 = self.yolo_head_P5(P5)
            
            # 返回顺序与单模态保持一致: (P5, P4, P3) -> (20x20, 40x40, 80x80)
            return out0, out1, out2