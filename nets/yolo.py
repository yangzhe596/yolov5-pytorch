import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640], high_res=False, four_features=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone_name  = backbone
        self.high_res       = high_res
        self.four_features  = four_features
        self.base_channels  = base_channels
        
        if backbone == "cspdarknet":
            #---------------------------------------------------#   
            #   生成CSPdarknet53的主干模型
            #   高分辨率四特征层模式：四个有效特征层
            #   160,160,128 (feat0)
            #   80,80,256 (feat1)
            #   40,40,512 (feat2)
            #   20,20,1024 (feat3)
            #   高分辨率三特征层模式：三个有效特征层
            #   160,160,128 (feat0)
            #   80,80,256 (feat1)
            #   40,40,512 (feat2)
            #   原始模式：三个有效特征层
            #   80,80,256 (feat1)
            #   40,40,512 (feat2)
            #   20,20,1024 (feat3)
            #---------------------------------------------------#
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained, high_res=high_res, four_features=four_features)
        else:
            #---------------------------------------------------#   
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            #---------------------------------------------------#
            self.backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape, high_res=high_res, four_features=four_features)
            
            if high_res:
                in_channels = {
                    'convnext_tiny'         : [96, 192, 384],  # 更高分辨率的特征通道
                    'convnext_small'        : [96, 192, 384],
                    'swin_transfomer_tiny'  : [96, 192, 384],
                }[backbone]
                feat0_c, feat1_c, feat2_c = in_channels 
                self.conv_1x1_feat0 = Conv(feat0_c, base_channels * 2, 1, 1)
                self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
                self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            else:
                in_channels = {
                    'convnext_tiny'         : [192, 384, 768],
                    'convnext_small'        : [192, 384, 768],
                    'swin_transfomer_tiny'  : [192, 384, 768],
                }[backbone]
                feat1_c, feat2_c, feat3_c = in_channels 
                self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
                self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
                self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        
        if high_res:
            # 高分辨率模式的FPN结构
            # 40, 40, 512 -> 40, 40, 256
            self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
            # 40, 40, 256 -> 80, 80, 256
            self.conv3_for_upsample1    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

            # 80, 80, 256 -> 80, 80, 128
            self.conv_for_feat1         = Conv(base_channels * 4, base_channels * 2, 1, 1)
            # 80, 80, 128 -> 160, 160, 128
            self.conv3_for_upsample2    = C3(base_channels * 4, base_channels * 2, base_depth, shortcut=False)

            # 下采样路径
            # 160, 160, 128 -> 80, 80, 128
            self.down_sample1           = Conv(base_channels * 2, base_channels * 2, 3, 2)
            # 80, 80, 128 cat 80, 80, 128 -> 80, 80, 256
            self.conv3_for_downsample1  = C3(base_channels * 4, base_channels * 4, base_depth, shortcut=False)

            # 80, 80, 256 -> 40, 40, 256
            self.down_sample2           = Conv(base_channels * 4, base_channels * 4, 3, 2)
            # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
            self.conv3_for_downsample2  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
            
            if four_features:
                # 四特征层模式：新增 P5 特征层
                # 40, 40, 512 -> 20, 20, 512
                self.down_sample3           = Conv(base_channels * 8, base_channels * 8, 3, 2)
                
                # P5 特征层的 1x1 卷积
                self.conv_for_P5 = Conv(base_channels * 24, base_channels * 16, 1, 1)
                
                # 检测头 - 支持4个特征层
                # 160, 160, 128 => 160, 160, 3 * (5 + num_classes)
                self.yolo_head_P2 = nn.Conv2d(base_channels * 2, len(anchors_mask[3]) * (5 + num_classes), 1)
                # 80, 80, 256 => 80, 80, 3 * (5 + num_classes)
                self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
                # 40, 40, 512 => 40, 40, 3 * (5 + num_classes)
                self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
                # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes)
                self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)
            else:
                # 三特征层模式：不使用 P5
                # 检测头 - 支持3个特征层
                # 160, 160, 128 => 160, 160, 3 * (5 + num_classes)
                self.yolo_head_P2 = nn.Conv2d(base_channels * 2, len(anchors_mask[2]) * (5 + num_classes), 1)
                # 80, 80, 256 => 80, 80, 3 * (5 + num_classes)
                self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[1]) * (5 + num_classes), 1)
                # 40, 40, 512 => 40, 40, 3 * (5 + num_classes)
                self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[0]) * (5 + num_classes), 1)
        else:
            # 原始模式的FPN结构
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

    def forward(self, x):
        if self.high_res:
            # 高分辨率模式：输出160x160, 80x80, 40x40, (20x20)特征层
            if self.backbone_name == "cspdarknet":
                feat0, feat1, feat2 = self.backbone(x)  # 160x160, 80x80, 40x40
                if self.four_features:
                    # 需要生成P5特征层 (20x20)
                    feat3 = self.backbone.get_p5_feature(x)  # 获取P5特征层
            else:
                feat0, feat1, feat2 = self.backbone(x)  # 160x160, 80x80, 40x40
                feat0 = self.conv_1x1_feat0(feat0)
                feat1 = self.conv_1x1_feat1(feat1)
                feat2 = self.conv_1x1_feat2(feat2)
                if self.four_features:
                    # 对于其他主干网络，也需要获取P5特征
                    feat3 = self.backbone.get_p5_feature(x)
                    if hasattr(self, 'conv_1x1_feat3'):
                        feat3 = self.conv_1x1_feat3(feat3)

            # 新的FPN结构，从最低分辨率开始
            # 40, 40, 512 -> 40, 40, 256
            P4          = self.conv_for_feat2(feat2)
            # 40, 40, 256 -> 80, 80, 256
            P4_upsample = self.upsample(P4)
            # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
            P3          = torch.cat([P4_upsample, feat1], 1)
            # 80, 80, 512 -> 80, 80, 256
            P3          = self.conv3_for_upsample1(P3)

            # 80, 80, 256 -> 80, 80, 128
            P3          = self.conv_for_feat1(P3)
            # 80, 80, 128 -> 160, 160, 128
            P3_upsample = self.upsample(P3)
            # 160, 160, 128 cat 160, 160, 128 -> 160, 160, 256
            P2          = torch.cat([P3_upsample, feat0], 1)
            # 160, 160, 256 -> 160, 160, 128
            P2          = self.conv3_for_upsample2(P2)
            
            # 下采样路径
            # 160, 160, 128 -> 80, 80, 128
            P2_downsample = self.down_sample1(P2)
            # 80, 80, 128 cat 80, 80, 128 -> 80, 80, 256
            P3 = torch.cat([P2_downsample, P3], 1)
            # 80, 80, 256 -> 80, 80, 256
            P3 = self.conv3_for_downsample1(P3)

            # 80, 80, 256 -> 40, 40, 256
            P3_downsample = self.down_sample2(P3)
            # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
            P4 = torch.cat([P3_downsample, P4], 1)
            # 40, 40, 512 -> 40, 40, 512
            P4 = self.conv3_for_downsample2(P4)
            
            if self.four_features:
                # 四特征层模式：新增 P5 特征层
                P4_downsample = self.down_sample3(P4)
                # 20, 20, 512 cat 20, 20, feat3 -> 20, 20, 1024
                P5 = torch.cat([P4_downsample, feat3], 1)
                # 20, 20, 1536 -> 20, 20, 1024
                # 使用预定义的1x1卷积调整通道数，而不是C3模块
                P5 = self.conv_for_P5(P5)

                #---------------------------------------------------#
                #   第四个特征层 (最高分辨率)
                #   out3=(batch_size,75,160,160)
                #---------------------------------------------------#
                out3 = self.yolo_head_P2(P2)
                #---------------------------------------------------#
                #   第三个特征层 (中高分辨率)
                #   out2=(batch_size,75,80,80)
                #---------------------------------------------------#
                out2 = self.yolo_head_P3(P3)
                #---------------------------------------------------#
                #   第二个特征层 (中低分辨率)
                #   out1=(batch_size,75,40,40)
                #---------------------------------------------------#
                out1 = self.yolo_head_P4(P4)
                #---------------------------------------------------#
                #   第一个特征层 (最低分辨率)
                #   out0=(batch_size,75,20,20)
                #---------------------------------------------------#
                out0 = self.yolo_head_P5(P5)
                
                return out0, out1, out2, out3
            else:
                # 三特征层模式：只使用 P2, P3, P4
                #---------------------------------------------------#
                #   第三个特征层 (最高分辨率)
                #   out2=(batch_size,75,160,160)
                #---------------------------------------------------#
                out2 = self.yolo_head_P2(P2)
                #---------------------------------------------------#
                #   第二个特征层 (中等分辨率)
                #   out1=(batch_size,75,80,80)
                #---------------------------------------------------#
                out1 = self.yolo_head_P3(P3)
                #---------------------------------------------------#
                #   第一个特征层 (最低分辨率)
                #   out0=(batch_size,75,40,40)
                #---------------------------------------------------#
                out0 = self.yolo_head_P4(P4)
                
                return out0, out1, out2
        else:
            # 原始模式：输出80x80, 40x40, 20x20三个特征层
            feat1, feat2, feat3 = self.backbone(x)
            if self.backbone_name != "cspdarknet":
                feat1 = self.conv_1x1_feat1(feat1)
                feat2 = self.conv_1x1_feat2(feat2)
                feat3 = self.conv_1x1_feat3(feat3)

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

