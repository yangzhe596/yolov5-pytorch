#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时修复：为 Fusion 模型修复 mAP 评估问题
通过修改回调函数支持双模态输入
"""
import torch
from utils.callbacks_coco import CocoEvalCallback

class FusionCocoEvalCallback(CocoEvalCallback):
    """支持 Fusion 模型的 COCO 评估回调"""
    
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, 
                 coco_json_path, image_dir, log_dir, cuda,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, 
                 nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, test_mode="fusion_rgb"):
        """
        Args:
            test_mode: 评估模式
                - "fusion_rgb": 使用 RGB 图像评估
                - "fusion_both": 同时使用 RGB 和 Event 图像评估
        """
        super().__init__(net, input_shape, anchors, anchors_mask, class_names, num_classes, 
                        coco_json_path, image_dir, log_dir, cuda,
                        map_out_path, max_boxes, confidence, 
                        nms_iou, letterbox_image, MINOVERLAP, test_mode)
        
        # Fusion 模型使用 RGB 图像作为主要模态进行评估
        self.fusion_test_mode = test_mode
    
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        """生成预测结果txt文件（支持 Fusion 模型）"""
        # 如果是 RGB 图像，同时生成 Event 图像路径
        if hasattr(self, 'modality_type') and self.modality_type == 'rgb_only':
            # 单 RGB 模式评估
            return super().get_map_txt(image_id, image, class_names, map_out_path)
        
        # Fusion 模式评估
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        
        # 预处理
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        
        # 生成对应的 Event 图像（如果是 Fusion 模型）
        # 这里简化处理：使用相同的 RGB 图像作为 Event 图像
        # 实际应用中应该使用真实的 Event 图像数据
        event_image_data = image_data.copy()
        
        # 准备 Fusion 模型输入
        if self.cuda:
            rgb_images = torch.from_numpy(image_data).cuda()
            event_images = torch.from_numpy(event_image_data).cuda()
            
            with torch.no_grad():
                outputs = self.net(rgb_images, event_images)
                outputs = self.bbox_util.decode_box(outputs)
                results = self.bbox_util.non_max_suppression(
                    torch.cat(outputs, 1), self.num_classes, self.input_shape,
                    image_shape, self.letterbox_image, 
                    conf_thres=self.confidence, nms_thres=self.nms_iou
                )
        else:
            rgb_images = torch.from_numpy(image_data)
            event_images = torch.from_numpy(event_image_data)
            
            with torch.no_grad():
                outputs = self.net(rgb_images, event_images)
                outputs = self.bbox_util.decode_box(outputs)
                results = self.bbox_util.non_max_suppression(
                    torch.cat(outputs, 1), self.num_classes, self.input_shape,
                    image_shape, self.letterbox_image, 
                    conf_thres=self.confidence, nms_thres=self.nms_iou
                )
        
        if results[0] is None:
            f.close()
            return
        
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            
            top, left, bottom, right = box
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))
            ))
        
        f.close()
        return

def patch_coco_eval_callback():
    """修补全局的 CocoEvalCallback 以支持 Fusion 模型"""
    # 临时保存原始类
    original_coco_eval = CocoEvalCallback
    
    # 返回修补后的类
    return FusionCocoEvalCallback

# 简单的测试代码
if __name__ == "__main__":
    print("Fusion 评估修复脚本")
    print("请在训练脚本中使用 FusionCocoEvalCallback 替换 CocoEvalCallback")