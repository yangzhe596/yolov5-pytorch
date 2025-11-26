import json

with open('/mnt/data/code/yolov5-pytorch/datasets/fred_fusion_test/annotations/instances_val.json', 'r') as f:
    data = json.load(f)

print(f"总图片数: {len(data['images'])}")
print(f"总标注数: {len(data['annotations'])}")
print(f"类别: {data['categories']}")
print(f"前5张图片信息:")
for img in data['images'][:5]:
    print(f"  ID={img['id']}, RGB={img['rgb_file_name']}, Event={img['event_file_name']}")
print(f"\n前5个标注:")
for ann in data['annotations'][:5]:
    print(f"  ImageID={ann['image_id']}, bbox={ann['bbox']}, modality={ann['modality']}")