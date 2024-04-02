from coco_eval import CocoEvaluator
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

dataset = CocoDetection(root="path_to_your_images", annFile="path_to_annotation_file")

dataloader = DataLoader(dataset, batch_size=2)

evaluator = CocoEvaluator(coco_gt=dataset.coco, iou_types=["bbox"])

model = ...

for batch in dataloader:
   predictions = model(batch)
   
   evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()