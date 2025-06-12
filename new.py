from PIL import Image
from pathlib import Path
import torchvision.transforms as T

# 데이터 경로 설정
input_dir = Path("raw_images/")
output_dir = Path("processed_images/")
output_dir.mkdir(parents=True, exist_ok=True)

# 이미지 증강 파이프라인
transform = T.Compose([
    T.Resize((640, 640)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.ToTensor()
])

# 이미지 전처리 및 저장
for img_path in input_dir.glob("*.jpg"):
    image = Image.open(img_path).convert("RGB")
    transformed = transform(image)
    T.ToPILImage()(transformed).save(output_dir / img_path.name)