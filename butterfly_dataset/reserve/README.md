---
license: cc0-1.0
task_categories:
- object-detection
tags:
- inaturalist
- lepidoptera
---

# Резервный набор

200 фотографий для подбора замен и расширения состава. Изображения скачаны в размере original из публичного хранилища iNaturalist.
На каждую фотографию есть строка в `metadata.jsonl` с путём, происхождением и проверками лицензии.
Категория и жизненная стадия описывают наблюдение в iNaturalist. Разметка рамок ещё не создана.

Для локальной загрузки через Hugging Face Datasets выполните:

```python
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="/path/to/reserve", split="train")
```
