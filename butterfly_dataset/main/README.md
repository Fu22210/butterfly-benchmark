---
license: cc0-1.0
task_categories:
- object-detection
tags:
- inaturalist
- lepidoptera
---

# Основной набор

400 фотографий для сравнения детекторов взрослых бабочек. Изображения скачаны в размере original из публичного хранилища iNaturalist.
На каждую фотографию есть строка в `metadata.jsonl` с путём, происхождением и проверками лицензии.
Категория и жизненная стадия описывают наблюдение в iNaturalist. В наборе есть 10 фото взрослых ночных бабочек. При разметке им и взрослым дневным бабочкам назначается одна целевая метка «бабочка». Разметка рамок ещё не создана.

Для локальной загрузки через Hugging Face Datasets выполните:

```python
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="/path/to/main", split="train")
```
