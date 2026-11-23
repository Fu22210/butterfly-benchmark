# Пилотный сбор iNaturalist

Эта ранняя выборка содержит 500 снимков. Разбиения охватывают основной пул, тренировку, экзамены и контроль. Для итогового лидерборда использовалась отдельная [командная выборка](../butterfly_dataset/).

[Сборщик](collect.py) · [Квоты отбора](sampling_config.json) · [Манифест](manifests/all500.jsonl) · [Описание процедуры](README.txt)

Распакуйте [pilot-media.zip](https://github.com/Fu22210/butterfly-benchmark/releases/download/v1.0.0/pilot-media.zip) в корень репозитория. После этого можно открыть локальную галерею pilot/gallery.html или проверить файлы из корня клона.

```bash
python3 -m pip install -r pilot/requirements.txt
python3 pilot/collect.py validate
```
