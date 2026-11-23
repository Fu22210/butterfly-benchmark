# Экзаменационный набор

[400 входных фотографий](dataset.jsonl) · [Шаблон](submission-template.jsonl) · [Итоговые предсказания](submission.jsonl)

Qwen и OWLv2 независимо обработали каждый кадр. Конфигурация ансамбля выбрана на командном наборе. Файл содержит 212 рамок, для 188 изображений модель вернула пустой список. Команды ниже выполняются из каталога model_comparison.

```bash
python3 strategy.py apply strategy_results/config.json contest/predictions /tmp/butterfly-answer.jsonl
python3 contest/validate_submission.py contest/dataset.jsonl contest/submission-template.jsonl /tmp/butterfly-answer.jsonl
```

Для полного нового прогона сначала скачайте оригиналы командой `python3 contest/download_originals.py contest/dataset.jsonl /tmp/butterfly-exam-images`, затем примените к этой папке [инструкцию инференса](../README.md). Оценка по скрытому эталону в материалах проекта отсутствует.
