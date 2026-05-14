# HW4 Big Data: MLOps Pipeline

Решение обучает ResNet18 для классификации AI vs Human images, логирует метрики в TensorBoard, сохраняет модели в локальный S3 MinIO и воспроизводит дообучение через DVC

## Структура

- `src/hw4_mlops.py` - единый CLI для обучения, оценки, загрузки/скачивания модели из S3 и сравнения версий
- `mlops_pipeline/params.yaml` - параметры данных, обучения, дообучения и S3
- `mlops_pipeline/dvc.yaml` - DVC pipeline: базовое обучение, скачивание модели из S3, дообучение, сравнение
- `mlops_pipeline/docker-compose.yml` - локальный MinIO
- `HW4_MLOps_Sergeeva.ipynb` - ноутбук с итоговыми метриками.

## DVC pipeline

Результаты:

- `artifacts/base/metrics.json`
- `artifacts/finetuned/metrics.json`
- `reports/comparison.md`
- `reports/comparison.csv`
- TensorBoard логи в `tb_logs/`
- модели в MinIO bucket `hw4-models`

## Полученные результаты

Итоговые артефакты были получены на GPU-машине с NVIDIA H100. PyTorch использовался с CUDA 12.8.

S3/MinIO:

- `s3://hw4-models/models/resnet18_base_train1.pt`
- `s3://hw4-models/models/resnet18_finetuned_train2.pt`

Финальное сравнение моделей:

| version | train_loss | train_accuracy | train_f1 | test_loss | test_accuracy | test_f1 | test_precision | test_recall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| base_train1 | 0.134119 | 0.948564 | 0.948641 | 0.126574 | 0.952715 | 0.953391 | 0.941091 | 0.966017 |
| finetuned_train2 | 0.120308 | 0.957718 | 0.958710 | 0.118510 | 0.959500 | 0.959194 | 0.961616 | 0.956784 |

После дообучения на `Train_2` качество на `Test_2` улучшилось: `test_accuracy` вырос на `0.0068`, `test_f1` вырос на `0.0058`, `test_loss` снизился на `0.0081`. Precision заметно вырос, recall немного снизился, то есть модель стала осторожнее относить объекты к positive-классу
