# HW4 Big Data: MLOps Pipeline

Решение обучает ResNet18 для классификации AI vs Human images, логирует метрики в TensorBoard, сохраняет модели в локальный S3 MinIO и воспроизводит дообучение через DVC.

## Структура

- `src/hw4_mlops.py` - единый CLI для обучения, оценки, загрузки/скачивания модели из S3 и сравнения версий.
- `params.yaml` - параметры данных, обучения, дообучения и S3.
- `dvc.yaml` - DVC pipeline: базовое обучение, скачивание модели из S3, дообучение, сравнение.
- `docker-compose.yml` - локальный MinIO.
- `HW4_MLOps_Sergeeva.ipynb` - отчетный ноутбук с итоговыми метриками.

## Подготовка

Скопируйте датасет в корень проекта:

```bash
cp -R "/path/to/ai-vs-human-generated-dataset-hw" .
```

Создайте окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запустите MinIO:

```bash
docker compose up -d
```

Если обучение идет на удаленной GPU-машине, MinIO тоже должен быть доступен с этой машины: проще всего запустить `docker compose up -d` там же, либо указать в `S3_ENDPOINT_URL` адрес хранилища, который виден с GPU-сервера.

По умолчанию используются:

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export S3_ENDPOINT_URL=http://localhost:9000
export S3_BUCKET=hw4-models
```

## Запуск базового обучения

```bash
python src/hw4_mlops.py train \
  --experiment-name base_train1 \
  --train-csv ai-vs-human-generated-dataset-hw/Train_1/train.csv \
  --train-root ai-vs-human-generated-dataset-hw/Train_1 \
  --test-csv ai-vs-human-generated-dataset-hw/Test_1/test.csv \
  --test-root ai-vs-human-generated-dataset-hw/Test_1 \
  --output-dir artifacts/base \
  --epochs 5 \
  --batch-size 32 \
  --lr 0.001 \
  --tb-logdir tb_logs \
  --s3-upload-key models/resnet18_base_train1.pt
```

TensorBoard:

```bash
tensorboard --logdir tb_logs
```

## DVC pipeline

Pipeline воспроизводит MLOps-часть: получает базовую модель из S3, дообучает на `Train_2`, оценивает на `Test_2`, загружает новую версию модели в S3 и строит сравнение.

```bash
dvc init
dvc repro
dvc dag
```

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

После дообучения на `Train_2` качество на `Test_2` улучшилось: `test_accuracy` вырос на `0.0068`, `test_f1` вырос на `0.0058`, `test_loss` снизился на `0.0081`. Precision заметно вырос, recall немного снизился, то есть модель стала осторожнее относить объекты к positive-классу.

## Быстрая проверка пайплайна

Для smoke test можно временно добавить в команды обучения:

```bash
--max-train-batches 2 --max-test-batches 2
```

Для финальной сдачи эти ограничения не используются.
