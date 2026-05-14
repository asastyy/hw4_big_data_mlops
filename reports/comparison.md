# Model comparison

| version          |   train_loss |   train_accuracy |   train_f1 |   test_loss |   test_accuracy |   test_f1 |   test_precision |   test_recall |
|:-----------------|-------------:|-----------------:|-----------:|------------:|----------------:|----------:|-----------------:|--------------:|
| base_train1      |     0.134119 |         0.948564 |   0.948641 |    0.126574 |        0.952715 |  0.953391 |         0.941091 |      0.966017 |
| finetuned_train2 |     0.120308 |         0.957718 |   0.95871  |    0.11851  |        0.9595   |  0.959194 |         0.961616 |      0.956784 |

## Delta finetuned - base

|                |       delta |
|:---------------|------------:|
| train_loss     | -0.0138113  |
| train_accuracy |  0.00915429 |
| train_f1       |  0.0100689  |
| test_loss      | -0.00806372 |
| test_accuracy  |  0.00678546 |
| test_f1        |  0.00580308 |
| test_precision |  0.0205256  |
| test_recall    | -0.00923307 |
