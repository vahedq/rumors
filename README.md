# Rumor has it!
In 2011, I worked on a project to identify fake news and misinformation on social media ([See this paper for more details](https://www.aclweb.org/anthology/D11-1147)). This repository contains the dataset built and used in that work as well as some fun additional experiments.

## To run
`python main.py  --task sentiment --method dense`

`task` should be one of `[sentiment/detection]` and method is either of `[nb|sgd|dense|bilstm|all]`

## Rumor Sentiment Experiments


```
=========== NBModel ===========
Classification Report:
              precision    recall  f1-score   support

     endorse       0.75      0.88      0.81       416
        deny       0.79      0.93      0.86       375
    question       1.00      0.16      0.27       139
     neutral       0.00      0.00      0.00        21

    accuracy                           0.77       951
   macro avg       0.63      0.49      0.48       951
weighted avg       0.79      0.77      0.73       951


Confusion Matrix:
          endorse  deny  question  neutral
endorse       366    50         0        0
deny           27   348         0        0
question       87    30        22        0
neutral        11    10         0        0


=========== SGDModel ===========
Classification Report:
              precision    recall  f1-score   support

     endorse       0.81      0.84      0.82       416
        deny       0.87      0.90      0.88       375
    question       0.67      0.59      0.63       139
     neutral       0.86      0.29      0.43        21

    accuracy                           0.81       951
   macro avg       0.80      0.65      0.69       951
weighted avg       0.81      0.81      0.81       951


Confusion Matrix:
          endorse  deny  question  neutral
endorse       349    39        28        0
deny           29   338         7        1
question       47    10        82        0
neutral         7     3         5        6


=========== SimpleDense ===========
Classification Report:
              precision    recall  f1-score   support

     endorse       0.80      0.83      0.81       416
        deny       0.86      0.89      0.87       375
    question       0.68      0.60      0.64       139
     neutral       0.78      0.33      0.47        21

    accuracy                           0.81       951
   macro avg       0.78      0.66      0.70       951
weighted avg       0.80      0.81      0.80       951


Confusion Matrix:
          endorse  deny  question  neutral
endorse       345    37        34        0
deny           37   332         4        2
question       43    12        84        0
neutral         7     5         2        7


=========== BiLSTM/Glove ===========
Classification Report:
              precision    recall  f1-score   support

     endorse       0.81      0.83      0.82       416
        deny       0.85      0.91      0.88       375
    question       0.69      0.55      0.61       139
     neutral       0.64      0.33      0.44        21

    accuracy                           0.81       951
   macro avg       0.75      0.66      0.69       951
weighted avg       0.81      0.81      0.81       951


Confusion Matrix:
          endorse  deny  question  neutral
endorse       347    38        28        3
deny           26   342         6        1
question       48    14        77        0
neutral         6     7         1        7
```


## Data
You can find the dataset and its readme under `data/`. If you use the data, please cite the following paper:

```
@InProceedings{qazvinian-EtAl:2011:EMNLP,
  author    = {Qazvinian, Vahed  and  Rosengren, Emily  and  Radev, Dragomir R.  and  Mei, Qiaozhu},
  title     = {Rumor has it: Identifying Misinformation in Microblogs},
  booktitle = {Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing},
  month     = {July},
  year      = {2011},
  address   = {Edinburgh, Scotland, UK.},
  publisher = {Association for Computational Linguistics},
  pages     = {1589--1599},
}
```
