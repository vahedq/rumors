# Rumor has it!
In 2011, I worked on a project to identify fake news and misinformation on social media ([See this paper for more details](https://www.aclweb.org/anthology/D11-1147)).

## To run
`python main.py  --task sentiment --method dense`

`task` should be one of `[sentiment/detection]` and method is either of `[nb|sgd|dense|bilstm|all]`


## Data
If you use the data, please cite the following paper:

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
