This dataset includes 5 sets of annotated tweets from 5 different stories: 

airfrance.txt:	Air France mid-air crash photos?
michelle.txt:	Michelle Obama hired too many staff?
palin.txt:	palin Sarah Palin getting divorced?
cell-ids.txt:	Cell phone numbers going public?
obama-ids.txt:	Is Barack Obama muslim? 


Each tweet in each dataset is annotated usign the following guidelines:

0 -> if the tweet is not about the rumor
11 -> if the tweet endorses the rumor
12 -> if the tweet denies the rumor
13 -> if the tweet questions the rumor
14 -> if the tweet is neutral
2 -> if the annotator is undetermined


Three of the datasets (airfrance.txt, michelle.txt, and palin.txt) have 4 columns with the following format:
<date> <userid> <tweet> <label>

The other 2, (obama-ids.txt and cell-ids.txt) have 2 columns with the following format:
<tweet id> <label>


Please cite this paper when you use this data:

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

