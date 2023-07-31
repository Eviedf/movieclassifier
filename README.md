## Multi-level Movie Genre classification with IMDB data

This repository contains two files: 
1. movie_classification.ipynb:  extract and explore the data from the OMDB api, and create a model for multi-level classification. Run it with your own OMDB API key
2. deploy.py: set up an API end-point to build an inference function

In order to run deploy.py, follow the following steps:
2. Make sure that the model and multilabel binarizer are stored in the relative directory models/
3. Run the API with 'uvicorn deploy:app --reload'
4. Navigate to http://127.0.0.1:8000/docs in your browser and then you will see the documentation page created automatically by FastAPI
5. Try it out!

#Two examples from the test dataset:
Shrek
When a green ogre named Shrek discovers his swamp has been 'swamped' with all sorts of fairytale creatures by the scheming Lord Farquaad, 
Shrek sets out with a very loud donkey by his side to 'persuade' Farquaad to give Shrek his swamp back. Instead, a deal is made. 
Farquaad, who wants to become the King, sends Shrek to rescue Princess Fiona, who is awaiting her true love in a tower guarded by a fire-breathing dragon. 
But once they head back with Fiona, it starts to become apparent that not only does Shrek, an ugly ogre, begin to fall in love with the lovely princess, 
but Fiona is also hiding a huge secret.

Enter your data here:
![image](https://github.com/Eviedf/movieclassifier/assets/25794934/08839dde-0aef-4025-83df-58b3f9d6ff80)

Find the output below: 
![image](https://github.com/Eviedf/movieclassifier/assets/25794934/4d052e11-a405-4ad3-af26-7215752d2368)


Hercules
Hercules, son of the Greek God, Zeus, is turned into a half-god, half-mortal by evil Hades, God of the Underworld, who plans to overthrow Zeus.
Hercules is raised on Earth and retains his god-like strength, but when he discovers his immortal heritage Zeus tells him that to return to Mount Olympus he must become a true hero.
Hercules becomes a famous hero with the help of his friend Pegasus and his personal trainer, Phil the satyr. Hercules battles monsters, Hades and the Titans, 
but it is his self-sacrifice to rescue his love Meg which makes him a true hero.

Enter your data here:
![image](https://github.com/Eviedf/movieclassifier/assets/25794934/9baabccc-a2bf-49ae-acdc-fd293cd89143)

Find the output below:
![image](https://github.com/Eviedf/movieclassifier/assets/25794934/4bc6ef6c-4211-4298-b67b-7bfd63681dd2)


