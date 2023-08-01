## Multi-level Movie Genre classification with IMDB data

This repository contains two files: 
1. movie_classification.ipynb:  extract and explore the data from the OMDB api, and create a model for multi-level classification. Run it with your own OMDB API key
2. deploy.py: set up an API end-point to build an inference function

In order to run deploy.py, follow the following steps:
1. Make sure that the model and multilabel binarizer are stored in the relative directory models/
2. Run the API with 'uvicorn deploy:app --reload'
3. Navigate to http://127.0.0.1:8000/docs in your browser and then you will see the documentation page created automatically by FastAPI
4. Try it out!

## Two examples from the test dataset:

**Shrek**

_When a green ogre named Shrek discovers his swamp has been 'swamped' with all sorts of fairytale creatures by the scheming Lord Farquaad, 
Shrek sets out with a very loud donkey by his side to 'persuade' Farquaad to give Shrek his swamp back. Instead, a deal is made. 
Farquaad, who wants to become the King, sends Shrek to rescue Princess Fiona, who is awaiting her true love in a tower guarded by a fire-breathing dragon. 
But once they head back with Fiona, it starts to become apparent that not only does Shrek, an ugly ogre, begin to fall in love with the lovely princess, 
but Fiona is also hiding a huge secret._ 

True genres: Animation, Adventure, Comedy

Enter your data here:
![image](https://github.com/Eviedf/movieclassifier/assets/25794934/4fd46f69-47da-4bd0-99b9-f7eafa0678e8)


Find the output below: 
![image](https://github.com/Eviedf/movieclassifier/assets/25794934/8340e0bd-f412-4057-880c-824ef6539764)


**Grave of the Fireflies**

_The story of Seita and Setsuko, two young Japanese siblings, living in the declining days of World War II. When an American firebombing separates 
the two children from their parents, the two siblings must rely completely on one another while they struggle to fight for their survival._ 

True genres: Animation, Drama, War

Enter your data here:
![image](https://github.com/Eviedf/movieclassifier/assets/25794934/d1ac46a4-b4ef-4ad2-ab0f-d32554d325f7)

Find the output below:
![image](https://github.com/Eviedf/movieclassifier/assets/25794934/120e6bd1-bcb2-4e9d-973d-18899c4e66ba)




