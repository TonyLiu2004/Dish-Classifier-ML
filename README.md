[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![GitHub Repo][github-shield]][github-repo-url]

[![Python][Python]][Python-url]
[![Streamlit][Streamlit]][Streamlit-url]
[![Tensorflow][Tensorflow]][Tensorflow-url]
[![LangChain][LangChain]][LangChain-url]
[![OpenAI ChatGPT][OpenAI ChatGPT]][OpenAI ChatGPT-url]
[![Kaggle][Kaggle]][Kaggle-url]
[![MongoDB][MongoDB]][MongoDB-url]

# Food Chain
Welcome to Food Chain, a top of the line food classifier that everyday people can use to figure out what they are eating and the common values associated with it such as recipe, nutrients, and even the origin! Please tell us how you liked our app and thank you for using it!


## Who are we?
* [Tony Liu](https://tonyliu2004.github.io/)
    * Tony Liu is a junior at Hunter College.

* [Nicklaus Yao](https://www.linkedin.com/in/nicklausyao/)
    * Nicklaus Yao is a senior at Hunter College.

* [David Rodriguez](https://drod75.github.io/)
    * David Rodriguez is a junior at Brooklyn College studying computer science with a minor in data science, 
    he spends his time working on projects and participating in his college community. 


## Purpose
The project aims to use machine learning and computer vision techniques to analyze food images and identify them. By using diverse datasets, the model will learn to recognize dishes based on visual features. Our project aims to inform users about what it is they are eating, including potential nutritional value and an AI generated response on how their dish might have been prepared. We want users to have an easy way to figure out what their favorite foods contain, to know any allergens in the food and to better connect to the food around them. This tool can also tell users the calories of their dish, they can figure out the nutrients with only a few steps!


## Key Features
* Image classification
* AI generated nutritional information
* AI generated recipe used for the dish
* Historical Data about the dish such as origin and where you can get it!


## How we did it
We used a combination of machine learning and computer vision techniques to analyze food images and identify them, we used the [EfficientNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet8) model to classify the images and train it in order to get the results we needed. Our data was from the [Food Recognition](https://www.kaggle.com/datasets/sainikhileshreddy/food-recognition-2022/data) Database on Kaggle. We also used [LangChain](https://www.langchain.com/) to generate the recipe and [MongoDB](https://www.mongodb.com/) to store the data for the RAG we used for the recipe generation.
After this we then built our app using [Streamlit](https://www.streamlit.io/) and implemented everything we used so far in order to develop an app that can classify images and generate recipes based on the image, or even just with the user's query of a recipe.


## Demo


## Tutorial
To use our site please follow the instructions below:
1. Go to the site via this url: https://github.com/TonyLiu2004/Dish-Classifier-ML
2. Click on the "Upload Image" button
3. Wait for the image to be processed
4. Done!


[contributors-shield]: https://img.shields.io/github/contributors/TonyLiu2004/Dish-Classifier-ML.svg?style=for-the-badge
[contributors-url]: https://github.com/TonyLiu2004/Dish-Classifier-ML/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/GeorgiosIoannouCoder/realesrgan.svg?style=for-the-badge
[stars-url]: https://github.com/TonyLiu2004/Dish-Classifier-ML/stargazers
[github-repo-url]: https://github.com/TonyLiu2004/Dish-Classifier-ML
[github-shield]: https://img.shields.io/badge/-GitHub-black.svg?style=for-the-badge&logo=github&colorB=000
[Python]: https://img.shields.io/badge/python-FFDE57?style=for-the-badge&logo=python&logoColor=4584B6
[Python-url]: https://www.python.org/
[Streamlit]: https://img.shields.io/badge/streamlit-ffffff?style=for-the-badge&logo=streamlit&logoColor=ff0000
[Streamlit-url]: https://streamlit.io/
[Tensorflow]: https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white 
[Tensorflow-url]: https://www.tensorflow.org/
[LangChain]: https://img.shields.io/badge/langchain-007FFF?style=for-the-badge&logo=langchain&logoColor=white
[LangChain-url]: https://www.langchain.com/
[OpenAI ChatGPT]: https://img.shields.io/badge/OpenAI-FFD000?style=for-the-badge&logo=OpenAI&logoColor=white
[OpenAI ChatGPT-url]: https://chat.openai.com/
[Kaggle]: https://img.shields.io/badge/kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white
[Kaggle-url]: https://www.kaggle.com/
[MongoDB]: https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white
[MongoDB-url]: https://www.mongodb.com/