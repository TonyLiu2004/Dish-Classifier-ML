# Food Classification
### **Context**: 
A project made for the CUNY Tech Prep Fall 2024 Cohort 10 Data Science Track, which right now aims to classify American based food.

### **Task**: 
Our objective is to develop a machine learning model that accurately classifies various dishes from images and outputs their names along with a confidence score.

### **Concept**: 
The project aims to use machine learning and computer vision techniques to analyze food images and identify them. By using diverse datasets, the model will learn to recognize dishes based on visual features.

### **Minimum Viable Product**: 
The Minimum Viable Product for this project will be a functional prototype of the food classification model that can:
 1. Accept Input Images: Users can upload images of dishes.
 2. Provide Classification Results: The model will return the name of the dish and a confidence score.
 3.  User Interface: A simple web interface where users can interact with the model.

### **Further Goals**
 1. Model returns dish ingredients
 2. Identify cuisine
 3. Estimate calorie
 4. Provide recipe
 5. Suggest dishes based on a user prompt of ingredients
   * Identify general ingredients from an image and suggest a dish that the user can make with the existing ingredients



### **Challenges**:
 1. Data Quality:
   * Challenge:
     * Obtaining accurate pictures of food and or making sure the AI model will be able to detect the same dish from different pictures.
   * Possible Solution:
     * Making sure the AI can detect key features and compare them accurately, while at the same time training the AI on various pictures of the same type of food.
     * Obtaining pictures with the same food but different lighting, background, and color filter (unsure of results).
 2. Model Accuracy:
   * Challenge:
     * Making sure the AI model does not confuse an apple for an orange, since they are both fruit and round and usually a bright color. Another example is two dishes of soup from different regions and cultures.
   * Possible Solution:
     * Pattern identification and texture is a key, knowing the common ingredient in a specific culture or region and how that affects the dish overall.
     * Showing the model foods from the same culture together, and how certain dishes are related to each other (risk).
 3. Labeling - Overfitting/Underfitting :
   * Challenge:
     * Different cultures may have unique names and variations for the same dish, complicating the labeling process.
   * Possible Solutions:
     * Provide other dish names within a certain accuracy margin.
     * Use AI to modify and generalize the input (risky)
     * Use AI to generate alternate dish names from output 



### **Risks**:
 1. Ethics:  Accessibility
   * Mitigation: Making sure the project takes in mind disabilities and how different people may  not be able to use certain features effectively.
 2. Misidentification: Complex dishes
   * Mitigation: Intensive datasets and context perfish trained on, ingredients and even how to where it was cooked could be needed.
 3. Possible AI Usage: Over usage and reliance
   * Mitigation: Users may rely on the AI too much and take its word as 100 percent accurate, telling users to always ask a person if possible is a great option.
 4. Data Privacy: Collecting possible user-uploaded images may raise privacy concerns.
   * Mitigation: Implement a privacy policy and ensure that user data stays anonymous, or obtain user consent for data usage.




### **Implementation**
 1. Architecture:
   * Github Repo that divides the AI training model to the application interface and compatibility. Folder structure and coding structure will be taken into account.
 2. Programming Languages:
   * Python, Html
 3. APIs:
   * Pandas, Numpy, Tensorflow, streamlit(maybe)
 4. Database:
   * (Pending Could Change) Heroku, Mongodb Atlus
 5. Deployment:
   * streamlit, heroku, aws, render




### Credits:
 * [David Rodriguez](https://drod75.github.io/)
 * [Nicklaus Yao](https://github.com/NickYaoo)
 * [Tony Liu](https://tonyliu2004.github.io/)
