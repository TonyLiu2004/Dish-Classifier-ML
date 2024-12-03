import numpy as np 
import pandas as pd  # type: ignore
import os
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import pymongo
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import json
from streamlit_extras.bottom_container import bottom

st.set_page_config(
    page_title="Food Chain", 
    page_icon="🍴"
)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_URI_RAG_RECIPE")

@st.cache_resource
def loadEmbedding():
    embedding = SentenceTransformer("thenlper/gte-large") 
    return embedding
embedding = loadEmbedding()


def getEmbedding(text):
    if not text.strip():
        print("Text was empty")
        return []
    encoded = embedding.encode(text)
    return encoded.tolist()


# Connect to MongoDB
def get_mongo_client(mongo_uri):
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

if not mongo_uri:
    print("MONGO_URI not set in env")

mongo_client = get_mongo_client(mongo_uri)

mongo_db = mongo_client['recipes']
mongo_collection = mongo_db['recipesCollection']

def vector_search(user_query, collection):
    query_embedding = getEmbedding(user_query)
    if query_embedding is None:
        return "Invalid query or embedding gen failed"
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 4  # Return top 4 matches
        }
    }

    unset_stage = {
        "$unset": "embedding"  # Exclude the 'embedding' field from the results
    }

    project_stage = {
        "$project": {
            "_id": 0,  # Exclude the _id field
            "name": 1,
            "minutes": 1,
            "tags": 1,
            "n_steps": 1,
            "description": 1,
            "ingredients": 1,
            "n_ingredients": 1,
            "formatted_nutrition": 1,
            "formatted_steps": 1,
            "score": {
                "$meta": "vectorSearchScore"  # Include the search score
            }
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]
    results = mongo_collection.aggregate(pipeline)
    return list(results)

def mongo_retriever(query):
    documents = vector_search(query, mongo_collection)
    return documents


template = """
You are an assistant for generating results based on user questions.
Use the provided context to generate a result based on the following JSON format:
{{
  "name": "Recipe Name",
  "minutes": 0,
  "tags": [
    "tag1",
    "tag2",
    "tag3"
  ],
  "n_steps": 0,
  "description": "A GENERAL description of the recipe goes here.",
  "ingredients": [
    "ingredient1",
    "ingredient2",
    "ingredient3"
  ],
  "n_ingredients": 0,
  "formatted_nutrition": [
    "Calorie : per serving",
    "Total Fat : % daily value",
    "Sugar : % daily value",
    "Sodium : % daily value",
    "Protein : % daily value",
    "Saturated Fat : % daily value",
    "Total Carbohydrate : % daily value"
  ],
  "formatted_steps": [
    "1. Step 1 of the recipe.",
    "2. Step 2 of the recipe.",
    "3. Step 3 of the recipe."
  ]
}}
Instructions:
1. Focus on the user's specific request and avoid irrelevant ingredients or approaches.
2. Do not return anything other than the JSON.
3. If the answer is unclear or the context does not fully address the prompt, return []
4. Base the response on simple, healthy, and accessible ingredients and techniques.
5. Rewrite the description in third person
Context: {context}
When choosing a recipe from the context, FOLLOW these instructions:
1. The recipe should be makeable from scratch, using only proper ingredients and not other dishes or pre-made recipes
Question: {question}
"""

custom_rag_prompt = ChatPromptTemplate.from_template(template)


llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2)


rag_chain = (
    {"context": mongo_retriever,  "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

def get_response(query):
    return rag_chain.invoke(query)


##############################################
# Classifier
img_size = 224

@st.cache_resource
def loadModel():
    model = load_model('models/efficientnet-fine-d1.keras')
    return model

model = loadModel()


class_names = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", 
    "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", 
    "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheese_plate", "cheesecake", "chicken_curry", 
    "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", 
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", 
    "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", 
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", 
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", 
    "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", 
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels", 
    "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", 
    "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", 
    "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", 
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", 
    "tuna_tartare", "waffles"
]

def classifyImage(input_image):
    input_image = input_image.resize((img_size, img_size))
    input_array = tf.keras.utils.img_to_array(input_image)

    # Add a batch dimension 
    input_array = tf.expand_dims(input_array, 0)  # (1, 224, 224, 3)
    
    predictions = model.predict(input_array)[0]
    print(f"Predictions: {predictions}")

    # Sort predictions to get top 5
    top_indices = np.argsort(predictions)[-5:][::-1]
    
    # Prepare the top 5 predictions with their class names and percentages
    top_predictions = [(class_names[i], predictions[i] * 100) for i in top_indices]
    for i, (class_name, confidence) in enumerate(top_predictions, 1):
        print(f"{i}. Predicted {class_name} with {confidence:.2f}% Confidence")

    return top_predictions

def capitalize_after_number(input_string):
    # Split the string on the first period
    if ". " in input_string:
        num, text = input_string.split(". ", 1)
        return f"{num}. {text.capitalize()}"
    return input_string
##############################################

#for displaying RAG recipe response
def display_response(response):
    """
    Function to format a JSON response into Streamlit's `st.write()` format.
    """
    if isinstance(response, str):
        # Convert JSON string to dictionary if necessary
        response = json.loads(response)
    
    st.write("### Recipe Details")
    st.write(f"**Name:** {response['name'].capitalize()}")
    st.write(f"**Preparation Time:** {response['minutes']} minutes")
    st.write(f"**Description:** {response['description'].capitalize()}")
    st.write(f"**Tags:** {', '.join(response['tags'])}")
    st.write("### Ingredients")
    st.write(", ".join([ingredient.capitalize() for ingredient in response['ingredients']]))
    st.write(f"**Total Ingredients:** {response['n_ingredients']}")
    st.write("### Nutrition Information (per serving)")
    st.write(", ".join(response['formatted_nutrition']))
    st.write(f"**Number of Steps:** {response['n_steps']}")
    st.write("### Steps")
    for step in response['formatted_steps']:
        st.write(capitalize_after_number(step))

def display_dishes_in_grid(dishes, cols=3):
    rows = len(dishes) // cols + int(len(dishes) % cols > 0)
    for i in range(rows):
        cols_data = dishes[i*cols:(i+1)*cols]
        cols_list = st.columns(len(cols_data))
        for col, dish in zip(cols_list, cols_data):
            with col:
                st.sidebar.write(dish.replace("_", " ").capitalize())
# #Streamlit

#Left sidebar title
st.sidebar.markdown(
    "<h1 style='font-size:32px;'>RAG Recipe</h1>", 
    unsafe_allow_html=True
)

st.sidebar.write("Upload an image and/or enter a query to get started! Explore our trained dish types listed below for guidance.")

uploaded_image = st.sidebar.file_uploader("Choose an image:", type="jpg")
query = st.sidebar.text_area("Enter your query:", height=100)

# gap
st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
selected_dish = st.sidebar.selectbox(
    "Search for a dish that our model can classify:", 
    options=class_names,
    index=0  
)

# Right title
st.title("Welcome to FOOD CHAIN!")
with st.expander("**What is FOOD CHAIN?**"):
    st.markdown(
        """
        The project aims to use machine learning and computer vision techniques to analyze food images 
        and identify them. By using diverse datasets, the model will learn to recognize dishes based on 
        visual features. Our project aims to inform users about what it is they are eating, including 
        potential nutritional value and an AI generated response on how their dish might have been prepared. 
        We want users to have an easy way to figure out what their favorite foods contain, to know any 
        allergens in the food and to better connect to the food around them. This tool can also tell users 
        the calories of their dish, they can figure out the nutrients with only a few steps!

        Thank you for using our project!
        """
    )

# bottom
with bottom():
    st.markdown("Made by the Classify Crew: [Contact List](https://linktr.ee/classifycrew)")


#################


# Image Classification Section
if uploaded_image and query:
    with st.expander("**Food Classification**", expanded=True, icon=':material/search_insights:'):
        st.title("Results: Image Classification")

        # Open the image
        input_image = Image.open(uploaded_image)

        # Display the image
        st.image(input_image, caption="Uploaded Image.", use_container_width=True)
            
        predictions = classifyImage(input_image)
        fpredictions = ""

        # Show the top predictions with percentages
        st.markdown("**Top Predictions:**")
        for class_name, confidence in predictions:
            if int(confidence) > 0.05:
                fpredictions += f"{class_name}: {confidence:.2f}%,"
            class_name = class_name.replace("_", " ")
            class_name = class_name.title()
            st.markdown(f"**{class_name}**: {confidence:.2f}%")
        print(fpredictions)

    # call openai to pick the best classification result based on query
    openAICall = [
        SystemMessage(
            content = "You are a helpful assistant that identifies the best match between classified food items and a user's request based on provided classifications and keywords."
        ),
        HumanMessage( 
            content = f"""
                Based on the following image classification with percentages of each food:
                {fpredictions}
                And the following user request:
                {query}
                Return to me JUST ONE of the classified images that most relates to the user request, based on the relevance of the user query.
                in the format: [dish]
            """
        ),
    ]

    # Call the OpenAI API
    openAIresponse = llm.invoke(openAICall)
    print("AI CALL RESPONSE: ", openAIresponse.content)

    # RAG the openai response and display
    print("RAG INPUT", openAIresponse.content + " " + query)

    with st.expander("Recipe Generation", expanded=True, icon=':material/menu_book:'):
        st.title('Results: RAG')
        RAGresponse = get_response(openAIresponse.content + " " + query)
        display_response(RAGresponse)
elif uploaded_image is not None:
    with st.expander("**Food Classification**", expanded=True, icon=':material/search_insights:'):
        st.title("Results: Image Classification")

        # Open the image
        input_image = Image.open(uploaded_image)

        # Display the image
        st.image(input_image, caption="Uploaded Image.", use_container_width=True)

        # Classify the image and display the result
        predictions = classifyImage(input_image)
        fpredictions = ""

        # Show the top predictions with percentages
        st.markdown("**Top Predictions:**")
        for class_name, confidence in predictions:
            if int(confidence) > 0.05:
                fpredictions += f"{class_name}: {confidence:.2f}%,"
            if int(confidence) > 5:
                class_name = class_name.replace("_", " ")
                class_name = class_name.title()
                st.markdown(f"**{class_name}**: {confidence:.2f}%")
        print(fpredictions)

elif query:
    with st.expander("**Recipe Generation**", expanded=True, icon=':material/menu_book:'):
        st.title("Results: RAG")
        response = get_response(query)
        print(response,'\n\n\n\n\n\n\n')
        display_response(response)
else:
    st.warning("Please input an image and/or a prompt.", icon=':material/no_meals:')

