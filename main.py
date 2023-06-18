import streamlit as st
import os 
from fastai.vision.all import *
import pathlib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from surprise import SVD, Dataset, Reader



@st.cache_data
def load_data():
    item_df = pd.read_excel('food_item.xlsx', header=None, names=['item_id', 'food','酸','甜','麻','香','酥','脆','咸','辣','软','多汁','嫩','鲜','蒜味','爽口','焦'])
    item_df.index.name = 'food_id'
    return item_df

def recommends_foods(favorite_food, item_df=load_data(), num_foods=5):
    # Drop unnecessary columns
    item_df_no_titles = item_df.drop(columns=['food'])

    # Compute the cosine similarity between items
    item_similarity = \
        cosine_similarity(item_df_no_titles.drop(columns=['item_id']))

    # Find the item_id of the favorite food
    favorite_food_id = \
        item_df[item_df['food'] == favorite_food]['item_id'].values[0]

    # Find the most similar foods
    most_similar_foods=item_similarity[favorite_food_id - 1].\
            argsort()[-num_foods - 1:-1][::-1]

    # Get the titles of the 3 most similar foods
    similar_food_titles= \
        item_df.loc[most_similar_foods[1:4]]['food']
    
    return similar_food_titles

path = os.path.dirname(os.path.abspath(__file__)) #找到文件夹目录
model_path = os.path.join(path,'export.pkl')  #链接目录与文件
learn_inf = load_learner(model_path)

st.title("津韵美食之旅——基于图像识别的美食推荐系统")

uploaded_file = st.file_uploader("请选择一张天津的美食图片...", type=["jpg", "png", "jpeg"])
pred = None
if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img.to_thumb(500, 500), caption='Your Image')
    pred, pred_idx, probs = learn_inf.predict(img)
    st.write(f'预测: {pred};相似性: {probs[pred_idx]:.04f}')

    st.subheader("推荐的三种食物如下")
    recommended_foods = recommends_foods(pred)
    for i, food in enumerate(recommended_foods):
        st.write(f"{i+1}. {food}")
    st.subheader("接下来请您为我向您推荐的食物进行打分")

#surprise推荐系统开始

# Load the flavor data
flavors_df = pd.read_excel('flavors.xlsx')
flavors_df.columns = ['food_id', 'flavor']

# Function to get flavor for a given food_id
def get_flavor(food_id):
    flavor = flavors_df.loc[flavors_df['food_id'] == food_id, 'flavor'].values
    if len(flavor) > 0:
        return flavor[0]
    else:
        return 'Unknown'

@st.cache_data
def load_data():
    # Load the data
    data_df = pd.read_excel('user_rating.xlsx')
    # Load the foods
    foods_df = pd.read_excel('food.xlsx', header=None)
    foods_df.columns = ['food']
    foods_df.index.name = 'food_id'
    return data_df, foods_df

def train_model(data_df):
    # Create a Reader object
    reader = Reader(rating_scale=(0, 5))

    # Load the data into a Dataset object
    data = Dataset.load_from_df(data_df[['user_id', 'food_id', 'rating']], reader)

    # Build a full trainset from data
    trainset = data.build_full_trainset()

    # Train a SVD model
    algo = SVD()
    algo.fit(trainset)

    return algo

def recommend_foods(algo, data_df, foods_df, new_user_id, new_ratings):
    # Convert ratings from 0-5 scale to 5 to 10 scale
    new_ratings = {food_id: info['rating'] for food_id, info in new_ratings.items()}

    # Add new user's ratings to the data
    new_ratings_df = pd.DataFrame({
    'user_id': [new_user_id]*len(new_ratings),
    'food_id': list(new_ratings.keys()),
    'rating': list(new_ratings.values())
    })

    data_df = pd.concat([data_df, new_ratings_df])

    # Generate recommendations for the new user
    iids = data_df['food_id'].unique() # Get the list of all food ids
    iids_new_user = data_df.loc[data_df['user_id'] == new_user_id, 'food_id'] # Get the list of food ids rated by the new user
    iids_to_pred = np.setdiff1d(iids, iids_new_user) # Get the list of food ids the new user has not rated

    # Predict the ratings for all unrated foods
    testset_new_user = [[new_user_id, iid, 0.] for iid in iids_to_pred]
    predictions = algo.test(testset_new_user)

    # Get the top 5 foods with highest predicted ratings
    top_5_iids = [pred.iid for pred in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]]
    top_5_foods = foods_df.loc[foods_df.index.isin(top_5_iids), 'food']

    return top_5_foods

def main():
    # Load data
    data_df, foods_df = load_data()

    # Choose an unused user_id for the new user
    new_user_id = data_df['user_id'].max() + 1

    if 'initial_ratings' not in st.session_state:
        recommended_foods = recommends_foods(pred)
        a = pd.DataFrame(recommended_foods,columns=['food'])
        a = a.rename_axis('food id')
        random_foods = a     
        st.session_state.initial_ratings = {}
        for food_id, food in zip(random_foods.index, random_foods['food']):
            flavor = get_flavor(food_id) 
            st.session_state.initial_ratings[food_id] = {'food': food,'flavor': flavor,  'rating': 3}

    # Ask user for ratings
    for food_id, info in st.session_state.initial_ratings.items():
        st.write(f"{info['food']} - Flavor: {info['flavor']}")
        info['rating'] = st.slider('给这个食物打分', 0, 5, step=1, value=info['rating'], key=f'init_{food_id}')    # 设置一个滑动条，用户能够拖动滑动条对这3条笑话进行评分

    # 设置一个按钮“Submit Ratings”，用户在点击按钮后，能够生成对该用户推荐的5种食物
    if st.button('提交打分'):
        # Add new user's ratings to the data
        new_ratings_df = pd.DataFrame({
            'user_id': [new_user_id] * len(st.session_state.initial_ratings),
            'food_id': list(st.session_state.initial_ratings.keys()),
            'rating': [info['rating'] for info in st.session_state.initial_ratings.values()]  
        })
        data_df = pd.concat([data_df, new_ratings_df])
        # Train model
        algo = train_model(data_df)

        # Recommend foods based on user's ratings
        recommended_foods = recommend_foods(algo, data_df, foods_df, new_user_id, st.session_state.initial_ratings)

        # Save recommended foods to session state
        st.session_state.recommended_foods = {}
        for food_id, food in zip(recommended_foods.index, recommended_foods):
            st.session_state.recommended_foods[food_id] = {'food': food, 'rating': 3}

   # Display recommended foods and ask for user's ratings
    if 'recommended_foods' in st.session_state:
        st.write('We recommend the following foods based on your ratings:')
        for food_id, info in st.session_state.recommended_foods.items():
            flavor = get_flavor(food_id)  # Get the flavor for the food_id
            st.write(f"{info['food']} - Flavor: {flavor}")
            info['rating'] = st.slider('Rate this food', 0, 5, step=1, value=info['rating'], key=f'rec_{food_id}')
            
        #设置按钮“Submit Recommended Ratings”，点击按钮生成本次推荐的分数percentage_of_total，
        #计算公式为：percentage_of_total = (total_score / 25) * 100。。
        if st.button('Submit Recommended Ratings'):
            # Calculate the percentage of total possible score
            total_score = sum([info['rating'] for info in st.session_state.recommended_foods.values()])
            percentage_of_total = (total_score / 25) * 100
            st.write(f'你在可能总分中所占的百分比: {percentage_of_total}%')

if pred is None:
    st.write("请上传一张图片以获取预测")
if pred is not None:
    main()

    