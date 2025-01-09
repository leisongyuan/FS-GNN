import os
import time
import pickle
from openai import OpenAI
import pandas as pd
import numpy as np
import logging
from scipy.sparse import load_npz

# 配置文件
FILE_PATH = "./aug_data/MovieLens1M/"
LOG_FILE = "augmented_user_attribute.log"

os.makedirs(FILE_PATH, exist_ok=True)

# 设置日志记录
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 构建用户画像的提示信息
def construct_user_prompt(user_id, history_list, item_attributes):
    """
    构建传递给GPT的提示信息，根据用户的历史交互记录生成用户画像。
    
    参数：
    - user_id: 用户的唯一标识
    - history_list: 用户交互过的电影ID列表
    - item_attributes: 包含电影属性信息的DataFrame
    
    返回：
    - prompt: 构建好的提示信息字符串，用于传递给GPT
    """
    pre_string = (
        "You are a knowledgeable assistant. I will provide you with a specific user_id and their interaction history, "
        "which includes a list of item_ids that the user has interacted with. Your task is to find the titles, genres, "
        "directors, countries, and languages associated with these item_ids from the provided item_attributes. Based on this information, "
        "please generate the user's profile with the following attributes:\n\n"
        "1. liked_genre: List the top 3 genres the user is most likely to prefer, separated by a pipe '|'.\n"
        "2. disliked_genre: List the top 2 genres the user is most likely to dislike, separated by a pipe '|'.\n"
        "3. liked_director: Identify the single director the user is most likely to favor.\n"
        "4. country: Identify the most likely country associated with the user's preferences.\n"
        "5. language: Identify the most likely language associated with the user's preferences.\n\n"
        "Please ensure the final output follows this strict format: user_id::liked_genre::disliked_genre::liked_director::country::language\n\n"
        "Example: 1::Drama|Comedy|Action::Horror|Crime::Steven Spielberg::USA::English\n"
    )
    
    # 限制交互电影的数量，防止输入token过长
    limited_history_list = history_list[:30]
    
    # 将用户的历史交互记录格式化为字符串
    history_string = "User's movie history:\n" + "".join(
        f"{item_attributes.loc[item_id, 'title']} ({item_attributes.loc[item_id, 'genre']}) - "
        f"Director: {item_attributes.loc[item_id, 'director']}, Country: {item_attributes.loc[item_id, 'country']}, "
        f"Language: {item_attributes.loc[item_id, 'language']}\n" for item_id in limited_history_list
    )
    
    # 返回完整的提示信息
    return f"User ID: {user_id}\n" + pre_string + history_string


# 处理请求和错误
def handle_request(prompt):
    # 调用OpenAI API获取生成的结果
    client = OpenAI(
        api_key=os.environ.get('sk-fnDLWtODH8hjcW5o1b69A68a8c17474584107f2eB5B183Ca'), 
        base_url="https://tbb.gpt88.top/v1/"
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please give me the answer as requested in system prompt."},
        ],
        model="gpt-4o-mini",
    )

    return response

# 生成用户画像
def request_user_profile(user_id, history_list, item_attributes, user_profiles):
    """
    使用GPT生成用户画像，并将结果存储在user_profiles字典中。
    
    参数：
    - user_id: 用户的唯一标识
    - history_list: 用户交互过的电影ID列表
    - item_attributes: 包含电影属性信息的DataFrame
    - user_profiles: 存储用户画像信息的字典
    """
    if user_id in user_profiles:
        logging.info(f"Profile for user {user_id} already generated, skipping.")
        return

    prompt = construct_user_prompt(user_id, history_list, item_attributes)  # 构建GPT的提示信息
    response = handle_request(prompt)
    if response:
        content = response.choices[0].message.content.strip()
        if content:
            logging.info(f"Response for user {user_id}: {content}")
            elements = content.split("::")
            if len(elements) == 6:  # 检查返回结果的格式是否正确
                user_profiles[user_id] = {
                    'liked_genre': elements[1],
                    'disliked_genre': elements[2],
                    'liked_director': elements[3],
                    'country': elements[4],
                    'language': elements[5]
                }
            else:
                logging.error(f"Unexpected format in response for user {user_id}: {content}")
        else:
            logging.error(f"No content returned for user {user_id} from GPT.")
        # 将更新后的用户画像保存到文件中
        with open(os.path.join(FILE_PATH, 'user_profiles'), 'wb') as f:
            pickle.dump(user_profiles, f)
    else:
        logging.error(f"No response received for user {user_id}.")

# 读取和处理数据
def load_data():
    """
    读取所需的输入数据，包括电影属性文件、用户交互矩阵和之前生成的用户画像。
    
    返回：
    - item_attributes: 包含电影属性信息的DataFrame
    - interaction_matrix: 用户交互矩阵（稀疏矩阵）
    - user_profiles: 存储用户画像信息的字典
    """
    if os.path.exists(os.path.join(FILE_PATH, "user_profiles")):
        with open(os.path.join(FILE_PATH, 'user_profiles'), 'rb') as f:
            user_profiles = pickle.load(f)
    else:
        user_profiles = {}
        with open(os.path.join(FILE_PATH, 'user_profiles'), 'wb') as f:
            pickle.dump(user_profiles, f)
    
    # 读取电影属性文件
    item_attributes = pd.read_csv(os.path.join(FILE_PATH, 'augmented_item_attribute_agg.csv'))
    
    # 读取用户交互矩阵（csv文件）
    try:
        interaction_data = pd.read_csv(os.path.join(FILE_PATH, 'train_matrix.csv'))
        print(f"Loaded interaction matrix with {interaction_data.shape[0]} interactions.")
    except Exception as e:
        print(f"Error loading interaction data: {e}")
        return item_attributes, None, user_profiles
    
    # 将csv文件转换为用户-电影交互矩阵
    num_users = interaction_data['row'].max() + 1
    num_items = interaction_data['col'].max() + 1
    interaction_matrix = np.zeros((num_users, num_items))

    for _, row in interaction_data.iterrows():
        interaction_matrix[int(row['row']), int(row['col'])] = row['data']

    # 检查矩阵的大小
    print("Interaction matrix shape:", interaction_matrix.shape)
    
    return item_attributes, interaction_matrix, user_profiles

# 生成增强的用户属性文件
def generate_augmented_user_attributes(user_attributes, user_profiles):
    """
    将生成的用户画像信息添加到用户属性文件中，并生成增强后的用户属性文件。
    
    参数：
    - user_attributes: 包含用户属性的DataFrame
    - user_profiles: 存储用户画像信息的字典
    """
    liked_genre_list = []
    disliked_genre_list = []
    liked_director_list = []
    country_list = []
    language_list = []

    # 遍历用户属性，匹配用户画像并添加到对应的列中
    for user_id in user_attributes['user_id']:
        if user_id in user_profiles:
            profile = user_profiles[user_id]
            liked_genre_list.append(profile.get('liked_genre', None))
            disliked_genre_list.append(profile.get('disliked_genre', None))
            liked_director_list.append(profile.get('liked_director', None))
            country_list.append(profile.get('country', None))
            language_list.append(profile.get('language', None))
        else:
            liked_genre_list.append(None)
            disliked_genre_list.append(None)
            liked_director_list.append(None)
            country_list.append(None)
            language_list.append(None)

    # 将生成的用户画像信息添加到DataFrame中
    user_attributes['liked_genre'] = pd.Series(liked_genre_list)
    user_attributes['disliked_genre'] = pd.Series(disliked_genre_list)
    user_attributes['liked_director'] = pd.Series(liked_director_list)
    user_attributes['country'] = pd.Series(country_list)
    user_attributes['language'] = pd.Series(language_list)
    
    # 保存增强后的用户属性文件
    user_attributes.to_csv(os.path.join(FILE_PATH, 'augmented_user_attribute_agg.csv'), index=False, header=True)

def main():
    """
    主流程函数，负责执行数据加载、用户画像生成和生成增强的用户属性文件。
    """
    item_attributes, interaction_matrix, user_profiles = load_data()
    user_attributes = pd.read_csv(os.path.join(FILE_PATH, 'user_attribute.csv'))
    
    # 遍历每个用户，得到每个用户的历史交互记录，进而生成对应的用户画像
    for user_id in range(interaction_matrix.shape[0]):

        # 获取用户的评分记录
        user_ratings = interaction_matrix[user_id, :]

        # 选择用户交互过的电影ID
        history_list = np.where(user_ratings > 0)[0] + 1
        
        # 生成对应的用户画像
        request_user_profile(user_id + 1, history_list, item_attributes, user_profiles)
        time.sleep(1)  # 每次请求后等待1秒

    # 生成增强的用户属性文件
    generate_augmented_user_attributes(user_attributes, user_profiles)

if __name__ == "__main__":
    main()