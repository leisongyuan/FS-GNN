import os
import time
import pickle
import openai
import pandas as pd
import logging

# 配置文件
API_KEY = 'xxxx'  # 请输入API_KEY
FILE_PATH = "./aug_data/MovieLens1M/"
LOG_FILE = "augmented_item_attribute.log"

openai.api_key = API_KEY
openai.api_base = "xxxx" # 请输入API_BASE

os.makedirs(FILE_PATH, exist_ok=True)

# 设置日志记录
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 构建提示信息
def construct_prompt(item_attributes, indices):
    pre_string = (
        "You are a knowledgeable assistant. The first three columns (id, title, genre) "
        "are provided from my CSV file. Your task is to provide the remaining three columns "
        "(director, country, language) based on the given information. "
        "Make sure to follow the format strictly: "
        "id::title::genre::director::country::language\n"
    )
    item_list_string = "".join(f"{item_attributes['id'][index]}, {item_attributes['title'][index]}, {item_attributes['genre'][index]}\n" for index in indices)
    example_string = (
        "\nExample:\n"
        "1::Toy Story (1995)::Animation|Children's|Comedy::Francis Ford Coppola::USA::English\n"
    )
    return pre_string + item_list_string + example_string


# 处理请求和错误
def handle_request(params, retry_func, *args):
    retry_limit = 3
    for attempt in range(retry_limit):
        try:
            response = openai.ChatCompletion.create(**params)
            return response
        except openai.error.InvalidRequestError as e:
            logging.error(f"InvalidRequestError: {str(e)}")
        except openai.error.RateLimitError as e:
            logging.warning(f"RateLimitError: {str(e)}. Waiting for 60 seconds before retrying...")
            time.sleep(60)
        except openai.error.AuthenticationError as e:
            logging.error(f"AuthenticationError: {str(e)}")
        except openai.error.APIConnectionError as e:
            logging.error(f"APIConnectionError: {str(e)}")
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAIError: {str(e)}")
        except Exception as e:
            logging.error(f"UnknownError: {str(e)}")
            time.sleep(5)
        if attempt < retry_limit - 1:
            time.sleep(5)  # 等待时间可调
    raise RuntimeError("Failed to get a valid response after several attempts.")

# ChatGPT属性生成
def request_attributes(item_attributes, indices, augmented_dict):
    if indices[0] in augmented_dict and all(v is not None for v in augmented_dict[indices[0]].values()):
        logging.info(f"Index {indices[0]} already processed with complete data.")
        return
    prompt = construct_prompt(item_attributes, indices)
    params = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.6,
    }
    response = handle_request(params, request_attributes, item_attributes, indices, augmented_dict)
    if response:
        content = response['choices'][0]['message']['content'].strip()
        if content:
            logging.info(f"Response for index {indices[0]}: {content}")
            elements = content.split("::")
            # 通过 indices 确保对齐
            if len(elements) == 6:
                id, title, genre, director, country, language = elements
                # 使用 indices 作为键
                augmented_dict[indices[0]] = {0: id, 1: title, 2: genre, 3: director, 4: country, 5: language}
            else:
                logging.error(f"Unexpected format in response for index {indices[0]}: {content}")
        else:
            logging.error(f"No content returned for index {indices[0]} from GPT.")
        with open(os.path.join(FILE_PATH, 'augmented_attribute_dict'), 'wb') as f:
            pickle.dump(augmented_dict, f)
    else:
        logging.error(f"No response received for index {indices[0]}.")

# 读取和处理数据
def load_data():
    if os.path.exists(os.path.join(FILE_PATH, "augmented_attribute_dict")):
        with open(os.path.join(FILE_PATH, 'augmented_attribute_dict'), 'rb') as f:
            augmented_attribute_dict = pickle.load(f)
    else:
        augmented_attribute_dict = {}
        with open(os.path.join(FILE_PATH, 'augmented_attribute_dict'), 'wb') as f:
            pickle.dump(augmented_attribute_dict, f)
    item_attributes = pd.read_csv(os.path.join(FILE_PATH, 'item_attribute.csv'), names=['id', 'title', 'genre'], skiprows=1)
    return item_attributes, augmented_attribute_dict

def check_and_retry_missing_data(augmented_dict, item_attributes):
    # 检查哪些行缺失数据
    missing_indices = [i for i, entry in augmented_dict.items() if any(v is None for v in entry.values())]
    
    # 重新运行缺失数据的请求
    for i in missing_indices:
        indices = [i]
        logging.info(f"Retrying request for missing data at index {i}")
        request_attributes(item_attributes, indices, augmented_dict)
        time.sleep(1)  # 每次请求后等待1秒

def main():
    item_attributes, augmented_dict = load_data()

    # 主要的数据处理过程
    for i in range(item_attributes.shape[0]):
        indices = [i]
        request_attributes(item_attributes, indices, augmented_dict)
        time.sleep(1)  # 每次请求后等待1秒

    # 检查并重试缺失的数据
    check_and_retry_missing_data(augmented_dict, item_attributes)

    # 生成新的CSV文件
    raw_item_attributes = pd.read_csv(os.path.join(FILE_PATH, 'item_attribute.csv'), names=['id', 'title', 'genre'], skiprows=1)
    director_list = []
    country_list = []
    language_list = []

    for i in range(raw_item_attributes.shape[0]):
        if i in augmented_dict:
            entry = augmented_dict[i]
            director_list.append(entry.get(3, None))
            country_list.append(entry.get(4, None))
            language_list.append(entry.get(5, None))
        else:
            director_list.append(None)
            country_list.append(None)
            language_list.append(None)

    raw_item_attributes['director'] = pd.Series(director_list)
    raw_item_attributes['country'] = pd.Series(country_list)
    raw_item_attributes['language'] = pd.Series(language_list)
    raw_item_attributes.to_csv(os.path.join(FILE_PATH, 'augmented_item_attribute_agg.csv'), index=False, header=True)

if __name__ == "__main__":
    main()
