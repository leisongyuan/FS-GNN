import openai
import pandas as pd
import numpy as np
import os
import pickle
import requests
import time
import logging
from sklearn.decomposition import PCA

# 请输入 API key 和 API base
API_KEY = 'xxxx'
API_BASE = "xxxx"
FILE_PATH = "./aug_data/MovieLens1M/"
EMB_DIM = 6

# 配置日志记录
logging.basicConfig(filename='get_item_embedding.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置 OpenAI API
openai.api_key = API_KEY
openai.api_base = API_BASE

# 获取文本嵌入的函数
def get_embedding(text):
    url = f"{API_BASE}/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    params = {
        "model": "text-embedding-ada-002",
        "input": text
    }
    logging.info(f"Requesting embedding for text: {text}")
    response = requests.post(url=url, headers=headers, json=params)
    response.raise_for_status()  # 如果请求失败，抛出异常
    message = response.json()
    logging.info(f"Received embedding for text: {text}")
    
    # 获取嵌入并检查特殊值
    embedding = message['data'][0]['embedding']
    embedding = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in embedding]
    
    return embedding


# 请求嵌入并更新字典
def LLM_request(toy_augmented_item_attribute, indices, augmented_atttribute_embedding_dict):
    if not indices:
        logging.info("No indices provided, skipping request.")
        return

    for key in augmented_atttribute_embedding_dict.keys():
        if indices[0] in augmented_atttribute_embedding_dict[key]:
            logging.info(f"Index {indices[0]} already in {key} embedding dictionary, skipping.")
            continue
        
        # 处理nan值
        text = toy_augmented_item_attribute[key][indices].values[0]
        if pd.isna(text):
            logging.warning(f"Text for key '{key}', index {indices[0]} is NaN. Using default embedding.")
            default_embedding = [0.0] * EMB_DIM  # 生成全零向量，长度为嵌入维度
            augmented_atttribute_embedding_dict[key][indices[0]] = default_embedding
            pickle.dump(augmented_atttribute_embedding_dict, open(FILE_PATH + 'augmented_atttribute_embedding_dict', 'wb'))
            continue

        try:
            embedding = get_embedding(text)
            augmented_atttribute_embedding_dict[key][indices[0]] = embedding
            # 保存更新后的字典
            pickle.dump(augmented_atttribute_embedding_dict, open(FILE_PATH + 'augmented_atttribute_embedding_dict', 'wb'))
            logging.info(f"Saved embedding for index {indices[0]} in {key} embedding dictionary.")
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP 错误: {e}")
            time.sleep(5)
            LLM_request(toy_augmented_item_attribute, indices, augmented_atttribute_embedding_dict)
        except Exception as ex:
            logging.error(f"未知错误: {ex}")
            time.sleep(5)
            LLM_request(toy_augmented_item_attribute, indices, augmented_atttribute_embedding_dict)


# 初始化嵌入字典
def init_embedding_dict(file_name):
    if os.path.exists(FILE_PATH + file_name):
        logging.info(f"文件 {file_name} 存在。加载中...")
        return pickle.load(open(FILE_PATH + file_name, 'rb'))
    else:
        logging.info(f"文件 {file_name} 不存在。创建新的字典...")
        embedding_dict_names = ['title', 'genre', 'director', 'country', 'language']
        return {name: {} for name in embedding_dict_names}

# 均值填充函数
def pad_embeddings_with_mean(embeddings, target_length):
    mean_value = np.mean(embeddings)
    return embeddings + [mean_value] * (target_length - len(embeddings))

# 合并嵌入字典的函数，并处理嵌入长度不一致的情况
def merge_embedding_dicts(augmented_atttribute_embedding_dict):
    logging.info("Merging embedding dictionaries.")
    augmented_total_embed_dict = {key: [] for key in augmented_atttribute_embedding_dict.keys()}

    for key, embed_dict in augmented_atttribute_embedding_dict.items():
        # 找出该字典中所有嵌入的最大长度
        max_length = max(len(embed_dict[i]) for i in embed_dict.keys())

        for i in embed_dict.keys():
            embedding = embed_dict[i]
            if len(embedding) < max_length:
                # 用该嵌入向量的均值进行填充
                embedding = pad_embeddings_with_mean(embedding, max_length)
                logging.info(f"Padding embedding at index {i} in {key} to length {max_length} using mean value.")
            
            augmented_total_embed_dict[key].append(embedding)

        # 将嵌入列表转换为 NumPy 数组
        augmented_total_embed_dict[key] = np.array(augmented_total_embed_dict[key])

    logging.info("Finished merging embedding dictionaries.")
    return augmented_total_embed_dict

# 批量处理函数
def process_batches(toy_augmented_item_attribute, augmented_atttribute_embedding_dict):
    for i in range(0, toy_augmented_item_attribute.shape[0]):
        indices = [i]
        logging.info(f"Processing indices: {indices}")
        LLM_request(toy_augmented_item_attribute, indices, augmented_atttribute_embedding_dict)

# 降维函数
def reduce_dimension(augmented_total_embed_dict, emb_dim):
    logging.info(f"Reducing dimensions of embeddings to {emb_dim} components.")
    for key, embedding_matrix in augmented_total_embed_dict.items():
        pca = PCA(n_components=emb_dim)
        augmented_total_embed_dict[key] = pca.fit_transform(embedding_matrix)
    logging.info("Finished dimension reduction.")
    return augmented_total_embed_dict

# 主程序
def main():
    logging.info("Starting embedding generation process.")
    
    # 初始化字典
    file_name = "augmented_atttribute_embedding_dict"
    augmented_atttribute_embedding_dict = init_embedding_dict(file_name)


    # 读取增强的项目属性文件，跳过第一行标题
    toy_augmented_item_attribute = pd.read_csv(FILE_PATH + 'augmented_item_attribute_agg.csv', 
                                           names=['id', 'title', 'genre', 'director', 'country', 'language'], 
                                           skiprows=1, header=None)

    # 处理批量请求
    process_batches(toy_augmented_item_attribute, augmented_atttribute_embedding_dict)

    # 合并字典
    augmented_total_embed_dict = merge_embedding_dicts(augmented_atttribute_embedding_dict)

    # 降维并保存结果
    augmented_total_embed_dict = reduce_dimension(augmented_total_embed_dict, EMB_DIM)
    final_matrix = np.hstack([augmented_total_embed_dict[key] for key in augmented_total_embed_dict.keys()])
    np.savetxt(FILE_PATH + "augmented_item_embed_matrix.csv", final_matrix, delimiter=",")
    
    logging.info(f"Reduced embedding matrix has been saved to {FILE_PATH + 'augmented_item_embed_matrix.csv'}")
    logging.info("Embedding generation process completed.")

# 执行主程序
if __name__ == "__main__":
    main()
