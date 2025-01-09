from openai import OpenAI
import pandas as pd
import numpy as np
import os
import pickle
import requests
import time
import logging
from sklearn.decomposition import PCA


FILE_PATH = "./aug_data/MovieLens100K/"
EMB_DIM = 6

# 配置日志记录
logging.basicConfig(filename='get_user_embedding.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 获取文本嵌入的函数
def get_embedding(prompt):
    client = OpenAI(
        api_key=os.environ.get('sk-fnDLWtODH8hjcW5o1b69A68a8c17474584107f2eB5B183Ca'), 
        base_url="https://tbb.gpt88.top/v1/"
    )

    response = client.embeddings.create(
        model='text-embedding-ada-002', input=prompt, encoding_format="float"
    )

    return np.array([dp.embedding for dp in response.data])

# 请求嵌入并更新字典
def LLM_request(user_attributes, indices, user_embedding_dict):
    if not indices:
        logging.info("No indices provided, skipping request.")
        return

    for key in user_embedding_dict.keys():
        if indices[0] in user_embedding_dict[key]:
            logging.info(f"Index {indices[0]} already in {key} embedding dictionary, skipping.")
            continue
        
        try:
            # 将数值类型转换为字符串
            embedding = get_embedding(str(user_attributes[key][indices].values[0]))
            user_embedding_dict[key][indices[0]] = embedding
            # 保存更新后的字典
            pickle.dump(user_embedding_dict, open(FILE_PATH + 'user_embedding_dict', 'wb'))
            logging.info(f"Saved embedding for index {indices[0]} in {key} embedding dictionary.")
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP 错误: {e}")
            time.sleep(5)
            LLM_request(user_attributes, indices, user_embedding_dict)
        except Exception as ex:
            logging.error(f"未知错误: {ex}")
            time.sleep(5)
            LLM_request(user_attributes, indices, user_embedding_dict)

# 初始化嵌入字典
def init_embedding_dict(file_name):
    if os.path.exists(FILE_PATH + file_name):
        logging.info(f"文件 {file_name} 存在。加载中...")
        return pickle.load(open(FILE_PATH + file_name, 'rb'))
    else:
        logging.info(f"文件 {file_name} 不存在。创建新的字典...")
        embedding_dict_names = ['age', 'gender', 'occupation', 'country', 'language']
        return {name: {} for name in embedding_dict_names}

# 合并嵌入字典
def merge_embedding_dicts(user_embedding_dict):
    logging.info("Merging embedding dictionaries.")
    user_total_embed_dict = {key: [] for key in user_embedding_dict.keys()}
    for key, embed_dict in user_embedding_dict.items():
        user_total_embed_dict[key] = np.array([embed_dict[i] for i in range(len(embed_dict))])
    logging.info("Finished merging embedding dictionaries.")
    return user_total_embed_dict

# 批量处理函数
def process_batches(user_attributes, user_embedding_dict):
    for i in range(0, user_attributes.shape[0]):
        indices = [i]
        logging.info(f"Processing indices: {indices}")
        LLM_request(user_attributes, indices, user_embedding_dict)

# 降维函数
def reduce_dimension(user_total_embed_dict, emb_dim):
    logging.info(f"Reducing dimensions of embeddings to {emb_dim} components.")
    for key, embedding_matrix in user_total_embed_dict.items():
        pca = PCA(n_components=emb_dim)
        user_total_embed_dict[key] = pca.fit_transform(embedding_matrix)
    logging.info("Finished dimension reduction.")
    return user_total_embed_dict

# 主程序
def main():
    logging.info("Starting embedding generation process.")
    
    # 初始化字典
    file_name = "user_embedding_dict"
    user_embedding_dict = init_embedding_dict(file_name)

    # 读取增强的用户属性文件，跳过第一行标题
    user_attributes = pd.read_csv(FILE_PATH + 'augmented_user_attribute_agg1.csv', 
                                  names=['user_id', 'age', 'gender', 'occupation', 'country', 'language'], 
                                  skiprows=1, header=None)

    # 处理批量请求
    process_batches(user_attributes, user_embedding_dict)

    # 合并字典
    user_total_embed_dict = merge_embedding_dicts(user_embedding_dict)

    # 降维并保存结果
    user_total_embed_dict = reduce_dimension(user_total_embed_dict, EMB_DIM)
    final_matrix = np.hstack([user_total_embed_dict[key] for key in user_total_embed_dict.keys()])
    np.savetxt(FILE_PATH + "augmented_user_embed_matrix1.csv", final_matrix, delimiter=",")
    
    logging.info(f"Reduced embedding matrix has been saved to {FILE_PATH + 'augmented_user_embed_matrix1.csv'}")
    logging.info("Embedding generation process completed.")

# 执行主程序
if __name__ == "__main__":
    main()
