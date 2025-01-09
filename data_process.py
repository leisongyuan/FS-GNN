import torch
import numpy as np


def process_movielens100k(graph, mode):

    # 重新读取文件内容
    def read_file(file_path, num_users, num_items):
        with open(file_path, 'r') as file:
            content = file.readlines()

        # 提取数据到三个列表中，同时确保前两列为int，最后一列为float
        user_list= []
        item_list = []
        rating_list = []

        # 遍历文件内容，提取每一行的数据
        for line in content:
            # 分割每一行的数据
            split_line = line.strip().split('\t')
            if len(split_line) >= 3:
                # 将前两列转换为int，最后一列转换为float
                user = int(split_line[0])- 1
                item = int(split_line[1]) + num_users - 1
                rating = float(split_line[2])

                # 添加到对应的列表
                user_list.append(user)
                item_list.append(item)
                rating_list.append(rating)

        # 将列表转换为 tensor
        user_tensor = torch.tensor(user_list, dtype=torch.long)
        item_tensor = torch.tensor(item_list, dtype=torch.long)

        # 将两个 tensor 合并为一个二维 tensor
        edge_index = torch.stack([user_tensor, item_tensor], dim=0)

        ratings = torch.tensor(rating_list, dtype=torch.float32)

        return edge_index, ratings
    
    # 选择处理方式
    if mode == 'ucs':
        # 用户冷启动
        train_path = './cold_data/ml100k/ucs_train.dat'
        val_path = './cold_data/ml100k/ucs_val.dat'
        test_path = './cold_data/ml100k/ucs_test.dat'
    elif mode == 'ics':
        # 项目冷启动
        train_path = './cold_data/ml100k/ics_train.dat'
        val_path = './cold_data/ml100k/ics_val.dat'
        test_path = './cold_data/ml100k/ics_test.dat'
    elif mode == 'warm':
        # 热启动
        train_path = './cold_data/ml100k/warm_train.dat'
        val_path = './cold_data/ml100k/warm_val.dat'
        test_path = './cold_data/ml100k/warm_test.dat'
    
    # 加载数据集
    num_movies = graph['movie'].x.shape[0]
    num_users = graph['user'].x.shape[0]
    train_edge_index, train_ratings = read_file(train_path, num_users, num_movies)
    val_edge_index, val_ratings = read_file(val_path, num_users, num_movies)
    test_edge_index, test_ratings = read_file(test_path, num_users, num_movies)
    user_fea = graph['user'].x
    movie_fea = graph['movie'].x
    user_fea_dim = user_fea.size(1)
    item_fea_dim = movie_fea.size(1)


    return user_fea, movie_fea, train_edge_index, val_edge_index, test_edge_index, train_ratings, val_ratings, test_ratings, user_fea_dim, item_fea_dim


def process_movielens1m(graph, mode):

    # 重新读取文件内容
    def read_file(file_path, num_users, num_items):
        with open(file_path, 'r') as file:
            content = file.readlines()

        # 提取数据到三个列表中，同时确保前两列为int，最后一列为float
        user_list= []
        item_list = []
        rating_list = []

        # 遍历文件内容，提取每一行的数据
        for line in content:
            # 分割每一行的数据
            split_line = line.strip().split('\t')
            if len(split_line) >= 3:
                # 将前两列转换为int，最后一列转换为float
                user = int(split_line[0])
                item = int(split_line[1]) + num_users
                rating = float(split_line[2])

                # 添加到对应的列表
                user_list.append(user)
                item_list.append(item)
                rating_list.append(rating)


        # 将列表转换为 tensor
        user_tensor = torch.tensor(user_list, dtype=torch.long)
        item_tensor = torch.tensor(item_list, dtype=torch.long)

        # 将两个 tensor 合并为一个二维 tensor
        edge_index = torch.stack([user_tensor, item_tensor], dim=0)

        ratings = torch.tensor(rating_list, dtype=torch.float32)

        return edge_index, ratings
    
    # 选择处理方式
    if mode == 'ucs':
        # 用户冷启动
        train_path = './cold_data/ml1m/ucs_train.dat'
        val_path = './cold_data/ml1m/ucs_val.dat'
        test_path = './cold_data/ml1m/ucs_test.dat'
    elif mode == 'ics':
        # 项目冷启动
        train_path = './cold_data/ml1m/ics_train.dat'
        val_path = './cold_data/ml1m/ics_val.dat'
        test_path = './cold_data/ml1m/ics_test.dat'
    elif mode == 'warm':
        # 热启动
        train_path = './cold_data/ml1m/warm_train.dat'
        val_path = './cold_data/ml1m/warm_val.dat'
        test_path = './cold_data/ml1m/warm_test.dat'
    
    # 加载数据集
    num_movies = graph['movie'].x.shape[0]
    num_users = graph['user'].x.shape[0]
    train_edge_index, train_ratings = read_file(train_path, num_users, num_movies)
    val_edge_index, val_ratings = read_file(val_path, num_users, num_movies)
    test_edge_index, test_ratings = read_file(test_path, num_users, num_movies)
    user_fea = graph['user'].x
    movie_fea = graph['movie'].x
    user_fea_dim = user_fea.size(1)
    item_fea_dim = movie_fea.size(1)

    return user_fea, movie_fea, train_edge_index, val_edge_index, test_edge_index, train_ratings, val_ratings, test_ratings, user_fea_dim, item_fea_dim


def process_yelp(graph, mode):

    # 重新读取文件内容
    def read_file(file_path, num_users, num_items):
        with open(file_path, 'r') as file:
            content = file.readlines()

        # 提取数据到三个列表中，同时确保前两列为int，最后一列为float
        user_list= []
        item_list = []
        rating_list = []

        # 遍历文件内容，提取每一行的数据
        for line in content:
            # 分割每一行的数据
            split_line = line.strip().split('\t')
            if len(split_line) >= 3:
                # 将前两列转换为int，最后一列转换为float
                user = int(split_line[0])
                item = int(split_line[1]) + num_users
                rating = float(split_line[2])

                # 添加到对应的列表
                user_list.append(user)
                item_list.append(item)
                rating_list.append(rating)


        # 将列表转换为 tensor
        user_tensor = torch.tensor(user_list, dtype=torch.long)
        item_tensor = torch.tensor(item_list, dtype=torch.long)

        # 将两个 tensor 合并为一个二维 tensor
        edge_index = torch.stack([user_tensor, item_tensor], dim=0)

        ratings = torch.tensor(rating_list, dtype=torch.float32)

        return edge_index, ratings
    
    # 选择处理方式
    if mode == 'ucs':
        # 用户冷启动
        train_path = './cold_data/ml1m/ucs_train.dat'
        val_path = './cold_data/ml1m/ucs_val.dat'
        test_path = './cold_data/ml1m/ucs_test.dat'
    elif mode == 'ics':
        # 项目冷启动
        train_path = './cold_data/ml1m/ics_train.dat'
        val_path = './cold_data/ml1m/ics_val.dat'
        test_path = './cold_data/ml1m/ics_test.dat'
    elif mode == 'warm':
        # 热启动
        train_path = './cold_data/ml1m/warm_train.dat'
        val_path = './cold_data/ml1m/warm_val.dat'
        test_path = './cold_data/ml1m/warm_test.dat'
    
    # 加载数据集
    num_movies = graph['movie'].x.shape[0]
    num_users = graph['user'].x.shape[0]
    train_edge_index, train_ratings = read_file(train_path, num_users, num_movies)
    val_edge_index, val_ratings = read_file(val_path, num_users, num_movies)
    test_edge_index, test_ratings = read_file(test_path, num_users, num_movies)
    user_fea = graph['user'].x
    movie_fea = graph['movie'].x
    user_fea_dim = user_fea.size(1)
    item_fea_dim = movie_fea.size(1)


    return user_fea, movie_fea, train_edge_index, val_edge_index, test_edge_index, train_ratings, val_ratings, test_ratings, user_fea_dim, item_fea_dim