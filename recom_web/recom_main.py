import numpy as np
import codecs
from recom_web import utils
from recom_web import algorithm
from recom_web import config
from recom_web import log_helper
from texttable import Texttable
from time import time


# 按行读取文件内容
def read_file(file_name):
    f = codecs.open(file_name, 'r', encoding='iso-8859-15')
    content_lines = f.readlines()
    f.close()
    return content_lines


# 输入按行读取的内容和每行的分割符，得到movie的字典映射信息, MovieID ---> [Title, Genres]
def get_movie_list_and_clean(movie_lines, split):
    movies_info_dict = {}
    for movie in movie_lines:
        movie_info = movie.split(split)
        if movie_info[0] != "" and movie_info[1] != "":
            movies_info_dict[int(movie_info[0])] = movie_info[1:]
    return movies_info_dict


# 输入按行读取的内容和每行的分割符，获取 [[UserID, MovieID, Rating], [...]...]
def get_user_and_rating_info(content_lines, split):
    user_rating = []
    for line in content_lines:
        rate = line.split(split)
        if 1 <= int(rate[2]) <= 5:
            user_rating.append([int(rate[0]), int(rate[1]), int(rate[2])])
    return user_rating


# 格式化成字典数据
# 1.用户字典：dict[user_id] = [(movie_id, rating)...]
# 2.电影字典：dict[movie_id] = [user_id1, user_id2, ...]
def create_user_and_rating_dict(user_rating):
    user_ratings_dict = {}
    movie_users_dict = {}
    for i in user_rating:
        # movie_rating = (i[1], i[2])
        if i[0] not in user_ratings_dict.keys():
            user_ratings_dict[i[0]] = {}
        user_ratings_dict[i[0]][i[1]] = i[2]

        if i[1] in movie_users_dict:
            movie_users_dict[i[1]].append(i[0])
        else:
            movie_users_dict[i[1]] = [i[0]]

    # print(user_ratings_dict[123].__str__())
    return user_ratings_dict, movie_users_dict


# 格式化成字典数据
# 1.用户字典：dict[user_id] = [(movie_id, rating)...]
def create_test_user_rating_dict(user_rating):
    user_ratings_dict = {}
    for i in user_rating:
        if i[0] not in user_ratings_dict.keys():
            user_ratings_dict[i[0]] = {}
        user_ratings_dict[i[0]][i[1]] = i[2]

    return user_ratings_dict


# 得到用户-项目评分矩阵
def get_all_user_movie_rating_matrix(user_ratings_dict, movie_users_dict, user_rating):
    max_user_id = max(user_ratings_dict.keys()) + 1
    max_movie_id = max(movie_users_dict.keys()) + 1

    rating = np.zeros((max_user_id, max_movie_id))
    user_rating_len = len(user_rating)

    for rating_id in range(user_rating_len):
        user_id = user_rating[rating_id][0]
        movie_id = user_rating[rating_id][1]
        rate = user_rating[rating_id][2]
        rating[user_id][movie_id] = rate

    return rating


# 获取用户的临近用户
def get_user_neighbors(user_id, user_ratings_dict, movie_users_dict):
    user_neighbor_ids = []
    # print(f'{user_id}  {type(user_id)}')
    for movie_id in user_ratings_dict[int(user_id)].keys():
        for neighbor_id in movie_users_dict[movie_id]:
            if neighbor_id != user_id and neighbor_id not in user_neighbor_ids:
                user_neighbor_ids.append(neighbor_id)

    return user_neighbor_ids


# 获得两个user的相似度 {[movie_id : rating]}
def get_user_neighbor_similarity(user_ratings, neighbor_ratings, type):
    if config.similarity_algorithm_dict[type] == 'pearson':
        item_set = set(user_ratings.keys()) & set(neighbor_ratings.keys())
        user_rating = []
        neighbor_rating = []

        for item in item_set:
            user_rating.append(user_ratings[item])
            neighbor_rating.append(neighbor_ratings[item])
        return algorithm.pearson(user_rating, neighbor_rating)

    elif config.similarity_algorithm_dict[type] == 'jaccard':
        return algorithm.jaccard(user_ratings.keys(), neighbor_ratings.keys())

    elif config.similarity_algorithm_dict[type] == 'adjusted_cosine':
        return algorithm.adjusted_cosine(user_ratings, neighbor_ratings)

    elif config.similarity_algorithm_dict[type] == 'pear_and_jacc':
        return algorithm.pear_and_jacc(user_ratings, neighbor_ratings)


# 获得用户的临近用户相识度矩阵
def init_user_similarity(user_id, user_ratings_dict, user_neighbors, type):

    user_neighbors_similarity = []
    user_rating = user_ratings_dict[int(user_id)]  # {[movie_id : rating]}

    for neighbor_id in user_neighbors:
        simil = get_user_neighbor_similarity(user_rating, user_ratings_dict[neighbor_id], type)
        user_neighbors_similarity.append([simil, neighbor_id])

    user_neighbors_similarity.sort(reverse=True)

    return user_neighbors_similarity


# 预测评分，生成推荐的项目
def get_recom_items(user_id, selected_simil_user, user_ratings_dict):
    user_current_dict = user_ratings_dict[int(user_id)]
    user_had_rating_ids = user_current_dict.keys()

    neighbor_had_ratings_average_dict = {}
    for neighbor_id in selected_simil_user.keys():
        current_dict = user_ratings_dict[int(neighbor_id)]
        neighbor_had_ratings_average_dict[neighbor_id] = sum(map(lambda x: current_dict[x], current_dict.keys())) / len(
            current_dict.keys())

    user_had_not_rating_ids = []
    for neighbor_id in selected_simil_user.keys():
        for item_id in user_ratings_dict[int(neighbor_id)].keys():
            if item_id not in user_had_rating_ids and item_id not in user_had_not_rating_ids:
                user_had_not_rating_ids.append(item_id)

    user_recom_item_ratings = []
    user_had_ratings_average = sum(map(lambda x: user_current_dict[x], user_had_rating_ids)) / len(user_had_rating_ids)
    all_simil_sum = sum(map(lambda x: selected_simil_user[x], selected_simil_user.keys()))

    for item_id in user_had_not_rating_ids:
        item_score = 0.0
        for neighbor_id in selected_simil_user.keys():
            if item_id in user_ratings_dict[int(neighbor_id)].keys():
                item_score = item_score + selected_simil_user[neighbor_id] * (
                    user_ratings_dict[int(neighbor_id)][item_id] - neighbor_had_ratings_average_dict[neighbor_id])

        item_predict_score = user_had_ratings_average + item_score / all_simil_sum
        user_recom_item_ratings.append([item_predict_score, item_id])

    user_recom_item_ratings.sort(reverse=True)
    return user_recom_item_ratings


# 主程序入口
if __name__ == '__main__':
    log_helper.info('Welcome to use the movie recommend system!')

    # 读取movies.dat 并进行清洗
    log_helper.info(f"reading movies info file, path = {config.movie_info_file_path}")
    movie_content_lines = utils.read_file(config.movie_info_file_path)
    movies_info_dict = get_movie_list_and_clean(movie_content_lines, "::")

    # 读取ratings_base.data 并进行清洗
    log_helper.info(f"reading ratings_base info file, path = {config.user_movie_ratings_base_file_path}")
    rating_base_content_lines = utils.read_file(config.user_movie_ratings_base_file_path)
    user_rating = get_user_and_rating_info(rating_base_content_lines, ":")
    # print(f'{len(user_rating)}')

    log_helper.info("init base user ratings dict and movie user dict")
    user_ratings_dict, movie_users_dict = create_user_and_rating_dict(user_rating)

    # 读取ratings_test.data 并进行清洗
    log_helper.info(f"reading ratings_test info file, path = {config.user_movie_ratings_test_file_path}")
    rating_test_content_lines = utils.read_file(config.user_movie_ratings_test_file_path)
    user_test_rating = get_user_and_rating_info(rating_test_content_lines, ":")

    log_helper.info("init test user ratings dict")
    test_user_rating_dict = create_test_user_rating_dict(user_test_rating)

    is_system_on = True
    top_n = 100
    simil_min = 0
    user_neighbor_num = 200
    while (is_system_on):
        user_id = input('\nPlease input your ID(\'exit\':exit the system):')
        if user_id == 'exit':
            is_system_on = False
            log_helper.warn("You had exit the system!")
        elif not user_id.isdigit():
            log_helper.warn(f'\'{user_id}\' is not a available ID, please try again!')
            continue
        else:
            log_helper.info(f'create recommend movies for user {user_id}')
            star_time = time()

            # 获得用户的临近用户相识度矩阵 [[simil, user_id],...]
            log_helper.info('init user neighbor\'s similarity matrix')

            user_neighbors = get_user_neighbors(user_id, user_ratings_dict, movie_users_dict)
            test_user_movies_dict = test_user_rating_dict[int(user_id)].keys()


            user_similarity = init_user_similarity(user_id, user_ratings_dict, user_neighbors, 1)
            selected_simil_user = {}
            n = 0
            log_helper.info(f'select top {user_neighbor_num} neighbors for user {user_id}')
            for user in user_similarity:
                selected_simil_user[user[1]] = user[0]  # { [user_id, simil_score] .....}
                n = n + 1
                if n == user_neighbor_num:
                    break

            # 获取给用户推荐的电影
            log_helper.info(f'predict recommend movie ratings for user {user_id}')
            user_recom_item_ratings = get_recom_items(user_id, selected_simil_user, user_ratings_dict)

            table = Texttable()
            table.set_deco(Texttable.HEADER)
            table.set_cols_dtype(['t',
                                  't',
                                  't',
                                  't'])
            table.set_cols_align(["l", "l", "l", "l"])
            rows = []
            rows.append([u"id", u"movie", u"user ratings", u"recom ratings"])

            log_helper.info(f'select top {top_n} recommend movies for user {user_id}')
            count_top_n = 0

            for item in user_recom_item_ratings:
                count_top_n = count_top_n + 1
                if item[1] in test_user_movies_dict:
                    rows.append(
                        [count_top_n, movies_info_dict[item[1]][0], test_user_rating_dict[int(user_id)][item[1]],
                         round(item[0], 2)])
                else:
                    rows.append([count_top_n, movies_info_dict[item[1]][0], 0, round(item[0], 2)])
                    # print(f'{count_top_n}  {item[1]}  {movies_info_dict[item[1]][0]}   {round(item[0],2)}  {0}')
                if count_top_n == top_n:
                    break

            log_helper.info(f'below is recommend movies for user {user_id}')
            table.add_rows(rows)
            print(table.draw())

            end_time = time()
            log_helper.info(f'create recommend movies for user {user_id} succeed, cost {round(end_time-star_time,2)}s')
