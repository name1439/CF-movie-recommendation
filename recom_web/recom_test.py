from recom_web import utils
from recom_web import config
from recom_web import log_helper
from recom_web import recom_main
from time import time


def test_main():
    log_helper.info(f'Welcome user to use the movie recommend system!')

    # 读取movies.dat 并进行清洗
    # log_helper.info(f"reading movies info file, path = {config.movie_info_file_path}")
    # movie_content_lines = utils.read_file(config.movie_info_file_path)
    # movies_info_dict = recom_main.get_movie_list_and_clean(movie_content_lines, "::")

    # 读取ratings_base.data 并进行清洗
    log_helper.info(f"reading ratings_base info file, path = {config.user_movie_ratings_base_file_path}")
    rating_base_content_lines = utils.read_file(config.user_movie_ratings_base_file_path)
    user_rating = recom_main.get_user_and_rating_info(rating_base_content_lines, ":")
    # print(f'{len(user_rating)}')

    log_helper.info("init base user ratings dict and movie user dict")
    user_ratings_dict, movie_users_dict = recom_main.create_user_and_rating_dict(user_rating)

    # 读取ratings_test.data 并进行清洗
    log_helper.info(f"reading ratings_test info file, path = {config.user_movie_ratings_test_file_path}")
    rating_test_content_lines = utils.read_file(config.user_movie_ratings_test_file_path)
    user_test_rating = recom_main.get_user_and_rating_info(rating_test_content_lines, ":")

    log_helper.info("init test user ratings dict")
    test_user_rating_dict = recom_main.create_test_user_rating_dict(user_test_rating)

    user_neighbor_num_dict = [10, 15, 20, 25, 30, 35, 40, 45, 50]

    user_neighbor_num_MAE = {}
    user_neighbor_num_recall = {}

    for unn in user_neighbor_num_dict:
        user_neighbor_num = unn
        # top_n = 20
        # user_neighbor_num = 10
        user_algorithm_MAE = {}
        # uesr_recom_items = {}
        user_algorithm_recall = {}

        test_user_ids = [45, 49, 153, 235, 313, 367, 473, 549, 563, 597, 795, 813, 875, 960, 979, 992, 1001, 1034, 1062,
                         1080, 1112, 1285, 1396, 1411, 1505, 1508, 1511, 1513, 1692, 1784, 1835, 1844, 1845, 1846, 1917,
                         2012, 2063, 2085, 2103, 2138, 2166, 2197, 2273, 2283, 2284, 2334, 2436, 2446, 2592, 2702, 2768,
                         2829, 2851, 2882, 2886, 2954, 3026, 3051, 3070, 3080, 3125, 3358, 3388, 3471, 3486, 3543, 3576,
                         3613, 3680, 3824, 3948, 3987, 4102, 4105, 4210, 4299, 4352, 4481, 4497, 4528, 4560, 4565, 4637,
                         4675, 4832, 4877, 4896, 4903, 4972, 5109, 5150, 5307, 5315, 5479, 5610, 5693, 5805, 5964, 5989,
                         6028]
        for id in test_user_ids:
            user_id = id

            print()
            log_helper.info(f'create recommend movies for user {user_id}, with {user_neighbor_num} neighbors')

            user_algorithm_MAE[user_id] = {}
            user_algorithm_recall[user_id] = {}
            # uesr_recom_items[user_id] = {}

            user_neighbors = recom_main.get_user_neighbors(user_id, user_ratings_dict, movie_users_dict)
            test_user_movies_dict = test_user_rating_dict[int(user_id)].keys()

            user_all_movie_count = len(test_user_movies_dict) + len(user_ratings_dict[int(user_id)].keys())

            for type in config.similarity_algorithm_dict.keys():
                star_time = time()
                # 获得用户的临近用户相识度矩阵 [[simil, user_id],...]
                log_helper.info(
                    f'init user neighbor\'s similarity matrix with \'{config.similarity_algorithm_dict[type]}\' algorithm')
                user_similarity = recom_main.init_user_similarity(user_id, user_ratings_dict, user_neighbors, type)

                selected_simil_user = {}
                n = 0
                log_helper.info(f'select top {user_neighbor_num} neighbors for user {user_id}')
                for user in user_similarity:
                    selected_simil_user[user[1]] = user[0]  # { [user_id, simil_score] .....}
                    n = n + 1
                    if n == user_neighbor_num:
                        break

                # 获取给用户推荐的电影
                # log_helper.info(f'predict recommend movie ratings for user {user_id}')
                user_recom_item_ratings = recom_main.get_recom_items(user_id, selected_simil_user, user_ratings_dict)

                # log_helper.info(f'select top {top_n} recommend movies for user {user_id}')
                # count_top_n = 0

                current_user_rating_sum = 0.0
                current_user_rating_len = 0

                for item in user_recom_item_ratings:
                    # count_top_n = count_top_n + 1
                    if item[1] in test_user_movies_dict:
                        user_rating = test_user_rating_dict[int(user_id)][item[1]]
                        current_user_rating_sum = current_user_rating_sum + abs(item[0] - user_rating)
                        current_user_rating_len = current_user_rating_len + 1

                        # if count_top_n == top_n:
                        #     break

                if current_user_rating_len != 0:
                    user_algorithm_MAE[user_id][type] = round(current_user_rating_sum / current_user_rating_len, 3)
                else:
                    user_algorithm_MAE[user_id][type] = 0

                user_algorithm_recall[user_id][type] = current_user_rating_len / user_all_movie_count
                end_time = time()
                log_helper.info(f'create recommend movies for user {user_id} succeed, cost {round(end_time-star_time,2)}s')

        # print(user_algorithm_MAE)

        # mae_avg = {}
        user_neighbor_num_MAE[user_neighbor_num] = {}
        user_neighbor_num_recall[user_neighbor_num] = {}

        for type in config.similarity_algorithm_dict.keys():
            mae_sum = 0.0
            recall_sum = 0.0
            for id in test_user_ids:
                mae_sum = mae_sum + user_algorithm_MAE[id][type]
                recall_sum = recall_sum + user_algorithm_recall[id][type]

            user_neighbor_num_MAE[user_neighbor_num][type] = round(mae_sum / 100, 4)
            user_neighbor_num_recall[user_neighbor_num][type] = round(recall_sum / 100, 4)

        # print(mae_avg)
    print(user_neighbor_num_MAE)
    print(user_neighbor_num_recall)

    # plt.bar([1, 3, 5, 7], [user_algorithm_Recall[0], user_algorithm_Recall[1], user_algorithm_Recall[2], user_algorithm_Recall[3]], label='Recall')
    # plt.bar([2, 4, 6, 8], [user_algorithm_MAE[0], user_algorithm_MAE[1], user_algorithm_MAE[2], user_algorithm_MAE[3]], label='MAE')
    #
    # plt.legend()
    # plt.xlabel('Recall  and  MAE')
    # plt.ylabel('value')
    # plt.show()


if __name__ == "__main__":
    star_time = time()

    test_main()

    end_time = time()

    print(f'{end_time-star_time}')
