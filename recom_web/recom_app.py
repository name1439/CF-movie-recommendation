import web
from recom_web import utils
from recom_web import config
from recom_web import log_helper
from recom_web import recom_main
from texttable import Texttable
from time import time

urls = (
    '/', 'index',
    '/list.*', 'list'
)

app = web.application(urls, globals())
render = web.template.render('templates')


class index:
    def GET(self):
        return render.index()


class list:
    def GET(self):
        inp = web.input()
        user_id = inp.userId

        # print(user_id)

        log_helper.info(f'Welcome user {user_id} to use the movie recommend system!')
        # 读取movies.dat 并进行清洗
        log_helper.info(f"reading movies info file, path = {config.movie_info_file_path}")
        movie_content_lines = utils.read_file(config.movie_info_file_path)
        movies_info_dict = recom_main.get_movie_list_and_clean(movie_content_lines, "::")

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

        top_n = 20
        user_neighbor_num = 50

        log_helper.info(f'create recommend movies for user {user_id}')
        star_time = time()

        user_algorithm_MAE = {}
        uesr_recom_items = {}
        user_algorithm_Recall = {}

        user_neighbors = recom_main.get_user_neighbors(user_id, user_ratings_dict, movie_users_dict)
        test_user_movies_dict = test_user_rating_dict[int(user_id)].keys()

        user_all_movie_count = len(test_user_movies_dict) + len(user_ratings_dict[int(user_id)].keys())

        for type in config.similarity_algorithm_dict.keys():
            # 获得用户的临近用户相识度矩阵 [[simil, user_id],...]
            log_helper.info(f'init user neighbor\'s similarity matrix with \'{config.similarity_algorithm_dict[type]}\' algorithm')
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
            log_helper.info(f'predict recommend movie ratings for user {user_id}')
            user_recom_item_ratings = recom_main.get_recom_items(user_id, selected_simil_user, user_ratings_dict)

            table = Texttable()
            table.set_deco(Texttable.HEADER)
            table.set_cols_dtype(['t',  # text
                                  't',  # float (decimal)
                                  't',
                                  't'])  # automatic
            table.set_cols_align(["l", "l", "l", "l"])
            rows = []
            rows.append([u"id", u"movie", u"user ratings", u"recom ratings"])

            log_helper.info(f'select top {top_n} recommend movies for user {user_id}')
            count_top_n = 0

            current_user_rating_sum = 0.0
            current_user_rating_len = 0

            uesr_recom_items[type] = []


            for item in user_recom_item_ratings:
                count_top_n = count_top_n + 1
                if item[1] in test_user_movies_dict:
                    user_rating = test_user_rating_dict[int(user_id)][item[1]]
                    current_item = [count_top_n, movies_info_dict[item[1]][0], user_rating, round(item[0], 2)]
                    rows.append(current_item)
                    uesr_recom_items[type].append(current_item)
                    current_user_rating_sum = current_user_rating_sum + abs(item[0] - user_rating)
                    current_user_rating_len = current_user_rating_len + 1

                else:
                    current_item = [count_top_n, movies_info_dict[item[1]][0], 0, round(item[0], 2)]
                    rows.append(current_item)
                    uesr_recom_items[type].append(current_item)
                if count_top_n == top_n:
                    break

            log_helper.info(f'below is recommend movies for user {user_id}')
            table.add_rows(rows)
            print(table.draw())

            if current_user_rating_len != 0:
                user_algorithm_MAE[type] = round(current_user_rating_sum / current_user_rating_len, 3)
            else:
                user_algorithm_MAE[type] = 0

            # user_algorithm_MAE[type] = round(current_user_rating_sum / current_user_rating_len, 3)

            user_algorithm_Recall[type] = round(current_user_rating_len / len(test_user_movies_dict), 3)

            end_time = time()
            log_helper.info(
                f'create recommend movies for user {user_id} succeed, cost {round(end_time-star_time,2)}s')

        # print(user_algorithm_MAE)

        # _list = ['name', 'test', 'haha', 'pip', 'hello']

        return render.list(uesr_recom_items, user_algorithm_MAE, user_algorithm_Recall)


if __name__ == "__main__":
    app.run()
