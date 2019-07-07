import pandas as pd
import codecs
import random
import time
from recom_web import config
from numpy import long


# 按行读取文件内容
def read_file(file_name):
    f = codecs.open(file_name, 'r', encoding='iso-8859-15')
    content_lines = f.readlines()
    f.close()
    return content_lines


def get_current_time():
    ct = time.time()
    time_head = format_time(ct)
    time_secs = (ct - long(ct)) * 1000
    time_stamp = "%s.%03d" % (time_head, time_secs)

    return time_stamp


def format_time(time_parm=time.time(), format='%Y-%m-%d %H:%M:%S'):
    time_ = time.localtime(time_parm)
    return time.strftime(format, time_)


def format_time_secs(time_parm=time.time()):
    time_head = format_time(time_parm)
    time_secs = (time_parm - long(time_parm)) * 1000
    time_stamp = "%s.%03d" % (time_head, time_secs)

    return time_stamp


# 使用pandas读取文件，获取DataFrame
def pandas_read_file(file_name, titles, seq):
    data = pd.read_csv(file_name, sep=seq, names=titles, engine='python')
    return data


# 生成在[min,max]区间内的一个随机数
def create_random(min_num, max_num):
    return random.randint(min_num, max_num)


# 生成n个在区间在[min, max]区间内的随机数
def create_random_list(min_num, max_num, n):
    random_list = []
    for i in range(n):
        random_num = create_random(min_num, max_num)
        while random_num in random_list:
            random_num = create_random(min_num, max_num)
        random_list.append(random_num)
    return random_list


# 把数组数据保存到filename中
def data_to_csv(file_name, data, sep):
    data = list(data)
    file_data = pd.DataFrame(data)
    file_data.to_csv(file_name, sep=sep, index=False, header=False, encoding='iso-8859-15')


# 把数据集分成训练集、测试集
def split_main(test_percent=0.2):
    titles = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pandas_read_file(config.user_movie_ratings_file_path, titles, "::")
    ratings_df['rating_id'] = ratings_df.index + 1
    ratings_df = ratings_df[['rating_id', 'user_id', 'movie_id', 'rating']]

    base_data = []
    test_data = []
    current_user_id = ''
    current_user_dict = {}
    first_in = True
    count = 0
    for index, rating in ratings_df.iterrows():
        if first_in:
            current_user_dict[int(rating['rating_id'])] = [rating['user_id'], rating['movie_id'], rating['rating']]
            current_user_id = rating['user_id']
            count = count + 1
            first_in = False
        else:
            if current_user_id == rating['user_id']:
                current_user_dict[int(rating['rating_id'])] = [rating['user_id'], rating['movie_id'], rating['rating']]
                count = count + 1
            else:
                max_id = max(current_user_dict.keys())
                min_id = min(current_user_dict.keys())
                current_test_count = int(count * test_percent)
                test_random_list = create_random_list(min_id, max_id, current_test_count)

                for rating_id in current_user_dict.keys():
                    if rating_id in test_random_list:
                        test_data.append(current_user_dict[rating_id])
                    else:
                        base_data.append(current_user_dict[rating_id])

                current_user_dict = {}
                current_user_dict[int(rating['rating_id'])] = [rating['user_id'], rating['movie_id'], rating['rating']]
                current_user_id = rating['user_id']
                count = 1

    data_to_csv(config.user_movie_ratings_base_file_path, base_data, ':')
    data_to_csv(config.user_movie_ratings_test_file_path, test_data, ':')

    print(f'{len(base_data)}  {len(test_data)}')  # 802280  197588


# 主程序入口
if __name__ == '__main__':
    star_time = time.time()

    split_main()

    # print(sorted(create_random_list(1, 6040, 100)))
    # test_user_ids = [153, 236, 282, 313, 367, 473, 563, 597, 795, 813, 875, 960, 979, 992, 1034, 1062, 1080, 1112, 1285, 1396, 1411, 1505, 1508, 1511, 1513, 1692, 1784, 1835, 1844, 1845, 1846, 1917, 2012, 2063, 2085, 2103, 2138, 2166, 2197, 2252, 2273, 2283, 2284, 2334, 2436, 2446, 2592, 2702, 2768, 2829, 2851, 2882, 2886, 2954, 2992, 3026, 3051, 3070, 3080, 3125, 3127, 3358, 3388, 3471, 3486, 3543, 3613, 3680, 3824, 3948, 3987, 4102, 4105, 4142, 4210, 4299, 4352, 4481, 4497, 4528, 4560, 4565, 4637, 4675, 4832, 4877, 4896, 4903, 4972, 5109, 5150, 5307, 5315, 5479, 5610, 5693, 5805, 5964, 5989, 6028]

    end_time = time.time()
    print(f'{end_time-star_time}')  # 244.52696180343628
