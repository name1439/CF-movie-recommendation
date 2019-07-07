# Jaccard相似度
def jaccard(user, neighbor):
    item_set = set(user) & set(neighbor)
    n = len(item_set)
    return n / (len(user) + len(neighbor) - n)


# 余弦相似度
def cosine(user, neighbor):
    fenzi_sum = sum(map(lambda x: x[0] * x[1], zip(user, neighbor)))
    fenmu_sum1 = sum(map(lambda x: x * x, user))
    fenmu_sum2 = sum(map(lambda x: x * x, neighbor))
    return fenzi_sum / (fenmu_sum1 ** 0.5) / (fenmu_sum2 ** 0.5)


# Pearson相关系数
def pearson(user, neighbor):
    user_avg = (sum(user) + 0.0) / len(user)
    neighbor_avg = (sum(neighbor) + 0.0) / len(neighbor)
    fenzi_sum = sum(map(lambda x: (x[0] - user_avg) * (x[1] - neighbor_avg), zip(user, neighbor)))
    fenmu_sum1 = sum(map(lambda x: (x - user_avg) * (x - user_avg), user))
    fenmu_sum2 = sum(map(lambda x: (x - neighbor_avg) * (x - neighbor_avg), neighbor))
    if not fenmu_sum1 or not fenmu_sum2:
        return cosine(user, neighbor)
    return fenzi_sum / (fenmu_sum1 ** 0.5) / (fenmu_sum2 ** 0.5)


# Adjusted cosine相关系数
def adjusted_cosine(user_ratings, neighbor_ratings):
    user_ratings_keys = user_ratings.keys()
    neighbor_ratings_keys = neighbor_ratings.keys()
    item_set = set(user_ratings_keys) & set(neighbor_ratings_keys)
    user_rating = []
    neighbor_rating = []

    for item in item_set:
        user_rating.append(user_ratings[item])
        neighbor_rating.append(neighbor_ratings[item])
    user_same_avg = (sum(user_rating) + 0.0) / len(user_rating)
    neighbor_same_avg = (sum(neighbor_rating) + 0.0) / len(neighbor_rating)

    user_all_avg = (sum(map(lambda x: user_ratings[x], user_ratings_keys)) + 0.0) / len(user_ratings_keys)
    neighbor_all_avg = (sum(map(lambda x: neighbor_ratings[x], neighbor_ratings_keys)) + 0.0) / len(neighbor_ratings_keys)

    fenzi_sum = sum(map(lambda x: (x[0] - user_same_avg) * (x[1] - neighbor_same_avg), zip(user_rating, neighbor_rating)))

    fenmu_user_sum = 0.0
    fenmu_neighbor_sum = 0.0

    for movie_id in user_ratings_keys:
        fenmu_user_sum = fenmu_user_sum + (user_ratings[movie_id] - user_all_avg) ** 2
    for movie_id in neighbor_ratings_keys:
        fenmu_neighbor_sum = fenmu_neighbor_sum + (neighbor_ratings[movie_id] - neighbor_all_avg) ** 2

    if not fenmu_user_sum or not fenmu_neighbor_sum:
        return cosine(user_rating, neighbor_rating)
    return fenzi_sum / (fenmu_user_sum ** 0.5) / (fenmu_neighbor_sum ** 0.5)


# Pearson相关系数 与 Jaccard公式
def pear_and_jacc(user_ratings, neighbor_ratings):
    user_ratings_keys = user_ratings.keys()
    neighbor_ratings_keys = neighbor_ratings.keys()
    itemset = set(user_ratings_keys) & set(neighbor_ratings_keys)
    user_same_rating = []
    neighbor_same_rating = []
    user_same_rating_sum = 0.0
    neighbor_same_rating_sum = 0.0

    for item in itemset:
        user_same_rating.append(user_ratings[item])
        user_same_rating_sum = user_same_rating_sum + user_ratings[item]
        neighbor_same_rating.append(neighbor_ratings[item])
        neighbor_same_rating_sum = neighbor_same_rating_sum + neighbor_ratings[item]

    user_all_rating_sum = sum(map(lambda x: user_ratings[x], user_ratings_keys))
    neighbor_all_rating_sum = sum(map(lambda x: neighbor_ratings[x], neighbor_ratings_keys))

    user_percent = user_same_rating_sum / user_all_rating_sum
    neighbor_percent = neighbor_same_rating_sum / neighbor_all_rating_sum

    pearson_simil = pearson(user_same_rating, neighbor_same_rating)
    jaccard_simil = jaccard(user_ratings_keys, neighbor_ratings_keys)

    return (pearson_simil * (user_percent ** 0.5) * (neighbor_percent ** 0.5) * jaccard_simil)


# _list = ['name', 'test', 'haha', 'pip', 'hello']

# 主程序入口
if __name__ == '__main__':
    pass
