

def recommend_movie(liked_movies_for_users, movies, user_liked):
    score = [0] * len(movies)
    for other_user in liked_movies_for_users:
        if other_user[user_liked]:
            for i, movie in enumerate(other_user):
                if movie:
                    score[i] += 1

    del movies[user_liked]
    del score[user_liked]

    return movies[score.index(max(score))]


if __name__ == '__main__':
    movies = ['Inside out', 'Minions', 'Avengers: Age Of Ultron', 'Ant-man']
    liked_movies_for_users = [
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1]
    ]

    user_liked = 0
    print(recommend_movie(liked_movies_for_users, movies, user_liked))
