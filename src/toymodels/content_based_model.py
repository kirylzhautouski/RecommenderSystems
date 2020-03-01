from movie import Movie


def find_best_match(movies, similar_to):
    best_score = 0
    best_match = None
    for movie in movies:
        match_score = movie.match(similar_to)
        if match_score > best_score:
            best_score = match_score
            best_match = movie

    return best_match


if __name__ == '__main__':
    movies = [
        Movie('Minions', animated=True, is_marvel=False,
              has_super_villain=True, passes_bechdel_test=False),
        Movie('Avengers: Age Of Ultron', animated=False, is_marvel=True,
              has_super_villain=True, passes_bechdel_test=True),
        Movie('Ant-Man', animated=False, is_marvel=True,
              has_super_villain=True, passes_bechdel_test=False),
    ]

    watched_and_liked = Movie('Inside out', animated=True, is_marvel=False,
                              has_super_villain=False, passes_bechdel_test=True)

    print(find_best_match(movies, watched_and_liked))
