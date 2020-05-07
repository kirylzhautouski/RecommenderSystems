# RecommenderSystems

Developed collaborative filtering models for solving the problem of recommending items (movies in this case) as my course work. 

The first model is based on the algorithm called k nearest neighbours. The similarity of items is counted based on one of the chosen metrics, either Euclidean
or cosine. `Trainset` and `Dataset` classes were developed for a more convenient way of storing and proccessing data.
![Class diagram](/screenshots/knn_class_diagram.png)

The second model is matrix factorization model. It represents users and items as embeddings and uses them to decide how much user will like the
following item. Model is built, trained and tested using NumPy, Pandas and PyTorch.

![Error loss](/screenshots/error_plot.png)
