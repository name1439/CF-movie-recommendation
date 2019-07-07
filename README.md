## CF-movie-recommendation
### An Implementation of the User-based Collaborative Filtering Algorithm.

#### Abstract
With the universal popularity of the Internet and the gradual maturity of e-commerce, the recommendation algorithm has become a research direction that has received more and more attention. Among the many recommendation algorithms, the user-based collaborative filtering recommendation algorithm is the earliest one, and is one of the widely used recommendation algorithms.

#### Content
The research of this algorithm consists of five modules, namely data preprocessing module, similarity calculation module, calculation module of neighboring user set, prediction scoring module and evaluation standard module. Four similarity calculation methods are used in the calculation of user similarity, which are Jaccard similarity, Pearson correlation coefficient, Adjusted cosine similarity, Jaccard similarity and Pearson correlation coefficient mixed similarity. Finally, the algorithm using different similarity calculation methods is evaluated using recall rate (Recall) and mean absolute error (MAE).

#### DataSet 
- MovieLens - 1M

#### Development language 
- Python 3.6.7

#### Installation
- Open the project with Pycharm, run the "recom_app.py" file.
- Use a browser to access "http://localhost:8080".

+ For training the algorithm, just run the "recom_test.py" file.


