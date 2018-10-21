EE219 Project 3
Collaborative Filtering
Winter 2018

Teammembers:
Fangyao Liu----UCLA ECE Graduate student
	       UID:204945018
	       Email:fangyao@okstate.edu
Xuan Hu    ----UCLA ECE Graduate student
	       UID:505031796
	       Email:x1310317@gmail.com
YanZhe Xu  ----UCLA ECE Graduate student
	       UID:404946757
	       Email:xuyanzhesjtu@g.ucla.edu
Zhechen Xu ----UCLA ECE Graduate student
	       UID:805030074
	       Email:xzhechen@ucla.edu

Introduction: In this project, we built a recommender system for the MovieLens dataset. Based on multiple users, movies and the movie rating information, we are able to give out movies recommendation for users. We built recommendation system using collaborative filtering methods. The basic idea of collaborative filtering methods is that these unspecified ratings can be imputed because the observed ratings are often highly correlated across various users and items.  We will implement and analyze the performance of two types of collaborative filtering methods: Neighborhood-based collaborative filtering and Model-based collaborative filtering. 



IMPLEMENTATION
    Environment:
        python 2.7
    Dependencies:
	a.surprise v0.1
	b.matplotlib v2.1.0
	c.numpy v1.13.3

Usage:
put util.py and recommendation.py with ratings.csv and movies.csv together.
$python recommendation.py