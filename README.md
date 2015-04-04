# spam_detection
I am creating a spam detector for teaching me *Naive Bayes* classifier. 

## How am I teaching Naive Bayes?
The theory of Naive Bayes is not difficult to understand. Please look to my Quora blogs for theory of Naive Bayes. To learn more practical aspects of Naive Bayes, I am solving the Kaggle Spam detection challenge problem. In this problem, using a dataset of 2500 emails containing 1829 ham emails, my goal is to create a classifier that can differentiate whether a particular email is spam or ham.

My first approach is very simple:
1. I fed the subject and body of emails to a count vectorizer and I created a word-count features for each email. Subsequently, I used MultinomialNB from sklearn and I got an accuracy of 0.6 which is very bad.
