import numpy as np
import torch

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score

import pickle
from os import path


def train_initial_model():
    dataset = get_movielens_dataset(variant='100K')

    train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

    model = ImplicitFactorizationModel(loss='adaptive_hinge',
                                    embedding_dim=128,  # latent dimensionality
                                    n_iter=10,  # number of epochs of training
                                    batch_size=1024,  # minibatch size
                                    l2=1e-9,  # strength of L2 regularization
                                    learning_rate=1e-3,
                                    use_cuda=torch.cuda.is_available())

    print('Fitting the model')

    model.fit(train, verbose=True)
    print(type(model))

    model_file = open('models/filmclub.model', 'wb')
    pickle.dump(model, model_file)
    model_file.close()

    dataset.num_users = 1000000

    dataset_file = open('data/dataset.pkl', 'wb')
    pickle.dump(dataset, dataset_file)
    dataset_file.close()

    train_rmse = rmse_score(model, train)
    test_rmse = rmse_score(model, test)

    print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))

def get_model_and_history():
    model_file = open('models/filmclub.model', 'rb')
    model = pickle.load(model_file)
    interactions_file = open('data/dataset.pkl', 'rb')
    interactions = pickle.load(interactions_file)

    return model, interactions

def get_prediction(user=999999, suggestions=[1,10,100]):
    model, interactions = get_model_and_history()
    user = np.array([user])
    film_suggestions = np.array(suggestions)
    preds = model.predict(user, film_suggestions)
    print(preds)
    print(np.max(preds))
    print(np.argmax(preds))
    return film_suggestions[np.argmax(preds)]

def update_model(film):
    model, interactions = get_model_and_history()
    interactions.user_ids = np.append(interactions.user_ids, [999999])
    interactions.item_ids = np.append(interactions.item_ids, [film])
    model.fit(interactions, verbose=True)
    return model, interactions

def model_exists():
    return path.exists("models/filmclub.mobel") & path.exists("data/dataset.pkl")

if model_exists:
    get_prediction([30], [1,2,3])
else:
    train_initial_model()
