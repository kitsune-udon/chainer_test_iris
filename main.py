import chainer
import chainer.links as L
from chainer import iterators, optimizers, training, datasets
from chainer.training import extensions
import chainer.functions as F
import numpy as np
from sklearn.datasets import load_iris

class MyModel(chainer.Chain):
    def __init__(self):
        super(MyModel, self).__init__(
                l1 = L.Linear(4, 100),
                l2 = L.Linear(100, 3))

    def __call__(self, x):
        h = self.l1(x)
        z = self.l2(h)
        return z

def iris_data():
    iris = load_iris()
    X = iris.data
    X = X.astype(np.float32)
    Y = iris.target
    Y = Y.flatten().astype(np.int32) # NOTICE: calling flatten
    return (X, Y)

verbose = True
report_params = [
        'epoch',
        'main/loss',
        'validation/main/loss',
        'main/accuracy',
        'validation/main/accuracy',
        ]

iris_data = iris_data()
train, test = datasets.split_dataset_random(datasets.TupleDataset(*iris_data), 100)
train_iter = iterators.SerialIterator(train, batch_size=10, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=1, repeat=False, shuffle=False)

model = L.Classifier(MyModel())
optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')

if verbose:
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())

trainer.run()

def predict(model, v):
    z = model.predictor(np.array([v], dtype=np.float32))
    return F.softmax(z).data.argmax(axis=1)[0]

v = [1., 1., 1., 1.]
print "{} is classified to {}.".format(v, predict(model, v))
