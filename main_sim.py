import argparse
import numpy as np
import os
import jax.numpy as jnp
import jax
import pickle
import scipy as sp
from jax import vmap, jit, grad


def hermite(k, u):
    return sp.special.eval_hermitenorm(k, u) / np.sqrt(np.math.factorial(k))

def teacher_1d(u):
    # return hermite(3, u)
    u0 = 1.
    u = u - u0
    return np.clip(u + 2, a_min=0, a_max=None) - 2 * np.clip(u + 1, a_min=0, a_max=None) + 2 * np.clip(u - 1, a_min=0, a_max=None) - np.clip(u - 2, a_min=0, a_max=None)

def hermite_coef(k):
    u = np.random.randn(1000000)
    return np.mean(teacher_1d(u) * hermite(k, u))

def student(c, b, u):
    return jnp.dot(c, jnp.clip(u - b, a_min=0)) / jnp.sqrt(b.shape[0])

student = vmap(student, (None, None, 0))

def hermite_coef_student(c, b, k):
    u = np.random.randn(1000000)
    return np.mean(student(c, b, u) * hermite(k, u))

def fstar(x, s=None):
    px = x[:,0] # project
    y = teacher_1d(px)
    if s is not None:
        for k in range(s):
            coef = hermite_coef(k)
            if abs(coef) > 0.001:
                y -= coef * hermite(k, px)
    return y

# fstar = jit(vmap(teacher))

def net(c, theta, b, x):
    px = jnp.dot(theta, x)
    return jnp.dot(c, jnp.clip(px - b, a_min=0)) / jnp.sqrt(b.shape[0])

net = vmap(net, (None, None, None, 0))

def features(theta, b, x):
    px = jnp.dot(theta, x)
    return jnp.clip(px - b, a_min=0) / jnp.sqrt(b.shape[0])

features = vmap(features, (None, None, 0))

def loss(c, theta, b, x, y, lmbda):
    pred = net(c, theta, b, x)
    return jnp.mean((y - pred) ** 2) + lmbda * jnp.dot(c, c)

lossgrads = jit(grad(loss, (0, 1)))


n_total = 100000
n_test = 10000

data_cache = {}
def get_data(seed, d, sigma, s=None):
    key = (seed, d, sigma, s)
    if key in data_cache:
        return data_cache[key]

    np.random.seed(seed)
    X = np.random.randn(2 * n_total, d)
    y = fstar(X, s=s) + sigma * np.random.randn(2 * n_total)
    data_cache[key] = (X, y)
    return X, y


def run(config, args):
    d = config['d']
    ntr = config['ntr']
    N = config['N']
    # N = 10 * int(np.sqrt(ntr))
    lmbda = config['lmbda']
    step = config['step']
    tau = config['tau']
    X, y = get_data(config['seed'] + config.get('rep', 0), d, config['sigma'], config.get('s'))
    s = config.get('s', 1)

    assert ntr + n_test <= n_total
    Xtr, Xte = X[:ntr], X[n_total-10000:n_total]
    ytr, yte = y[:ntr], y[n_total-10000:n_total]

    Xtune = X[n_total:n_total+ntr]
    ytune = y[n_total:n_total+ntr]

    theta = np.random.randn(d)
    theta /= np.linalg.norm(theta)
    # c = np.random.randn(N) / np.sqrt(N)
    c = np.random.randn(N)
    b = tau * np.random.randn(N)

    if theta[0] < 0:
        theta *= -1
    # if hermite_coef(s) * hermite_coef_student(c, b, s) * theta[0] ** s < 0:
    if hermite_coef(s) * hermite_coef_student(c, b, s) < 0:
        c *= -1
    print(theta[0], hermite_coef(s), hermite_coef_student(c, b, s))

    def ridge_eval(theta, X, y, lmbda):
        n = X.shape[0]
        phi = features(theta, b, X)
        c_ridge = np.linalg.solve(phi.T.dot(phi) + n * lmbda * np.eye(N), phi.T.dot(y))
        testpreds_ridge = net(c_ridge, theta, b, Xte)
        testloss_ridge = np.mean((yte - testpreds_ridge) ** 2)
        return testloss_ridge

    ms = []
    test_losses = []
    for it in range(config['iters']):
        if args.eval_delta > 0 and it % args.eval_delta == 0:
            testpreds = net(c, theta, b, Xte)
            testloss = np.mean((yte - testpreds) ** 2)
            # print(np.linalg.norm(c))
            testloss_ridge = ridge_eval(theta, Xtr, ytr, 0.001 * lmbda)
            testloss_ridge_finetune = ridge_eval(theta, Xtune, ytune, 0.001 * lmbda)
            # print(hermite_coef(s), hermite_coef_student(c, b, s))
            print(it, theta[0], testloss, testloss_ridge, testloss_ridge_finetune)
            ms.append(abs(theta[0]))
            test_losses.append(testloss)

        gc, gth = lossgrads(c, theta, b, Xtr, ytr, lmbda)
        # if it < 10000:
        theta -= 100. * step * gth
        # else:
        #     theta -= 20. * step * gth
        theta /= np.linalg.norm(theta)
        if it >= 500:
            c -= step * gc

    # idx = np.argmin(test_losses)
    # return ms[idx], test_losses[idx]
    testpreds = net(c, theta, b, Xte)
    testloss = np.mean((yte - testpreds) ** 2)
    testloss_ridge, testloss_finetune = [], []
    for lmb in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        testloss_ridge.append((lmb, ridge_eval(theta, Xtr, ytr, lmb)))
        testloss_finetune.append((lmb, ridge_eval(theta, Xtune, ytune, lmb)))
    return {'m': theta[0], 'testloss': testloss, 'testloss_ridge': testloss_ridge, 'testloss_finetune': testloss_finetune}


grid = {
    'd': [10, 20, 50, 100],
    's': [1, 2, 3, 4],
    'step': [1e-4, 1e-3, 1e-2],
    'N': [10, 30, 100],
    # 'N': [5000],
    # 'lmbda': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],
    'lmbda': [1e-4, 1e-3, 1e-2, 1e-1],
    'ntr': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    'rep': list(range(10)),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('single index model')
    parser.add_argument('task_id', type=int)
    parser.add_argument('num_tasks', type=int)
    parser.add_argument('--task_offset', type=int, default=0)
    parser.add_argument('--name', default='test')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--ntr', type=int, default=1000)
    parser.add_argument('--lmbda', type=float, default=1e-4)
    parser.add_argument('--step', type=float, default=0.001)
    parser.add_argument('--s', type=int, default=None)
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_delta', type=int, default=-1)

    args = parser.parse_args()

    if not args.interactive:
        os.makedirs(os.path.join('res', args.name), exist_ok=True)
        outfile = os.path.join('res', args.name, f'out_{args.task_offset + args.task_id}.pkl')

    if args.interactive:
        grid = {'d': [args.d], 's': [args.s], 'step': [args.step], 'N': [args.N], 'lmbda': [args.lmbda], 'ntr': [args.ntr]}

    from itertools import product

    config = {'iters': args.iters, 'sigma': 0.01, 'seed': args.seed, 'tau': 10.}

    res = []
    for i, vals in enumerate(product(*grid.values())):
        if i % args.num_tasks != (args.task_id - 1):
            continue
        kv = dict(zip(grid.keys(), vals))
        print(kv, flush=True)
        cfg = config.copy()
        cfg.update(kv)

        # m, testloss = run(config, args)
        # print(m, testloss)
        out = run(cfg, args)
        print(out['m'], out['testloss'])

        res.append((cfg, out))
        if not args.interactive:
            pickle.dump(res, open(outfile, 'wb'))


