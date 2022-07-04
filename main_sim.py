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
    # u0 = 1.
    # u = u - u0
    return np.clip(u + 2, a_min=0, a_max=None) - 2 * np.clip(u + 1, a_min=0, a_max=None) + 2 * np.clip(u - 1, a_min=0, a_max=None) - np.clip(u - 2, a_min=0, a_max=None)

def hermite_coef(k):
    u = np.random.randn(1000000)
    return np.mean(teacher_1d(u) * hermite(k, u))

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
    return jnp.dot(c, jnp.clip(px - b, a_min=0))

net = vmap(net, (None, None, None, 0))

def loss(c, theta, b, x, y, lmbda):
    pred = net(c, theta, b, x)
    return jnp.mean((y - pred) ** 2) + lmbda * jnp.dot(c, c)

lossgrads = jit(grad(loss, (0, 1)))


n_total = 30000

data_cache = {}
def get_data(seed, d, sigma, s=None):
    key = (seed, d, sigma, s)
    if key in data_cache:
        return data_cache[key]

    np.random.seed(seed)
    X = np.random.randn(n_total, d)
    y = fstar(X, s=s) + sigma * np.random.randn(n_total)
    data_cache[key] = (X, y)
    return X, y


def run(config):
    d = config['d']
    ntr = config['ntr']
    N = config['N']
    # N = 10 * int(np.sqrt(ntr))
    lmbda = config['lmbda']
    step = config['step']
    tau = config['tau']
    X, y = get_data(config['seed'], d, config['sigma'], config.get('s'))

    Xtr, Xte = X[:ntr], X[-10000:]
    ytr, yte = y[:ntr], y[-10000:]

    theta = np.random.randn(d)
    theta /= np.linalg.norm(theta)
    c = np.random.randn(N) / np.sqrt(N)
    b = tau * np.random.randn(N)

    ms = []
    test_losses = []
    for it in range(config['iters']):
        gc, gth = lossgrads(c, theta, b, Xtr, ytr, lmbda)
        theta -= 100 * step * gth
        theta /= np.linalg.norm(theta)
        c -= step * gc

        testpreds = net(c, theta, b, Xte)
        testloss = np.mean((yte - testpreds) ** 2)
        print(abs(theta[0]), testloss)
        ms.append(abs(theta[0]))
        test_losses.append(testloss)

    idx = np.argmin(test_losses)
    return ms[idx], test_losses[idx]


grid = {
    'step': [1e-4, 1e-3],
    'N': [10, 100, 1000, 2000],
    # 'N': [5000],
    'lmbda': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],
    'ntr': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
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

    args = parser.parse_args()

    if not args.interactive:
        os.makedirs(os.path.join('res', args.name), exist_ok=True)
        outfile = os.path.join('res', args.name, f'out_{args.task_offset + args.task_id}.pkl')

    if args.interactive:
        grid = {'step': [args.step], 'N': [args.N], 'lmbda': [args.lmbda], 'ntr': [args.ntr]}

    from itertools import product

    config = {'d': args.d, 'iters': 1000, 'sigma': 0.001, 'seed': 42, 'tau': 10., 's': args.s}

    res = []
    for i, vals in enumerate(product(*grid.values())):
        if i % args.num_tasks != (args.task_id - 1):
            continue
        kv = dict(zip(grid.keys(), vals))
        print(kv, flush=True)
        config.update(kv)

        m, testloss = run(config)
        print(m, testloss)

        res.append((config, m, testloss))
        if not args.interactive:
            pickle.dump(res, open(outfile, 'wb'))


