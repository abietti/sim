import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import pickle

plt.style.use('ggplot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('private FRL')
    parser.add_argument('--name', default='d10_it1000')
    parser.add_argument('--sigma', type=float, default=0.001)
    parser.add_argument('--s', default='1')
    parser.add_argument('--d', default='10')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--skipn', type=int, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--best_only', action='store_true')
    args = parser.parse_args()

    table = []

    args.d = [int(d) for d in args.d.split(',')]
    args.s = [int(s) for s in args.s.split(',')]

    outname = os.path.join('figures', args.name)

    if len(args.d) > 1:
        ds_label = [(d, args.s[0], f'd = {d}') for d in args.d]
        caption = f's = {args.s[0]}'
        outname += f'_s{args.s[0]}'
    elif len(args.s) > 1:
        ds_label = [(args.d[0], s, f's = {s}') for s in args.s]
        caption = f'd = {args.d[0]}'
        outname += f'_d{args.d[0]}'
    else:
        d, s = args.d[0], args.s[0]
        ds_label = [(d, s, None)]
        caption = f'd = {d}, s = {s}'
        outname += f'_s{args.s[0]}_d{args.d[0]}'

    if args.best_only:
        caption += ' (best)'


    for fname in glob.glob(os.path.join('res', args.name, 'out_*.pkl')):
        res = pickle.load(open(fname, 'rb'))
        for config, out in res:
            # print(config['d'], config['s'], config['N'])
            if config['d'] not in args.d:
                continue
            if config['s'] not in args.s:
                continue
            if config['N'] != args.N:
                continue
            if config['ntr'] == args.skipn:
                continue
            table.append((config['ntr'], config['d'], config['s'], config['N'],
                config['rep'], out['m'],
                abs(out['m']), out['testloss'],
                np.array(out['testloss_ridge'])[:,1].min(),
                np.array(out['testloss_finetune'])[:,1].min()
            ))
            # print(table[-1])

    assert table is not None, 'table empty'

    df = pd.DataFrame(table, columns=['n', 'd', 's', 'N', 'rep', 'm', 'am', 'testloss', 'testloss_ridge', 'testloss_finetune'])

    plt.figure(figsize=(4, 3))
    # nm = [(n, np.max(mvals)) for n, mvals in thetas.items()]
    # nm.sort()
    # ns, ms = list(zip(*nm))
    # print(ns, ms)

    for d, s, label in ds_label:
        dff = df.loc[(df.d == d) & (df.s == s)].groupby(['n', 'rep']).am.max().groupby('n')
        if args.best_only:
            vals = list(dff.max().items())
        else:
            vals = list(dff.mean().items())
        std = np.array(dff.std())
        # vals.sort()
        ns, vs = list(zip(*vals))
        vs = np.array(vs)
        plt.semilogx(ns, vs, label=label)
        if not args.best_only:
            plt.fill_between(ns, vs - std, vs + std, alpha=.1)

    plt.title(caption)
    plt.xlabel('n')
    plt.ylabel('|m|')
    if len(ds_label) > 1:
        plt.legend()
    if args.save:
        plt.savefig(outname + '_corr.pdf', pad_inches=0, bbox_inches='tight')

    plt.figure(figsize=(4, 3))
    # nloss = [(n, np.min(lossvals)) for n, lossvals in losses.items()]
    # nloss.sort()
    # ns, loss = list(zip(*nloss))
    # plt.loglog(ns, np.array(loss) - args.sigma**2)

    for d, s, label in ds_label:
        dff = df.loc[(df.d == d) & (df.s == s)].groupby(['n', 'rep']).testloss.min().groupby('n')
        if args.best_only:
            vals = list(dff.min().items())
        else:
            vals = list(dff.mean().items())
        std = np.array(dff.std())
        # vals.sort()
        ns, vs = list(zip(*vals))
        vs = np.array(vs) - args.sigma**2
        plt.loglog(ns, vs, label=label)
        if not args.best_only:
            plt.fill_between(ns, vs - std, vs + std, alpha=.1)

    plt.title(caption)
    plt.xlabel('n')
    plt.ylabel('excess risk')
    if len(ds_label) > 1:
        plt.legend()
    if args.save:
        plt.savefig(outname + '_loss.pdf', pad_inches=0, bbox_inches='tight')

    plt.figure(figsize=(4, 3))

    for d, s, label in ds_label:
        dff = df.loc[(df.d == d) & (df.s == s)].groupby(['n', 'rep']).testloss_ridge.min().groupby('n')
        if args.best_only:
            vals = list(dff.min().items())
        else:
            # vals = list(dff.mean().items())
            # std = np.array(dff.std())
            vals = list(dff.apply(lambda x: np.sort(x)[:-1].mean()).items())
            std = np.array(dff.apply(lambda x: np.sort(x)[:-1].std()))
        # vals.sort()
        ns, vs = list(zip(*vals))
        vs = np.array(vs) - args.sigma**2
        plt.loglog(ns, vs, label=label)
        if not args.best_only:
            plt.fill_between(ns, vs - std, vs + std, alpha=.1)

    plt.title('ridge, ' + caption)
    plt.xlabel('n')
    plt.ylabel('excess risk')
    if len(ds_label) > 1:
        plt.legend()
    if args.save:
        plt.savefig(outname + '_loss_ridge.pdf', pad_inches=0, bbox_inches='tight')
