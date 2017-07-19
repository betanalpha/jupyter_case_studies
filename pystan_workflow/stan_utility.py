import pystan
import pickle
import numpy

def check_div(sampler_params):
    """Check transitions and ended with a divergence"""
    divergent = [x for y in sampler_params for x in y['divergent__']]
    n = sum(divergent)
    N = len(divergent)
    print('{} of {} iterations ended with a divergence({}%)'.format(n, N,
            100 * n / N))
    if n > 0:
        print('Try running with larger adapt_delta to remove the divergences')

def check_treedepth(sampler_params, max_depth = 10):
    """Check transitions that ended prematurely due to maximum treedepth limit"""
    depths = [x for y in sampler_params for x in y['treedepth__']]
    n = sum(1 for x in depths if x == max_depth)
    N = len(depths)
    print(('{} of {} iterations saturated the maximum tree depth of {}'
            + '({}%)').format(n, N, max_depth, 100 * n / N))
    if n > 0:
        print('Run again with max_depth set to a larger value to avoid saturation')

def check_energy(sampler_params):
    """Checks the energy Bayesian fraction of missing information (E-BFMI)"""
    for chain_num, s in enumerate(sampler_params):
        energies = s['energy__']
        numer = sum((energies[i] - energies[i - 1])**2 for i in range(1, len(energies))) / len(energies)
        denom = numpy.var(energies)
        if numer / denom < 0.2:
            print('Chain {}: E-BFMI = {}'.format(chain_num, numer / denom))
            print('E-BFMI below 0.2 indicates you may need to reparameterize your model')

def partition_div(fit):
    """ Returns parameter arrays separted into divergent and non-divergent transitions"""
  
    # Get parameter names and shapes
    params = fit.extract(inc_warmup=False)
    names = params.keys()
    sizes = [0] * len(names)
    for n, name in enumerate(names):
        if type(params[name][0]) == numpy.float64:
            sizes[n] = 1
        elif type(params[name][0]) == numpy.ndarray:
            sizes[n] = len(params[name][0])
        else:
            print("Type not recognized!")

    start_idx = [0] * len(names)
    for n in range(1, len(names)):
        start_idx[n] = start_idx[n - 1] + sizes[n - 1]

    # Reformat unpermuted parameter samples
    unpermuted_params = fit.extract(permuted=False, inc_warmup=False)
    T = len(unpermuted_params)
    C = len(unpermuted_params[0])
    N = T * C

    reformat_params = {}
    for n, name in enumerate(names):
        if sizes[n] == 1:
            reformat_params[name] = [None] * N
        else:
            reformat_params[name] = [ [None] * (sizes[n]) for _ in xrange(N)]

    for c in range(C):
        for t in range(T):
            vals = unpermuted_params[t][c]
            for n, name in enumerate(names):
                if sizes[n] == 1:
                    reformat_params[name][c * T + t] = vals[start_idx[n]]
                else:
                    reformat_params[name][c * T + t] = vals[start_idx[n]:start_idx[n] + sizes[n]]

    # Grab divergences from sampler parameters
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    div = reduce(lambda x, y: x + y['divergent__'].tolist(), sampler_params, [])

    # Append divergences to reformated parameters
    reformat_params[u'divergent'] = div

    div_idx = [idx for idx, d in enumerate(div) if d == 1]
    div_params = dict((key, [val[i] for i in div_idx]) for key, val in reformat_params.iteritems())

    nondiv_idx = [idx for idx, d in enumerate(div) if d == 0]
    nondiv_params = dict((key, [val[i] for i in nondiv_idx]) for key, val in reformat_params.iteritems())

    return nondiv_params, div_params

def compile_model(filename, model_name=None, **kwargs):
    """This will automatically cache models - great if you're just running a
    script on the command line.

    See http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html"""
    from hashlib import md5

    with open(filename) as f:
        model_code = f.read()
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        if model_name is None:
            cache_fn = 'cached-model-{}.pkl'.format(code_hash)
        else:
            cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
        try:
            sm = pickle.load(open(cache_fn, 'rb'))
        except:
            sm = pystan.StanModel(model_code=model_code)
            with open(cache_fn, 'wb') as f:
                pickle.dump(sm, f)
        else:
            print("Using cached StanModel")
        return sm
