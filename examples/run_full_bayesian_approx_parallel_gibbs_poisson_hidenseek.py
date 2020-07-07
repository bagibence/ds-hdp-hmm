# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3 (hidenseek)
#     language: python
#     name: hidenseek
# ---

# %%
# #%load_ext autoreload
# #%autoreload 2

# %%
## load packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from ds_hdp_hmm.gibbs_approx_parallel import *
from ds_hdp_hmm.util import *

# %%
from tqdm.auto import tqdm

# %%
from hidenseek.db_interface import *
connect_to_db()

# %%
seed_vec = [111,222,333,444,555,666,777,888,999,1000];

# %%
#seed = int((int(sys.argv[1])-1)%10); ## random seed
seed = np.random.seed(seed_vec[0]) ## fix randomness

# %%
## set params
p = 3;
v0_range = [0.01, 0.99];
v1_range = [0.01, 2]; ## [0.001, 10] if p=2
alpha0_a_pri = 1;
alpha0_b_pri = 0.01;
gamma0_a_pri = 2;
gamma0_b_pri = 1;
v0_num_grid = 30;
v1_num_grid = 30;

# %%
iters = 500
n_cores = 4
init_way = 'prior'
rlt_path = './'
L = 20

# %%
session = Session[13]

# %%
trains = [trial.get_spike_trains(100).transpose('time', 'neuron') for trial in session.seek_trials]
trains_np = [t.values for t in trains]

# %%
yt_all_ls = trains_np[:9]
yt_test_ls = trains_np[9:]

# %%
m_multi = yt_all_ls[0].shape[-1];

# %%
prior_params = {'lam_a_pri' : 1.0,
                     'lam_b_hyper_pri_shape' : 1.0,
                     'lam_b_hyper_pri_rate' : 1.0,
                     'm_multi' : m_multi}

# %%
mode = 'poisson'

# %%
# start gibbs
zt_sample = []
wt_sample = []
kappa_vec_sample = []
hyperparam_sample = []
post_sample = []
loglik_test_sample = []
pi_mat_sample = []
pi_init_sample = []
uniq_sample = []
lik_params_sample = []

# %%
if init_way == 'prior':
    rho0, rho1, alpha0, alpha0_init, gamma0, init_suff_stats, L, beta_vec, kappa_vec, pi_bar, pi_init, lik_params = init_gibbs_full_bayesian(p, v0_range, v1_range, alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri,prior_params, L,mode)

# %%
for it in tqdm(range(iters)):
    zt, wt, n_mat, num_1_vec, num_0_vec, n_ft, K, uniq, suff_stats = sample_zw_fmp(yt_all_ls, pi_bar, kappa_vec, pi_init, L, lik_params, mode, init_suff_stats, n_cores);
    
    ## sample hdp prior parameters
    kappa_vec = sample_kappa(num_1_vec, num_0_vec, rho0, rho1);
    m_mat, m_init = sample_m(n_mat, n_ft, beta_vec, alpha0, alpha0_init);
    beta_vec = sample_beta(m_mat, m_init, gamma0);
    pi_bar, pi_init = sample_pi(n_mat, n_ft, alpha0, alpha0_init, beta_vec);
    
    ## sample observation likelihood parameters
    lik_params = sample_lik_params(suff_stats, mode);
    
    ## sample hyperparams
    alpha0, alpha0_init = sample_alpha(m_mat, n_mat, alpha0, m_init, n_ft, alpha0_init, alpha0_a_pri, alpha0_b_pri);
    gamma0 = sample_gamma(K, m_mat, m_init, gamma0, gamma0_a_pri, gamma0_b_pri);
    rho0, rho1, posterior_grid = sample_rho(v0_range, v1_range, v0_num_grid, v1_num_grid, num_1_vec, num_0_vec,p);
    prior_params = sample_lam_b_pri(lik_params, prior_params, uniq, K);
    init_suff_stats = init_suff_stats_func(prior_params, L, mode);
    
    ## compute loglik
    if it%10 == 0:
        pi_all = (np.diag(kappa_vec)+np.matmul(np.diag(1-kappa_vec),pi_bar));
        loglik_test = compute_log_marginal_lik_poisson_fmp(L, pi_all, pi_init, suff_stats,yt_test_ls,n_cores);
        loglik_test_sample.append(loglik_test);
        post_sample.append(posterior_grid);
        pi_mat_sample.append(copy.deepcopy(pi_all));
        pi_init_sample.append(copy.deepcopy(pi_init));
        uniq_sample.append(copy.deepcopy(uniq));
        
        zt_sample.append(copy.deepcopy(zt));
        wt_sample.append(copy.deepcopy(wt));
        kappa_vec_sample.append(kappa_vec.copy());
        hyperparam_sample.append(np.array([alpha0, gamma0, rho0, rho1, alpha0_init]));
        lik_params_sample.append(copy.deepcopy(lik_params));
        
    if it%100 == 0:
        np.savez(rlt_path+file_name+'_full_bayesian_approx_rlt_'+str(seed) +'.npz', zt=zt_sample, wt=wt_sample, kappa=kappa_vec_sample, hyper=hyperparam_sample, loglik=loglik_test_sample, post=post_sample, pi_mat=pi_mat_sample, pi_init=pi_init_sample,uniq=uniq_sample,lik_params=lik_params_sample);

# %%
## save results
#seed = int((int(sys.argv[1])-1)%10);
np.savez(rlt_path+file_name+'_full_bayesian_approx_rlt_'+str(seed) +'.npz', zt=zt_sample, wt=wt_sample, kappa=kappa_vec_sample, hyper=hyperparam_sample, loglik=loglik_test_sample, post=post_sample, pi_mat=pi_mat_sample, pi_init=pi_init_sample,uniq=uniq_sample,lik_params=lik_params_sample);


# %%
plt.plot(np.stack(loglik_test_sample))

# %%
state_samples = xr.concat([xr.concat([xr.DataArray(arr, dims = ['time'], coords = {'time' : range(len(arr))})
                                      for arr in sample], dim = 'trial')
                           for sample in zt_sample], dim = 'iteration')

# %%
state_samples.sel(time = 0, trial = 0).values

# %%
len(post_sample[0])

# %%
_ = np.unique(state_samples.values.flatten())
_[~np.isnan(_)]

# %%
len(_[~np.isnan(_)])

# %%
for i in state_samples.iteration:
    print(len(np.unique(state_samples.sel(iteration = i).dropna('time').values.flatten())))

# %%
trans_mat_samples = xr.concat([xr.DataArray(mat, dims = ['from', 'to']) for mat in pi_mat_sample], dim = 'iteration')

# %%
fig, ax = plt.subplots(figsize = (5, 60), nrows = len(trans_mat_samples.iteration))
for i in trans_mat_samples.iteration:
    sns.heatmap(trans_mat_samples.sel(iteration = i), ax = ax[i])

# %%
post = np.stack(post_sample)

# %%
import seaborn as sns

# %%
import xarray as xr
import hvplot.xarray

# %%
post_xr = xr.DataArray(post, dims = ['a', 'b', 'c'])

# %%
post_xr = post_xr.assign_coords({'a' : range(50), 'b' : range(30), 'c' : range(30)})

# %%
post_xr

# %%
import holoviews as hv

# %%
fig, ax = plt.subplots(figsize = (5, 60), nrows = 50)
for i in range(post.shape[0]):
    sns.heatmap(post[i, :, :], ax = ax[i])

# %%
fig, ax = plt.subplots(figsize = (20, 5), ncols = 2)
sns.heatmap(zt[0, :, :], ax = ax[0])
sns.heatmap(zt[1, :, :], ax = ax[1])

# %%
for asdf in zt:
    print(asdf.shape)

# %%
first_trial_state_samples = np.stack([zti[0] for zti in zt_sample])

# %%
first_trial_state_samples.shape

# %%
states = [np.bincount(first_trial_state_samples[:, i], minlength = K).argmax()
          for i in range(first_trial_state_samples.shape[1])]

# %%
state_probs = [np.bincount(first_trial_state_samples[:, i], minlength = K)
          for i in range(first_trial_state_samples.shape[1])]

# %%
sns.heatmap(np.stack(state_probs).T)

# %%
from hidenseek.util.plotting import *
tab20, norm = get_tab20_and_norm(K)

# %%
sns.heatmap(np.array(states)[None, :], cmap = tab20, norm = norm)

# %%
fig, ax = plt.subplots(figsize = (20, 5), ncols = 2)
sns.heatmap(zt[:, 0, :], ax = ax[0])
sns.heatmap(zt[:, 1, :], ax = ax[1])

# %%
