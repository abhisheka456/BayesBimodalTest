import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import itertools
from emcee import PTSampler
import seaborn as sns


class BayesBimodalTest():
    """ A Test module for bimodality, and more general N-modality in a data set

    Performs an MCMC parameter estimation for a list `Ns` of N values where N
    is the number of components in the mixture model. For each N, the evidence
    is calculated using thermodynamic integration.

    Params
    -----
    ntemps : int
        The number of temperatures to use in the MCMC fitting - increase this
        to reduce the error on the evidence estimate
    betamin : int of list of ints
        The minimum beta = 1/T to use in fitting. It should be checked,
        in the diagnostic plot, that this is small enough to capture the
        high-temperature behaviour in the numerical integration. If a list,
        then the length should match that of Ns since each N can take a
        different minimum temperature.
    nburn0, nburn, nprod : int
        The number of steps to take in the three stages of the MCMC fitting
    nwalkers : int
        The number of walkers to use

    """

    def __init__(self, data, ntemps=20, betamin=-22,
                 nburn0=100, nburn=100, nprod=100, nwalkers=100):
        self.data = data
        self.data_min = np.min(data)
        self.data_max = np.max(data)
        self.data_std = np.std(data)
        self.ntemps = ntemps
        self.betas = np.logspace(0, betamin, ntemps)
        self.nburn0 = nburn0
        self.nburn = nburn
        self.nprod = nprod
        self.nwalkers = nwalkers
        self.saved_data = {}

    def log_unif(self, x, a, b):
        if (x < a) or (x > b):
            return -np.inf
        else:
            return np.log(1./(b-a))

    def get_uniform_prior_lims(self, key):
        if key == "mu":
            return [self.data_min, self.data_max]
        if key == "sigma":
            return [1e-20*self.data_std, 10*self.data_std]
        if key == "p":
            return [0, 1]

    def create_initial_p0(self, N):
        """ Generates a sensible starting point for the walkers based on the
            prior and label degenerecy checks
        """

        param_keys = ['mu'] * N + ['sigma'] * N + ['p'] * (N-1)
        p0 = []
        for k in range(self.ntemps):
            p0_1 = []
            for j in range(self.nwalkers):
                p0_2 = []
                for i, key in enumerate(param_keys):
                    component = np.mod(i, N)
                    [lower, upper] = self.get_uniform_prior_lims(key)
                    dp = (upper - lower) / float(N)
                    component_range = [lower + component*dp,
                                       lower + (component+1)*dp]
                    p0_2.append(np.random.uniform(*component_range))
                p0_1.append(p0_2)
            p0.append(p0_1)

        return p0

    def get_new_p0(self, sampler, ndim, scatter_val=1e-3):
        pF = sampler.chain[:, :, -1, :].reshape(
            self.ntemps, self.nwalkers, ndim)[0, :, :]
        lnp = sampler.lnprobability[:, :, -1].reshape(
            self.ntemps, self.nwalkers)[0, :]
        p = pF[np.argmax(lnp)]
        p0 = [[p + scatter_val * p * np.random.randn(ndim)
              for i in xrange(self.nwalkers)] for j in xrange(self.ntemps)]
        return p0

    def logp_Nmodal(self, params):
        N = (len(params) + 1) / 3
        mus = params[:N]
        ps = params[2*N:]
        if any(np.diff(mus) < 0):
            return -np.inf
        if np.sum(ps) > 1:
            return -np.inf

        logp = 0
        logp += np.sum([self.log_unif(p, *self.get_uniform_prior_lims('mu'))
                        for p in mus])
        logp += np.sum([self.log_unif(p, *self.get_uniform_prior_lims('sigma'))
                        for p in params[N:2*N]])
        logp += np.sum([self.log_unif(p, *self.get_uniform_prior_lims('p'))
                        for p in params[2*N:]])
        return logp

    def logl_Nmodal(self, params, data):
        N = (len(params) + 1) / 3
        mu = np.array(params[:N])
        sigma = np.array(params[N:2*N])
        p = params[2*N:]
        p = np.append(p, (1 - np.sum(p)))
        res = (data.reshape((len(data), 1)) - mu.T)
        r = np.log(np.sum(p/(sigma*np.sqrt(2*np.pi)) *
                          np.exp(-res**2/(2*sigma**2)), axis=1))
        return np.sum(r)

    def fit_Nmodal(self, N):
        """ Fit the N-modal distribution

        params is a 3N-1 vector, with the first N as the mu's, the second N
        the sigmas, and the last N-1 being the p's.
        """

        saved_data = {}
        ndim = N*3 - 1
        sampler = PTSampler(self.ntemps, self.nwalkers, ndim, self.logl_Nmodal,
                            self.logp_Nmodal, loglargs=[self.data],
                            betas=self.betas)
        p0 = self.create_initial_p0(N)

        if self.nburn0 != 0:
            out = sampler.run_mcmc(p0, self.nburn0)
            saved_data["chains0"] = sampler.chain[0, :, : , :]
            p0 = self.get_new_p0(sampler, ndim)
            sampler.reset()
        else:
            saved_data["chains0"] = None
        out = sampler.run_mcmc(p0, self.nburn + self.nprod)
        saved_data["chains"] = sampler.chain[0, :, :, :]

        saved_data["sampler"] = sampler
        saved_data["samples"] = sampler.chain[0, :, self.nburn:, :].reshape(
            (-1, ndim))
        self.saved_data['N{}'.format(N)] = saved_data
        self.summarise_posteriors(N)

    def summarise_posteriors(self, N):
        saved_data = self.saved_data['N{}'.format(N)]
        saved_data['mus'] = [
            np.mean(saved_data['samples'][:, i]) for i in range(N)]
        saved_data['sigmas'] = [
            np.mean(saved_data['samples'][:, i]) for i in range(N, 2*N)]
        saved_data['ps'] = [
            np.mean(saved_data['samples'][:, i]) for i in range(2*N, 3*N-1)]
        saved_data['ps'].append(1-np.sum(saved_data['ps']))
        self.saved_data['N{}'.format(N)] = saved_data

    def diagnostic_plot(self, Ns=None, fname="diagnostic.png",
                        trace_line_width=0.1, hist_line_width=1.5):

        if type(Ns) is int:
            Ns = [Ns]

        fig = plt.figure(figsize=(8, 11))
        if self.ntemps > 1:
            nrows = 5
        else:
            nrows = 4

        colors = [sns.xkcd_rgb["pale red"],
                  sns.xkcd_rgb["medium green"],
                  sns.xkcd_rgb["denim blue"],
                  sns.xkcd_rgb["seafoam"],
                  sns.xkcd_rgb["rich purple"]
                  ]

        burn0s = np.arange(0, self.nburn0)
        prods = np.arange(self.nburn0, self.nburn0+self.nburn + self.nprod)

        ax00 = plt.subplot2grid((nrows, 2), (0, 0), colspan=2)
        ax10 = plt.subplot2grid((nrows, 2), (1, 0))
        ax11 = plt.subplot2grid((nrows, 2), (1, 1))
        ax20 = plt.subplot2grid((nrows, 2), (2, 0))
        ax21 = plt.subplot2grid((nrows, 2), (2, 1))
        ax30 = plt.subplot2grid((nrows, 2), (3, 0))
        ax31 = plt.subplot2grid((nrows, 2), (3, 1))
        Laxes = [ax10, ax20, ax30]
        Raxes = [ax11, ax21, ax31]

        ax00.hist(self.data, bins=50, color="b", histtype="step", normed=True)
        x_plot = np.linspace(self.data.min(), self.data.max(), 100)

        for i, N in enumerate(Ns):

            c = colors[i]
            saved_data = self.saved_data['N{}'.format(N)]
            zi = zip(saved_data['ps'], saved_data['mus'], saved_data['sigmas'])
            for j, (p, mu, sigma) in enumerate(zi):
                ax00.plot(x_plot, p*ss.norm.pdf(x_plot, mu, sigma),
                          color=c, label="$N{}_{}$".format(N, j))

            for j, (lax, rax) in enumerate(zip(Laxes, Raxes)):
                if j == 2:
                    krange = N-1
                else:
                    krange = N
                for k in range(krange):
                    lax.hist(saved_data['samples'][:, j*N+k], bins=50,
                             linewidth=hist_line_width, histtype="step",
                             color=c)

                    if saved_data['chains0'] is not None:
                        rax.plot(burn0s, saved_data['chains0'][:, :, j*N+k].T,
                                 lw=trace_line_width, color=c)
                    rax.plot(prods, saved_data['chains'][:, :, j*N+k].T,
                             lw=trace_line_width, color=c)

        ax00.set_xlabel("Data")
        ax00.legend(loc=2, frameon=False)
        ax10.set_title("Mean posterior")
        ax11.set_title("Mean trace")
        ax20.set_title("Sigma posterior")
        ax21.set_title("Sigma trace")
        ax30.set_title("p posterior")
        ax31.set_title("p trace")

        for ax in Raxes:
            lw = 1.1
            ax.axvline(self.nburn0, color="k", lw=lw, alpha=0.4)
            ax.axvline(self.nburn0+self.nburn, color="k", lw=lw, alpha=0.4)
            ax.axvline(self.nburn0+self.nburn+self.nprod, color="k",
                       lw=lw, alpha=0.4)

        if self.ntemps > 1:
            ax40 = plt.subplot2grid((nrows, 2), (nrows-1, 0), colspan=2)
            for i, N in enumerate(Ns):
                betas = self.betas(self.Ns.index(N))
                alllnlikes = self.saved_data["N{}".format(N)][
                    'sampler'].lnlikelihood[:, :, self.nburn:]
                mean_lnlikes = np.mean(np.mean(alllnlikes, axis=1), axis=1)
                ax40.semilogx(betas, mean_lnlikes, "-o", color=colors[i])
                ax40.set_title("Linear thermodynamic integration")

        fig.tight_layout()
        fig.savefig(fname)

    def OccamFactor(self):
        Umu = self.get_uniform_prior_lims('mu')
        Usigma = self.get_uniform_prior_lims('sigma')
        occams_factor = (np.log10(Umu[1] - Umu[0])
                         + np.log10(Usigma[1] - Usigma[0])
                         + np.log10(1))
        print "Occams factor is {}".format(occams_factor)

    def BayesFactor(self, print_result=True):
        evi_err = []
        for N in self.Ns:
            name = "N{}".format(N)
            sampler = self.saved_data[name]['sampler']
            lnevidence, lnevidence_err = sampler.thermodynamic_integration_log_evidence()
            log10evidence = lnevidence/np.log(10)
            log10evidence_err = lnevidence_err/np.log(10)
            evi_err.append((N, log10evidence, log10evidence_err))

        if print_result:
            for mA, mB in itertools.combinations(evi_err, 2):
                mA_deg, mA_evi, mA_err = mA
                mA_name = "{}-modal".format(mA_deg)
                mB_deg, mB_evi, mB_err = mB
                mB_name = "{}-modal".format(mB_deg)
                bf = mB_evi - mA_evi
                bf_err = np.sqrt(mA_err**2 + mB_err**2)
                print "log10 Bayes Factor ({}, {}) = {} +/- {}".format(
                    mB_name, mA_name, bf, bf_err)

        self.evi_err = evi_err
