import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import itertools
from emcee import PTSampler
from scipy.special import erf
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
    p_lower_bound: float [0, 1]
        The lower bound for the p prior. Default is 0, but if one wants to
        force modes to exist this can be set to a positive float less than
        one, for example `p = 0.1`.

    """

    def __init__(self, data, ntemps=20, betamin=-22, nburn0=100, nburn=100,
                 nprod=100, nwalkers=100, p_lower_bound=0):
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
        self.p_lower_bound = p_lower_bound
        self.fitted_Ns = []

    def saved_data_name(self, N, skew):
        if skew is False:
            return "N{}".format(N)
        elif skew is True:
            return "NS{}".format(N)

    def log_unif(self, x, a, b):
        if (x < a) or (x > b):
            return -np.inf
        else:
            return np.log(1./(b-a))

    def log_norm(self, x, mu, sigma):
        return -.5*((x-mu)**2/sigma**2 + np.log(sigma**2*2*np.pi))

    def get_uniform_prior_lims(self, key):
        if key == "mu":
            return [self.data_min, self.data_max]
        if key == "sigma":
            return [1e-20*self.data_std, 2*self.data_std]
        if key == "p":
            return [self.p_lower_bound, 1]
        if key == "alpha":  # Used to generate p0
            return [-10*self.data_std, 10*self.data_std]

    def get_normal_prior_lims(self, key):
        if key == "alpha":
            return [0, 10*self.data_std]

    def create_initial_p0(self, N, skew=False):
        """ Generates a sensible starting point for the walkers based on the
            prior and label degenerecy checks
        """

        if skew is False:
            param_keys = ['mu'] * N + ['sigma'] * N + ['p'] * (N-1)
        else:
            param_keys = ['mu'] * N + ['sigma'] * N + ['alpha'] * N + ['p'] * (N-1)
        p0 = []
        for k in range(self.ntemps):
            p0_1 = []
            for j in range(self.nwalkers):
                p0_2 = []
                for i, key in enumerate(param_keys):
                    if key == "mu":
                        component = np.mod(i, N)
                        [lower, upper] = self.get_uniform_prior_lims(key)
                        dp = (upper - lower) / float(N)
                        component_range = [lower + component*dp,
                                           lower + (component+1)*dp]
                        p0_2.append(np.random.uniform(*component_range))
                    else:
                        p0_2.append(np.random.uniform(*self.get_uniform_prior_lims(key)))
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
        if np.sum(ps) > 1-self.p_lower_bound:
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

        name = self.saved_data_name(N, False)
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
        self.saved_data[name] = saved_data
        self.summarise_posteriors(N)

    def logp_SkewNmodal(self, params):
        N = (len(params) + 1) / 4
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
        logp += np.sum([self.log_norm(p, *self.get_normal_prior_lims('alpha'))
                        for p in params[2*N:3*N]])
        logp += np.sum([self.log_unif(p, *self.get_uniform_prior_lims('p'))
                        for p in params[3*N:]])

        return logp

    def logl_SkewNmodal(self, params, data):
        N = (len(params) + 1) / 3
        mu = np.array(params[:N])
        sigma = np.array(params[N:2*N])
        alpha = np.array(params[2*N:3*N])
        p = params[3*N:]
        p = np.append(p, (1 - np.sum(p)))
        res = (data.reshape((len(data), 1)) - mu.T)
        arg = alpha * res / (np.sqrt(2) * sigma)
        r = np.log(np.sum(2*p/(sigma*np.sqrt(2*np.pi)) *
                          np.exp(-res**2/(2*sigma**2)) *
                          .5*(1+erf(arg)), axis=1))
        return np.sum(r)

    def fit_SkewNmodal(self, N):
        """ Fit the N-modal distribution

        params is a 3N-1 vector, with the first N as the mu's, the second N
        the sigmas, and the last N-1 being the p's.
        """

        name = self.saved_data_name(N, True)
        saved_data = {}
        ndim = N*4 - 1
        sampler = PTSampler(self.ntemps, self.nwalkers, ndim, self.logl_SkewNmodal,
                            self.logp_SkewNmodal, loglargs=[self.data],
                            betas=self.betas)
        p0 = self.create_initial_p0(N, skew=True)

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
        self.saved_data[name] = saved_data
        self.summarise_posteriors(N, skew=True)

    def summarise_posteriors(self, N, skew=False):
        name = self.saved_data_name(N, skew)
        saved_data = self.saved_data[name]
        saved_data['mus'] = [
            np.mean(saved_data['samples'][:, i]) for i in range(N)]
        saved_data['sigmas'] = [
            np.mean(saved_data['samples'][:, i]) for i in range(N, 2*N)]
        if skew:
            saved_data['alphas'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(2*N, 3*N)]
            saved_data['ps'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(3*N, 4*N-1)]
            saved_data['ps'].append(1-np.sum(saved_data['ps']))
        else:
            saved_data['ps'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(2*N, 3*N-1)]
            saved_data['ps'].append(1-np.sum(saved_data['ps']))

        self.saved_data[name] = saved_data

    def pdf(self, x, mu, sigma, p, alpha=0):
        phi = p*ss.norm.pdf(x, mu, sigma)
        Phi = .5 * (1 + erf(alpha * (x - mu)/sigma))
        return 2 * phi * Phi

    def diagnostic_plot(self, Ns=None, skews=None, fname="diagnostic.png",
                        trace_line_width=0.1, hist_line_width=1.5, 
                        separate=False):

        if self.ntemps > 1:
            nrows = 5
        else:
            nrows = 4

        if Ns is None and skews is None:
            Ns = []
            skews = []
            for key in self.fitted_Ns:
                Ns.append(int(key[-1]))
                if "S" in key:
                    skews.append(True)
                else:
                    skews.append(False)
        if type(Ns) is int:
            Ns = [Ns]

        if skews:
            if len(skews) != len(Ns):
                raise ValueError("len(skews) == len(Ns)")
            nrows += 1
            skewed = True
        else:
            skews = [False] * len(Ns)
            skewed = False

        fig = plt.figure(figsize=(8, 11))

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

        if skewed:
            ax40 = plt.subplot2grid((nrows, 2), (4, 0))
            ax41 = plt.subplot2grid((nrows, 2), (4, 1))
            Laxes.append(ax40)
            Raxes.append(ax41)


        ax00.hist(self.data, bins=50, color="b", histtype="step", normed=True)
        x_plot = np.linspace(self.data.min(), self.data.max(), 100)

        for i, (N, skew) in enumerate(zip(Ns, skews)):
            y = np.zeros(len(x_plot))
            name = self.saved_data_name(N, skew)
            c = colors[i]
            saved_data = self.saved_data[name]
            if skew:
                alphas = saved_data['alphas']
            else:
                alphas = [0] * N
            zi = zip(saved_data['ps'], saved_data['mus'], saved_data['sigmas'],
                     alphas)

            for j, (p, mu, sigma, alpha) in enumerate(zi):
                if separate:
                    ax00.plot(x_plot, self.pdf(x_plot, mu, sigma, p, alpha),
                              color=c, label="$N{}_{}$".format(N, j))
                else:
                    y += self.pdf(x_plot, mu, sigma, p, alpha)
            if separate is False:
                ax00.plot(x_plot, y, linestyle="-", color=c,
                          label="$N{}_{}$".format(N, j))

            for j, (lax, rax) in enumerate(zip(Laxes, Raxes)):
                if skew is False and skewed is True:
                    if j == 2:
                        continue # Skip alpha
                    if j > 2:
                        j -= 1
                if j == len(Laxes)-1:
                    krange = N-1
                elif skew is False and skewed is True and j == len(Laxes)-2:
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
        if skewed:
            ax30.set_title(r"$\alpha$ posterior")
            ax31.set_title(r"$\alpha$ trace")
        Laxes[-1].set_title("p posterior")
        Raxes[-1].set_title("p trace")

        for ax in Raxes:
            lw = 1.1
            ax.axvline(self.nburn0, color="k", lw=lw, alpha=0.4)
            ax.axvline(self.nburn0+self.nburn, color="k", lw=lw, alpha=0.4)
            ax.axvline(self.nburn0+self.nburn+self.nprod, color="k",
                       lw=lw, alpha=0.4)

        if self.ntemps > 1:
            ax40 = plt.subplot2grid((nrows, 2), (nrows-1, 0), colspan=2)
            for i, (N, skew) in enumerate(zip(Ns, skews)):
                betas = self.betas
                name = self.saved_data_name(N, skew)
                alllnlikes = self.saved_data[name][
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

    def RecalculateEvidence(self, sampler):
        """ Recalculate the evidence when logl contains nans """

        nburn = self.nburn

        betas = sampler.betas
        alllnlikes = sampler.lnlikelihood[:, :, nburn:]
        mean_lnlikes = np.mean(np.mean(alllnlikes, axis=1), axis=1)

        mean_lnlikes = mean_lnlikes[::-1]
        betas = betas[::-1]

        idxs = np.isinf(mean_lnlikes)
        mean_lnlikes = mean_lnlikes[~idxs]
        betas = betas[~idxs]
        lnevidence = np.trapz(mean_lnlikes, betas)
        z1 = np.trapz(mean_lnlikes, betas)
        z2 = np.trapz(mean_lnlikes[::-1][::2][::-1],
                      betas[::-1][::2][::-1])
        lnevidence_err = np.abs(z1 - z2)

        return lnevidence, lnevidence_err

    def BayesFactor(self, Ns=None, skews=None, print_result=True):

        if Ns is None and skews is None:
            Ns = []
            skews = []
            for key in self.fitted_Ns:
                Ns.append(int(key[-1]))
                if "S" in key:
                    skews.append(True)
                else:
                    skews.append(False)
        elif type(Ns) == list and skews is None:
            skews = [False] * len(Ns)
        elif len(skews) != len(Ns):
            raise ValueError("len(skews) == len(Ns)")

        evi_err = []
        for N, skew in zip(Ns, skews):
            name = self.saved_data_name(N, skew)
            sampler = self.saved_data[name]['sampler']
            fburnin = float(self.nburn)/(self.nburn+self.nprod)
            lnevidence, lnevidence_err = sampler.thermodynamic_integration_log_evidence(
                fburnin=fburnin)
            if np.isinf(lnevidence):
                print "Recalculating evidence for {} due to inf".format(name)
                lnevidence, lnevidence_err = sampler.RecalculateEvidence(sampler)
            log10evidence = lnevidence/np.log(10)
            log10evidence_err = lnevidence_err/np.log(10)
            evi_err.append((name, log10evidence, log10evidence_err))

        if print_result:
            for mA, mB in itertools.combinations(evi_err, 2):
                mA_name, mA_evi, mA_err = mA
                mB_name, mB_evi, mB_err = mB
                bf = mB_evi - mA_evi
                bf_err = np.sqrt(mA_err**2 + mB_err**2)
                print "log10 Bayes Factor ({}, {}) = {} +/- {}".format(
                    mB_name, mA_name, bf, bf_err)

        self.evi_err = evi_err
