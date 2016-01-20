import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from emcee import PTSampler
import seaborn as sns


class BayesBimodalTest():
    def __init__(self, data, Ns=[1, 2], ntemps=20, betamin=-22,
                 nburn0=100, nburn=100, nprod=100, nwalkers=100):
        self.data = data
        self.Ns = Ns
        self.max_N = max(Ns)
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

        for N in Ns:
            self.fit_Nmodal(N)
        self.summarise_posteriors()

    def log_unif(self, x, a, b):
        if (x < a) or (x > b):
            return -np.inf
        else:
            return np.log(1./(b-a))

    def get_uniform_prior_lims(self, key):
        if key == "mu":
            return [self.data_min, self.data_max]
        if key == "sigma":
            return [1e-20*self.data_std, 2*self.data_std]
        if key == "p":
            return [0, 1]

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
        if any(np.diff(mus) < 0):
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
        param_keys = ['mu'] * N + ['sigma'] * N + ['p'] * (N-1)
        p0 = [[[np.random.uniform(*self.get_uniform_prior_lims(key))
                for key in param_keys]
               for i in range(self.nwalkers)]
              for j in range(self.ntemps)]

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

    def summarise_posteriors(self):
        for N in self.Ns:
            saved_data = self.saved_data['N{}'.format(N)]
            saved_data['mus'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(N)]
            saved_data['sigmas'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(N, 2*N)]
            saved_data['ps'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(2*N, 3*N-1)]
            saved_data['ps'].append(1-np.sum(saved_data['ps']))
            self.saved_data['N{}'.format(N)] = saved_data

    def diagnostic_plot(self, fname="diagnostic.png", trace_line_width=0.1,
                        hist_line_width=1.5):

        fig = plt.figure(figsize=(8, 11))
        if self.ntemps > 1:
            nrows = 5
        else:
            nrows = 4

        colors = [sns.xkcd_rgb["pale red"],
                  sns.xkcd_rgb["medium green"],
                  sns.xkcd_rgb["denim blue"]
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
        x_plot = np.linspace(self.data.min(), self.data.max(), 1000)

        for i, N in enumerate(self.Ns):

            saved_data = self.saved_data['N{}'.format(N)]
            zi = zip(saved_data['ps'], saved_data['mus'], saved_data['sigmas'])
            for j, (p, mu, sigma) in enumerate(zi):
                c = colors[i+j]
                ax00.plot(x_plot, p*ss.norm.pdf(x_plot, mu, sigma),
                          color=c, label="N{}({})".format(N, j))

            for j, (lax, rax) in enumerate(zip(Laxes, Raxes)):
                if j == 2:
                    krange = N-1
                else:
                    krange = N
                for k in range(krange):
                    c = colors[i+k]
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

        #if self.ntemps > 1:
        #    ax40 = plt.subplot2grid((nrows, 2), (nrows-1, 0), colspan=2)
        #    for i, d in enumerate(self.degrees):
        #        betas = self.betas
        #        alllnlikes = self.stored_data["p{}".format(d)][
        #            'sampler'].lnlikelihood[:, :, self.nburn:]
        #        mean_lnlikes = np.mean(np.mean(alllnlikes, axis=1), axis=1)
        #        ax40.semilogx(betas, mean_lnlikes, "-o", color=colors[i])
        #        ax40.set_title("Linear thermodynamic integration")



        #if self.ntemps > 1:
        #    ax40 = plt.subplot2grid((nrows, 2), (4, 0))
        #    betas = self.unimodal_sampler.betas
        #    alllnlikes = self.unimodal_sampler.lnlikelihood[:, :, self.nburn:]
        #    mean_lnlikes = np.mean(np.mean(alllnlikes, axis=1), axis=1)
        #    ax40.semilogx(betas, mean_lnlikes, "-o", color="k")
        #    ax40.set_title("Unimodal thermodynamic integration")

        #    ax41 = plt.subplot2grid((nrows, 2), (4, 1))
        #    betas = self.bimodal_sampler.betas
        #    alllnlikes = self.bimodal_sampler.lnlikelihood[:, :, self.nburn:]
        #    mean_lnlikes = np.mean(np.mean(alllnlikes, axis=1), axis=1)
        #    ax41.semilogx(betas, mean_lnlikes, "-o", color="k")
        #    ax41.set_title("Bimodal thermodynamic integration")

        fig.tight_layout()
        fig.savefig(fname)

    def BayesFactor(self, print_result=True):
        (unimodal_lnevidence, unimodal_lnevidence_err) = self.unimodal_sampler.thermodynamic_integration_log_evidence()
        unimodal_log10evidence = unimodal_lnevidence/np.log(10)
        unimodal_log10evidence_err = unimodal_lnevidence_err/np.log(10)

        (bimodal_lnevidence, bimodal_lnevidence_err) = self.bimodal_sampler.thermodynamic_integration_log_evidence()
        bimodal_log10evidence = bimodal_lnevidence/np.log(10)
        bimodal_log10evidence_err = bimodal_lnevidence_err/np.log(10)

        bf = bimodal_log10evidence - unimodal_log10evidence
        bf_err = np.sqrt(bimodal_log10evidence_err**2 + unimodal_log10evidence_err**2)

        Umu = self.get_uniform_prior_lims('mu')
        Usigma = self.get_uniform_prior_lims('sigma')
        occams_factor = np.log10(Umu[1] - Umu[0]) + np.log10(Usigma[1] - Usigma[0])
        if print_result:
            print "Bayes factor of {} +/- {}".format(bf, bf_err)
            print "Occams factor is {:1.2f}".format(occams_factor)
        else:
            return bf, bf_err, occams_factor
