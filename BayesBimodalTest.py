import numpy as np
import matplotlib.pyplot as plt
from emcee import PTSampler


class BayesBimodalTest():
    def __init__(self, data, ntemps=20, betamin=-22, nburn0=100, nburn=100, nprod=100,
                 nwalkers=100):
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

        self.fit_unimodal()
        self.fit_bimodal()
        self.summarise_posteriors()
        self.diagnostic_plot()

    def unif(self, x, a, b):
        if (x < a) or (x > b):
            return -np.inf
        else:
            return 1./(b-a)

    def get_uniform_prior_lims(self, key):
        if key == "mu":
            return [self.data_min, self.data_max]
        if key == "sigma":
            return [1e-20*self.data_std, 2*self.data_std]
        if key == "p":
            return [0, 1]

    def logp_unimodal(self, params):
        muA, sigmaA = params
        logp = 0
        logp += self.unif(muA, *self.get_uniform_prior_lims('mu'))
        logp += self.unif(sigmaA, *self.get_uniform_prior_lims('sigma'))
        return logp

    def logl_unimodal(self, params, data):
        muA, sigmaA = params
        resA = np.array(data-muA)
        r = np.log(1/(sigmaA*np.sqrt(2*np.pi))*np.exp(-resA**2/(2*sigmaA**2)))
        return np.sum(r)

    def get_new_p0(self, sampler, ndim, scatter_val=1e-3):
        pF = sampler.chain[:, :, -1, :].reshape(
            self.ntemps, self.nwalkers, ndim)[0, :, :]
        lnp = sampler.lnprobability[:, :, -1].reshape(
            self.ntemps, self.nwalkers)[0, :]
        p = pF[np.argmax(lnp)]
        p0 = [[p + scatter_val * p * np.random.randn(ndim)
              for i in xrange(self.nwalkers)] for j in xrange(self.ntemps)]
        return p0

    def fit_unimodal(self):
        sampler = PTSampler(self.ntemps, self.nwalkers, 2, self.logl_unimodal,
                            self.logp_unimodal, loglargs=[self.data], betas=self.betas)
        param_keys = ['mu', 'sigma']
        p0 = [[[np.random.uniform(*self.get_uniform_prior_lims(key))
                for key in param_keys]
               for i in range(self.nwalkers)]
              for j in range(self.ntemps)]
        out = sampler.run_mcmc(p0, self.nburn0)
        self.unimodal_chains0 = sampler.chain[0, :, : , :]
        p0 = self.get_new_p0(sampler, 2)
        sampler.reset()
        out = sampler.run_mcmc(p0, self.nburn + self.nprod)
        self.unimodal_chains = sampler.chain[0, :, : , :]

        self.unimodal_sampler = sampler
        self.unimodal_samples = sampler.chain[0, :, self.nburn:, :].reshape((-1, 2))

    def logp_bimodal(self, params):
        muA, muB, sigmaA, sigmaB, p = params
        logp = 0
        logp += self.unif(muA, *self.get_uniform_prior_lims('mu'))
        logp += self.unif(muB, *self.get_uniform_prior_lims('mu'))
        logp += self.unif(sigmaA, *self.get_uniform_prior_lims('sigma'))
        logp += self.unif(sigmaB, *self.get_uniform_prior_lims('sigma'))
        logp += self.unif(p, *self.get_uniform_prior_lims('p'))
        if muA > muB:
            logp += -np.inf
        return logp

    def logl_bimodal(self, params, data):
        muA, muB, sigmaA, sigmaB, p = params
        resA = np.array(data-muA)
        resB = np.array(data-muB)
        r = np.log(p/(sigmaA*np.sqrt(2*np.pi))*np.exp(-resA**2/(2*sigmaA**2)) +
                   (1-p)/(sigmaB*np.sqrt(2*np.pi))*np.exp(-resB**2/(2*sigmaB**2)))
        return np.sum(r)

    def fit_bimodal(self):
        sampler = PTSampler(self.ntemps, self.nwalkers, 5, self.logl_bimodal,
                            self.logp_bimodal, loglargs=[self.data], betas=self.betas)
        param_keys = ['mu', 'mu', 'sigma', 'sigma', 'p']
        p0 = [[[np.random.uniform(*self.get_uniform_prior_lims(key))
                for key in param_keys]
               for i in range(self.nwalkers)]
              for j in range(self.ntemps)]

        out = sampler.run_mcmc(p0, self.nburn0)
        self.bimodal_chains0 = sampler.chain[0, :, : , :]
        p0 = self.get_new_p0(sampler, 5)
        sampler.reset()
        out = sampler.run_mcmc(p0, self.nburn + self.nprod)
        self.bimodal_chains = sampler.chain[0, :, : , :]

        self.bimodal_sampler = sampler
        self.bimodal_samples = sampler.chain[0, :, self.nburn:, :].reshape((-1, 5))

    def summarise_posteriors(self):
        pass

    def diagnostic_plot(self):
        fig = plt.figure()
        unimodal_color = "k"
        bimodal_colorA = "r"
        bimodal_colorB = "g"

        burn0s = np.arange(0, self.nburn0)
        prods = np.arange(self.nburn0, self.nburn0+self.nburn + self.nprod)

        ax00 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
        ax00.hist(self.data, bins=50, color="b", histtype="step")

        ax10 = plt.subplot2grid((4, 2), (1, 0))
        ax10.hist(self.unimodal_samples[:, 0], bins=50,
                  histtype="step", color=unimodal_color)
        ax10.hist(self.bimodal_samples[:, 0], bins=50,
                  histtype="step", color=bimodal_colorA)
        ax10.hist(self.bimodal_samples[:, 1], bins=50,
                  histtype="step", color=bimodal_colorB)

        ax11 = plt.subplot2grid((4, 2), (1, 1))
        ax11.plot(prods, self.unimodal_chains[:, :, 0].T, lw=0.01,
                  color=unimodal_color)
        ax11.plot(burn0s, self.bimodal_chains0[:, :, 0].T, lw=0.01,
                  color=bimodal_colorA)
        ax11.plot(prods, self.bimodal_chains[:, :, 0].T, lw=0.01,
                  color=bimodal_colorA)
        ax11.plot(burn0s, self.bimodal_chains0[:, :, 1].T, lw=0.01,
                  color=bimodal_colorB)
        ax11.plot(prods, self.bimodal_chains[:, :, 1].T, lw=0.01,
                  color=bimodal_colorB)

        ax20 = plt.subplot2grid((4, 2), (2, 0))
        ax20.hist(self.unimodal_samples[:, 1], bins=50,
                  histtype="step", color=unimodal_color)
        ax20.hist(self.bimodal_samples[:, 2], bins=50,
                  histtype="step", color=bimodal_colorA)
        ax20.hist(self.bimodal_samples[:, 3], bins=50,
                  histtype="step", color=bimodal_colorB)

        ax21 = plt.subplot2grid((4, 2), (2, 1))
        ax21.plot(prods, self.unimodal_chains[:, :, 1].T, lw=0.01,
                  color=unimodal_color)
        ax21.plot(burn0s, self.bimodal_chains0[:, :, 2].T, lw=0.01,
                  color=bimodal_colorA)
        ax21.plot(prods, self.bimodal_chains[:, :, 2].T, lw=0.01,
                  color=bimodal_colorA)
        ax21.plot(burn0s, self.bimodal_chains0[:, :, 3].T, lw=0.01,
                  color=bimodal_colorB)
        ax21.plot(prods, self.bimodal_chains[:, :, 3].T, lw=0.01,
                  color=bimodal_colorB)

        ax30 = plt.subplot2grid((4, 2), (3, 0))
        ax30.hist(self.bimodal_samples[:, 4], bins=50, histtype="step",
                  color=bimodal_colorA)

        ax31 = plt.subplot2grid((4, 2), (3, 1))
        ax31.plot(burn0s, self.bimodal_chains0[:, :, 4].T, lw=0.01,
                  color=bimodal_colorA)
        ax31.plot(prods, self.bimodal_chains[:, :, 4].T, lw=0.01,
                  color=bimodal_colorA)

        for ax in [ax11, ax21, ax31]:
            ax.axvline(self.nburn0, color="k", lw=0.1, alpha=0.4)
            ax.axvline(self.nburn0+self.nburn, color="k", lw=0.1, alpha=0.4)
            ax.axvline(self.nburn0+self.nburn+self.nprod, color="k", lw=0.1, alpha=0.4)
        plt.savefig("temp.pdf")
