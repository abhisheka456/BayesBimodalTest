import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import itertools
from emcee import PTSampler
from scipy.special import erf

try:
    import ipyparallel as ipp
    use_parallel_global = True
except ImportError:
    use_parallel_global = False


class BayesBimodalTest():
    """ A Bayesian model comparison for N-modality in a data set

    Usage
    -----
        To fit a unimodal and bimodal Gaussian mixture model to data

        ```
        test = BayesBimodalTest(data)
        test.fit_Nmodal(1)
        test.fit_Nmodal(2)
        ```

        Then, to calculate the Bayes factor

        ```
        test.BayesFactor([1, 2])
        ```


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
    verbose: bool
        Print additional information
    mu_scale_factor: float
        Scale factor for the width of the uniform mixture means
    sigma_scale_factor: float
        Scale factor for the half-Cauchy prior on the standard deviation
    alpha_scale_factor: float
        Standard deviation scale factor used for the Normal skew prior
    p_scale_factor: float
        Beta shape parameter used for the p Beta prior

    Note:
        The priors for all parameters are chosen automatically, but can be
        scaled. To see how this is implemented, or overwrite the defaults, see
        the `prior_spec`dictionary.

    """

    def __init__(self, data, ntemps=20, betamin=-22, nburn0=100, nburn=100,
                 nprod=100, nwalkers=100, verbose=False,
                 mu_scale_factor=10, sigma_scale_factor=2,
                 alpha_scale_factor=10, p_shape_factor=(1, 1)):
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
        self.fitted_Ns = []
        self.verbose = verbose

        self.mu_scale_factor = mu_scale_factor
        self.sigma_scale_factor = sigma_scale_factor
        self.alpha_scale_factor = alpha_scale_factor
        self.p_shape_factor = p_shape_factor

        drange = self.data_max - self.data_min
        dave = 0.5*(self.data_max + self.data_min)
        self.prior_spec = {'mu': {'func': self.log_unif,
                                  'lower': dave - 0.5*mu_scale_factor * drange,
                                  'upper': dave + 0.5*mu_scale_factor * drange},
                           'sigma': {'func': self.log_half_cauchy,
                                     'gamma': sigma_scale_factor * self.data_std},
                           'p': {'func': self.log_beta,
                                 'shape_factor': p_shape_factor},
                           'alpha': {'func': self.log_norm,
                                     'mu': 0,
                                     'sigma': alpha_scale_factor * self.data_std}
                           }

    def vprint(self, messg):
        if self.verbose:
            print(messg)

    def invert_saved_data_name(self, name):
        """ Helper function to generate N and skew from data filename """
        if "S" in name:
            return int(name[-1]), True
        else:
            return int(name[-1]), False

    def saved_data_name(self, N, skew):
        """ Helper function to generate the saved data filename """
        if skew is False:
            return "N{}".format(N)
        elif skew is True:
            return "NS{}".format(N)

    def get_p0_lims(self, key):
        """ Return uniform prior limits from the parameter name (key)

        All walkers are initialised using a uniform distribiution with limits
        specified by this routine.

        """

        if key == "mu":
            return [self.data_min, self.data_max]
        if key == "sigma":
            return [0, self.data_std]
        if key == "p":
            return [0, 1]
        if key == "alpha":
            return [-10*self.data_std, 10*self.data_std]

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
                        [lower, upper] = self.get_p0_lims(key)
                        dp = (upper - lower) / float(N)
                        component_range = [lower + component*dp,
                                           lower + (component+1)*dp]
                        p0_2.append(np.random.uniform(*component_range))
                    else:
                        p0_2.append(np.random.uniform(
                            *self.get_p0_lims(key)))
                p0_1.append(p0_2)
            p0.append(p0_1)

        return p0

    def get_new_p0(self, sampler, ndim, scatter_val=1e-3):
        """ Returns new initial positions for walkers are burn0 stage

        This returns new positions for all walkers by scattering points about
        the maximum posterior with scale `scatter_val`.

        """
        pF = sampler.chain[:, :, -1, :].reshape(
            self.ntemps, self.nwalkers, ndim)[0, :, :]
        lnp = sampler.lnprobability[:, :, -1].reshape(
            self.ntemps, self.nwalkers)[0, :]
        p = pF[np.argmax(lnp)]
        p0 = [[p + scatter_val * p * np.random.randn(ndim)
              for i in xrange(self.nwalkers)] for j in xrange(self.ntemps)]
        return p0

    def unpack_params(self, params, skew=False):
        if skew:
            N = (len(params) + 1) / 4
        else:
            N = (len(params) + 1) / 3

        mu = params[:N]
        sigma = params[N:2*N]
        if skew:
            alpha = params[2*N:3*N]
            p = params[3*N:]
        else:
            p = params[2*N:]

        p = np.append(p, 1-np.sum(p))

        if skew:
            return mu, sigma, alpha, p
        else:
            return mu, sigma, p

    def log_unif(self, x, lower, upper):
        if (x < lower) or (x > upper):
            return -np.inf
        else:
            return -np.log(upper-lower)

    def log_half_cauchy(self, x, gamma):
        if x < 0:
            return - np.inf
        else:
            return -np.log(np.pi*(gamma + x**2 / gamma))

    def log_beta(self, x, shape_factor):
        alpha, beta = shape_factor
        gamma = np.math.gamma
        if (0 > x) or (x > 1):
            return -np.inf
        beta = (gamma(alpha+beta)/(gamma(alpha)*gamma(beta)) * (
                x**(alpha-1.) * (1-x)**(beta-1.)))
        return np.log(beta)

    def log_norm(self, x, mu, sigma):
        return -.5*((x-mu)**2/sigma**2 + np.log(sigma**2*2*np.pi))

    def get_prior(self, x, key):
        """ Wrapper to calculate the prior given self.prior_spec """
        d = self.prior_spec[key].copy()
        func = d.pop('func')
        return func(x, **d)

    def logp_Nmodal(self, params):
        mu, sigma, p = self.unpack_params(params)
        if any(np.diff(mu) < 0):
            return -np.inf
        if np.sum(p) > 1:
            return -np.inf

        sumv = 0
        sumv += np.sum([self.get_prior(x, 'mu') for x in mu])
        sumv += np.sum([self.get_prior(x, 'sigma') for x in sigma])
        if len(p) > 1:
            sumv += np.sum([self.get_prior(x, 'p') for x in p])

        return sumv

    def logl_Nmodal(self, params, data):
        mu, sigma, p = self.unpack_params(params)
        res = (data.reshape((len(data), 1)) - mu.T)
        r = np.log(np.sum(p/(sigma*np.sqrt(2*np.pi)) *
                          np.exp(-res**2/(2*sigma**2)), axis=1))
        return np.sum(r)

    def fit_Nmodal(self, N):
        """ Fit the N-component Gaussian mixture modal

        Note:
            `params` is a 3N-1 vector, with the first N as the mu's, the second
            N the sigmas, and the last N-1 being the p's.
        """

        name = self.saved_data_name(N, False)
        ndim = N * 3 - 1
        sampler = PTSampler(self.ntemps, self.nwalkers, ndim,
                            logl=self.logl_Nmodal,
                            logp=self.logp_Nmodal, loglargs=[self.data],
                            betas=self.betas)
        p0 = self.create_initial_p0(N)

        self.fit_method(sampler, p0, name, ndim)

    def logp_SkewNmodal(self, params):
        mu, sigma, alpha, p = self.unpack_params(params, skew=True)

        if any(np.diff(mu) < 0):
            return -np.inf
        if np.sum(p) > 1:
            return -np.inf

        sumv = 0
        sumv += np.sum([self.get_prior(x, 'mu') for x in mu])
        sumv += np.sum([self.get_prior(x, 'sigma') for x in sigma])
        sumv += np.sum([self.get_prior(x, 'alpha') for x in alpha])
        sumv += np.sum([self.get_prior(x, 'p') for x in p])

        return sumv

    def logl_SkewNmodal(self, params, data):
        mu, sigma, alpha, p = self.unpack_params(params, skew=True)
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
        """ Fit the N-component Gaussian mixture modal

        Note:
            `params` is a 3N-1 vector, with the first N as the mu's, the second
            N the sigmas, and the last N-1 being the p's.
        """


        name = self.saved_data_name(N, True)
        ndim = N*4 - 1
        sampler = PTSampler(self.ntemps, self.nwalkers, ndim,
                            logl=self.logl_SkewNmodal,
                            logp=self.logp_SkewNmodal, loglargs=[self.data],
                            betas=self.betas)
        p0 = self.create_initial_p0(N, skew=True)

        self.fit_method(sampler, p0, name, ndim)

    def fit_method(self, sampler, p0, name, ndim):
        saved_data = {}
        if self.nburn0 != 0:
            sampler_out0 = self._try_parallel(sampler, p0, self.nburn0)
            saved_data["chains0"] = sampler_out0.chain[0, :, : , :]
            p0 = self.get_new_p0(sampler_out0, ndim)
            sampler.reset()
        else:
            saved_data["chains0"] = None

        # Fit burn and production
        sampler_out1 = self._try_parallel(sampler, p0, self.nburn + self.nprod)

        # Add chains and samples to saved data
        saved_data["chains"] = sampler_out1.chain[0, :, :, :]
        saved_data["samples"] = sampler_out1.chain[0, :, self.nburn:, :].reshape(
            (-1, ndim))

        # Calculate and save the evidence
        fburnin = float(self.nburn)/(self.nburn+self.nprod)
        lnevidence, lnevidence_err = sampler_out1.thermodynamic_integration_log_evidence(
            fburnin=fburnin)
        if np.isinf(lnevidence):
            print("Recalculating evidence for {} due to inf".format(name))
            lnevidence, lnevidence_err = self.RecalculateEvidence(
                sampler_out1)
        log10evidence = lnevidence/np.log(10)
        log10evidence_err = lnevidence_err/np.log(10)
        saved_data["log10evidence"] = log10evidence
        saved_data["log10evidence_err"] = log10evidence_err
        saved_data["alllnlikes"] = sampler_out1.lnlikelihood[:, :, self.nburn:]

        self.saved_data[name] = saved_data
        self.summarise_posteriors(name)
        self.fitted_Ns.append(name)

        print("Fitted {} model".format(name))


    def summarise_posteriors(self, name=None, N=None, skew=False):
        if name:
            N, skew = self.invert_saved_data_name(name)
        else:
            name = self.saved_data_name(N, skew)
        saved_data = self.saved_data[name]
        saved_data['mu'] = [
            np.mean(saved_data['samples'][:, i]) for i in range(N)]
        saved_data['sigma'] = [
            np.mean(saved_data['samples'][:, i]) for i in range(N, 2*N)]
        if skew:
            saved_data['alpha'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(2*N, 3*N)]
            saved_data['p'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(3*N, 4*N-1)]
            saved_data['p'].append(1-np.sum(saved_data['p']))
        else:
            saved_data['p'] = [
                np.mean(saved_data['samples'][:, i]) for i in range(2*N, 3*N-1)]
            saved_data['p'].append(1-np.sum(saved_data['p']))

        self.saved_data[name] = saved_data

    def pdf(self, x, mu, sigma, p, alpha=0):
        phi = ss.norm.pdf(x, mu, sigma)
        Phi = .5 * (1 + erf(alpha * (x - mu)/sigma))
        return 2 * p * phi * Phi

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

        colors = ["r", "b", "k", "g", "orange", "m", "olive"]

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
                alpha = saved_data['alpha']
            else:
                alpha = [0] * N
            zi = zip(saved_data['p'], saved_data['mu'], saved_data['sigma'],
                     alpha)

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
                    samples = saved_data['samples'][:, j*N+k]
                    hist, bin_edges = np.histogram(samples, bins=50)
                    bin_mids = 0.5*(bin_edges[:-1] + bin_edges[1:])
                    lax.plot(bin_mids, hist, color=c)

                    if saved_data['chains0'] is not None:
                        rax.plot(burn0s, saved_data['chains0'][:, :, j*N+k].T,
                                 lw=trace_line_width, color=c)
                    rax.plot(prods, saved_data['chains'][:, :, j*N+k].T,
                             lw=trace_line_width, color=c)

                if krange == N-1:  # Add distribution of missing p val
                    oth_samples = [saved_data['samples'][:, j*N+k] for k
                                   in range(krange)]
                    samples = 1-np.sum(oth_samples, axis=0)
                    hist, bin_edges = np.histogram(samples, bins=50)
                    bin_mids = 0.5*(bin_edges[:-1] + bin_edges[1:])
                    lax.plot(bin_mids, hist, color=c)


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
            ax_evi = plt.subplot2grid((nrows, 2), (nrows-1, 0), colspan=2)
            for i, (N, skew) in enumerate(zip(Ns, skews)):
                betas = self.betas
                name = self.saved_data_name(N, skew)
                alllnlikes = self.saved_data[name]['alllnlikes']
                mean_lnlikes = np.mean(np.mean(alllnlikes, axis=1), axis=1)
                ax_evi.semilogx(betas, mean_lnlikes, "-o", color=colors[i])
                ax_evi.set_title("Linear thermodynamic integration")

        fig.tight_layout()
        fig.savefig(fname)

    def OccamFactor(self):
        Umu = self.get_uniform_prior_lims('mu')
        Usigma = self.get_uniform_prior_lims('sigma')
        occams_factor = (np.log10(Umu[1] - Umu[0])
                         + np.log10(Usigma[1] - Usigma[0])
                         + np.log10(1))
        print("Occams factor is {}".format(occams_factor))

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
            log10evidence = self.saved_data[name]['log10evidence']
            log10evidence_err = self.saved_data[name]['log10evidence_err']
            evi_err.append((name, log10evidence, log10evidence_err))

        if print_result:
            for mA, mB in itertools.combinations(evi_err, 2):
                mA_name, mA_evi, mA_err = mA
                mB_name, mB_evi, mB_err = mB
                bf = mB_evi - mA_evi
                bf_err = np.sqrt(mA_err**2 + mB_err**2)
                print("log10 Bayes Factor ({}, {}) = {} +/- {}".format(
                    mB_name, mA_name, bf, bf_err))

        self.evi_err = evi_err


    def _try_parallel(self, sampler, p0, N, run_line=""):
        """ Attempt to run the sampler using IPython parallel, if not run seriel

        Parameters
        sampler: an emcee sampler object
        p0: array
            the initial values of the nwalkers
        N: int
            Number of steps to run
        run_line: str
            Executable python to be run on each machine

        returns: sampler
        """

        p0 = np.array(p0)

        # First check to see if the parallel has been set by the user
        try:
            use_parallel = sampler.use_parallel
        except AttributeError:
            use_parallel = True

        if use_parallel_global is False:
            use_parallel = False

        if use_parallel:
            try:
                rc = ipp.Client()
                dview = rc[:]
                dview.execute("import numpy as np")
                dview.block = True
                dview.use_dill()  # Standard pickle fails with functions
                sampler_result = self._run_sampler(dview, sampler, run_line, p0=p0, N=N)
                rc.purge_everything()
                rc.clear('all')
                rc.metadata.clear()
                rc.results.clear()
                dview.results.clear()
                return sampler_result
            except IOError:
                print("No clusters found, continue with seriel MCMC simulation")
                sampler.run_mcmc(p0, N)
                return sampler
        else:
            print("Running seriel MCMC simulation")
            sampler.run_mcmc(p0, N)
            return sampler


    def _run_sampler(self, dview, sampler, run_line, **kwargs):
        """ Run the sampler on all instances of the dview

        This spreads the total workload over the n views. The basic idea is to
        reduce the number of walkers on each view in inverse proportion to the
        number of view that we have. So, it will reduce the time for any simultion
        by a factor 1/nviews while still producing equivalent results.

        Parameters
        ----------
        dview: a direct view of the machines
        sampler: an emcee sampler object
        run_line: str
            Executable python to be run on each machine
        kwargs:
            Any variables to pass to the machines, must contain p0, N
        """

        nviews = len(dview)

        for key, val in kwargs.items():
            dview[key] = val

        def MakeEven(num, j=0):
            """ Make the number even by adding 1 if j is odd and subtracting if
                j is even
            """
            if num % 2 == 0:
                return num
            else:
                return num + [-1, 1][j % 2]

        nwalkers_list = [MakeEven(int(sampler.nwalkers/nviews), j)
                         for j in range(len(dview)-1)]
        nwalkers_list.append(MakeEven(sampler.nwalkers - sum(nwalkers_list)))
        self.vprint("Splitting the original {} walkers over {} machines:\n".format(
              sampler.nwalkers, nviews) +
              "#: nwalk, p0 index range")
        for i, nwalkers in enumerate(nwalkers_list):
            j = sum(nwalkers_list[:i])
            k = sum(nwalkers_list[:i+1])
            self.vprint("{} : {}, {}->{}".format(i, nwalkers, j, k))
            dview.push(dict(nwalkers=nwalkers, ID=i, j=j, k=k), targets=i)
        self.vprint("Total: {}".format(sum(nwalkers_list)))

        dview['sampler'] = sampler
        dview.execute(run_line)

        self.vprint("Execute code")
        dview.execute(
            "sampler.nwalkers = nwalkers\n"
            "sampler.reset()\n"
            "p0 = p0[:, j:k, :]\n"
            "sampler.run_mcmc(p0, N)\n"
            "chain = sampler.chain[:, :, :, :]\n"
            "lnlikelihood = sampler.lnlikelihood[:, :, :]\n"
            "lnprobability = sampler.lnprobability[:, :, :]")

        chain = np.concatenate(dview.get("chain"), axis=1)
        lnlikelihood = np.concatenate(dview.get("lnlikelihood"), axis=1)
        lnprobability = np.concatenate(dview.get("lnprobability"), axis=1)

        return fake_sampler_data(chain, lnlikelihood, lnprobability,
                            sampler.betas, sampler.dim)

class fake_sampler_data():
    """ A fake PTSampler instance  that can be monkey patched"""
    def __init__(self, chain, lnlikelihood, lnprobability, betas, dim):
        self.chain = chain
        self.lnlikelihood = lnlikelihood
        self.lnprobability = lnprobability
        self.betas = betas
        self.dim = dim

fake_sampler_data.thermodynamic_integration_log_evidence = (
    PTSampler.thermodynamic_integration_log_evidence.im_func)

