import numpy as np
import matplotlib as plt
sym_size = 8
# max_display = 10
axis_color = 'black'
alpha = 0.5

def summary_plot(shap_values, features=None, ax=None, max_display=10):
    """Create a SHAP summary plot, colored by feature values when they are provided.
    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.
    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand
    feature_names : list
        Names of the features (length # features)
    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)
    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.
    """

    feature_names = features.columns
    features = features.values
    num_features = shap_values.shape[1]

    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)):]

    row_height = 0.4
    ax.axvline(x=0, color="#999999", zorder=-1)

    for pos, i in enumerate(feature_order):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_values[:, i]
        values = features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]
        values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        # trim the color range, but prevent the color range from collapsing
        vmin = np.nanpercentile(values, 5)
        vmax = np.nanpercentile(values, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(values, 1)
            vmax = np.nanpercentile(values, 99)
            if vmin == vmax:
                vmin = np.min(values)
                vmax = np.max(values)

        assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

        # plot the nan values in the interaction feature as grey
        nan_mask = np.isnan(values)
        ax.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                   vmax=vmax, s=sym_size, alpha=alpha, linewidth=0,
                   zorder=3, rasterized=len(shaps) > 500)

        # plot the non-nan values colored by the trimmed feature value
        cvals = values[np.invert(nan_mask)].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
        cvals[cvals_imp > vmax] = vmax
        cvals[cvals_imp < vmin] = vmin
        ax.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                   cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, s=sym_size,
                   c=cvals, alpha=alpha, linewidth=0,
                   zorder=3, rasterized=len(shaps) > 500)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ticklabels = (range(len(feature_order)), [feature_names[i] for i in feature_order])
    ax.set_yticks(ticklabels[0])
    ax.set_yticklabels(ticklabels[1], fontsize=8)
#     plt.yticks(ticklabels[0], ticklabels[1], fontsize=8)
    ax.tick_params('x', labelsize=8)
    ax.set_ylim(-1, len(feature_order))
