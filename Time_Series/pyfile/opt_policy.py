import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import RL as rl
import numpy as np

def q_with_optimalaction(Q):
    S = dict()
    for state in Q.keys():
        opt_action = np.argmax(Q[state])
        S[state] = opt_action

    return S


def policy_visualize(Q, env, decks):
    Q = rl.convert_to_sum_states(Q, env)
    Q_ = q_with_optimalaction(Q)
    optQ = rl.fill_missing_sum_states(rl.filter_states(Q_), default_value = 0.5)

    data = pd.DataFrame(list(optQ.items()))
    for i in data[0]:
        if i == data[0][0]:
            x = np.array(i[0])
            y = np.array(i[1])
            z = np.array(i[2])
        else:
            x = np.append(x, i[0])
            y = np.append(y, i[1])
            z = np.append(z, i[2])
    data["player_hand"] = x
    data["show_card"] = y
    data["use_ace"] = z
    data.drop(0, axis = 1, inplace = True)

    use_ace_set = data[data["use_ace"] == True]
    nouse_ace_set = data[data["use_ace"] == False]

    use_ace_set = use_ace_set.pivot(index = "player_hand", columns = "show_card"
                                    , values=1).sort_index(ascending=False)
    nouse_ace_set = nouse_ace_set.pivot(index = "player_hand", columns = "show_card",
                                        values=1).sort_index(ascending=False)

    """ax1, ax2 = plt.axes()
    ax1.set_title("Optimal Policy with use ace")
    ax2.set_title("Optimal Policy without use ace")

    fig1 = sns.heatmap(use_ace_set, ax = ax1).get_figure()
    fig2 = sns.heatmap(nouse_ace_set, ax = ax2).get_figure()

    fig1.savefig("figures/Optimal Policy with use ace in {}deck.jpg".format(decks))
    fig2.savefig("figures/Optimal Policy without use ace in {}decks.jpg".format(decks))"""


    fig, ax = plt.subplots(1, 2, figsize = (20, 10))
    fig.suptitle("optimal policy in {}decks".format(decks), fontsize = 16)
    ax[0].set_title("with use ace")
    ax[1].set_title("without use ace")
    color = ["k", "w", "g"]
    cmap = sns.color_palette(color, n_colors = 3)

    sns.heatmap(use_ace_set, ax = ax[0], cmap = cmap, linewidths = .5, linecolor = "lightgray",
                cbar_kws = {"ticks":[0., 0.5, 1.]})
    sns.heatmap(nouse_ace_set, ax = ax[1], cmap = cmap, linewidths = .5, linecolor = "lightgray",
                cbar_kws={"ticks": [0., 0.5, 1.]})


    fig.savefig("figures/Optimal Policy in {}deck.jpg".format(decks))


def sum_policy_visualize(Q, decks):
    Q_ = q_with_optimalaction(Q)
    optQ = rl.fill_missing_sum_states(rl.filter_states(Q_), default_value = 0.5)

    data = pd.DataFrame(list(optQ.items()))
    for i in data[0]:
        if i == data[0][0]:
            x = np.array(i[0])
            y = np.array(i[1])
            z = np.array(i[2])
        else:
            x = np.append(x, i[0])
            y = np.append(y, i[1])
            z = np.append(z, i[2])
    data["player_hand"] = x
    data["show_card"] = y
    data["use_ace"] = z
    data.drop(0, axis = 1, inplace = True)

    use_ace_set = data[data["use_ace"] == True]
    nouse_ace_set = data[data["use_ace"] == False]

    use_ace_set = use_ace_set.pivot(index = "player_hand", columns = "show_card"
                                    , values=1).sort_index(ascending=False)
    nouse_ace_set = nouse_ace_set.pivot(index = "player_hand", columns = "show_card",
                                        values=1).sort_index(ascending=False)


    fig, ax = plt.subplots(1, 2, figsize = (20, 10))
    fig.suptitle("sum_optimal policy in {}decks".format(decks), fontsize = 16)
    ax[0].set_title("with use ace")
    ax[1].set_title("without use ace")
    color = ["k", "w", "g"]
    cmap = sns.color_palette(color, n_colors = 3)

    sns.heatmap(use_ace_set, ax = ax[0], cmap = cmap, linewidths = .5, linecolor = "lightgray",
                cbar_kws = {"ticks":[0., 0.5, 1.]})
    sns.heatmap(nouse_ace_set, ax = ax[1], cmap = cmap, linewidths = .5, linecolor = "lightgray",
                cbar_kws={"ticks": [0., 0.5, 1.]})


    fig.savefig("figures/sum_Optimal Policy in {}deck.jpg".format(decks))