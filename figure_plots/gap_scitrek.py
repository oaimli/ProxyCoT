import matplotlib.pyplot as plt
import numpy as np
def plot_performance():
    ks = ["Qwen3-4B", "Qwen3-235B", "Gemma3-4B", "Gemma3-27B", "o4-mini"]
    xs = range(1, 10, 2)
    xs_psv1 = (np.array(xs) - 0.4).tolist()
    xs_psv2 = (np.array(xs) + 0.4).tolist()
    print(xs_psv1)
    print(xs_psv2)

    performance_full = [30.8, 45.2, 3.0, 8.1, 50.0]
    performance_proxy = [67.2, 72.7, 34.2, 60.9, 91.4]

    bar_width = 0.8
    plt.figure(figsize=(9, 5))
    plt.bar(xs_psv1, height=performance_full, width=bar_width, color='deepskyblue', label="Full")
    plt.bar(xs_psv2, height=performance_proxy, width=bar_width, color='darkcyan', label="Proxy")
    plt.xticks(xs, ks, fontproperties='Times New Roman', fontsize=22, rotation=12)
    plt.yticks(fontproperties='Times New Roman', fontsize=22)
    plt.ylim(1, 100)
    plt.ylabel(r"EM (%)", fontsize=22, family='Times New Roman')
    plt.subplots_adjust(bottom=0.15)
    plt.legend(
        ncol=2,
        prop={"family": 'Times New Roman', "size": 20}
    )
    # plt.show()
    plt.savefig('gap_scitrek.png', dpi=1024)


if __name__=="__main__":
    plot_performance()