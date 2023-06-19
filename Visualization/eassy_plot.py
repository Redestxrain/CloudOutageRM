# import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成随机数据，符合指数分布
# lambdas = [1, 0.5, 0.2]
# data_list = [np.random.exponential(1/lamda, size=1000) for lamda in lambdas]
#
# # 绘制直方图
# fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(8,6))
# for ax, data, lamda in zip(axs, data_list, lambdas):
#     ax.hist(data, bins=50, density=True)
#
#     # 计算指数分布的概率密度函数
#     x = np.linspace(0, 10, 1000)
#     pdf = lamda * np.exp(-lamda * x)
#
#     # 绘制指数分布的概率密度函数，并添加图例
#     ax.plot(x, pdf, label=r'$\lambda$ = %s' % lamda)
#     ax.legend()
#
#     # 添加标题和标签
#     ax.set_title("Exponential Distribution (lambda = %s)" % lamda)
#     ax.set_xlabel("Value")
#     ax.set_ylabel("Probability Density")
#
# # 调整子图之间的间距
# fig.tight_layout()
#
# # 显示图形
# plt.show()

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 生成随机数据，符合柏拉图分布
# data = np.random.pareto(a=1, size=1000)
#
# # 使用Seaborn绘制核密度估计图
# sns.kdeplot(data, kernel='exponential', label='Pareto Distribution')
#
# # 添加标题和标签
# plt.title("Pareto Distribution")
# plt.xlabel("Value")
# plt.ylabel("Density")
#
# # 显示图形
# plt.show()

# # create the output directory if it doesn't exist
# import os
#
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from scipy.stats import expon, beta
#
# output_dir = '/Users/hsu/Downloads/論文pic'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # dtt_ddos = pd.read_csv('/Users/hsu/PycharmProjects/CloudOutageRM_v1/Sim_data/E_df.csv')
# # dtt_ddos.drop('Unnamed: 0', axis=1, inplace=True)
#
# # dt_ddos = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/OneYearDowntime_exp_without_ddos')
# dtt_ddos = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/oneYearDowntimeTimes_exp_without_ddos')


# from Risk_estimate import dtt_ddos
# dtt_ddos = dt_ctl
# #
# #


# def plot_cdf(data):
#     # params = beta.fit(data)
#     # x = np.linspace(min(data), max(data), 1000)
#     # cdf = beta.pdf(x, *params)
#     # x1 = np.linspace(beta.ppf(0.00001, *params), beta.ppf(0.99999, *params), 1000)
#     # plt.plot(x1, cdf, label=i)
#     # plt.hist(data, bins=30, density=1, cumulative=False, histtype='step',label=i)
#     # plt.xlabel('times')
#     # plt.ylabel('Probability')
#     # plt.legend(loc='upper right')
#
#     # Calculate counts and bins
#     counts, bins = np.histogram(data, bins=40)
#
#     # Normalize counts
#     counts = counts / counts.sum()
#
#     # Plot PMF
#     plt.bar(bins[:-1], counts, width=(bins[1] - bins[0]), alpha=0.7, label=i)
#
#     plt.xlabel(f'times')
#     plt.ylabel('Probability')
#     plt.legend(loc='upper right')
#
#     plt.show()


#
# # plt.figure(figsize=(8,6))
# #
# # for i in dtt_ddos.columns:
# #     plot_cdf(dtt_ddos[i])
# #     plt.title('CentruyLink Cloud Service one year downtime simulation')
# #     # save the plot to the output directory
# #     filename = 'CentruyLink Cloud Service one year downtime simulation.png'
# #     plt.savefig(os.path.join(output_dir, filename))
# # #
#
# for i in dtt_ddos.columns[:5]:
#     plot_cdf(dtt_ddos[i])
#     plt.title('Alibaba ECS one year downtime times simulation PMF')
#     # save the plot to the output directory
#     filename = 'Alibaba ECS one year downtime times simulation PDF.png'
#     plt.savefig(os.path.join(output_dir, filename))
#
# # 繪製第六、七個欄位的 cdf
# plt.figure(figsize=(8,6))
# for i in dtt_ddos.columns[5:7]:
#     plot_cdf(dtt_ddos[i])
#     plt.title('DigitalOcean one year downtime times simulation PMF')
#     filename = 'DigitalOcean one year downtime times simulation PDF.png'
#     plt.savefig(os.path.join(output_dir, filename))
#
# plt.figure(figsize=(8,6))
# for i in dtt_ddos.columns[7:8]:
#     plot_cdf(dtt_ddos[i])
#     plt.title('IBM Vitrual Private Cloud one year downtime times simulation PMF')
#     filename = 'IBM Vitrual Private Cloud one year downtime times simulation PDF.png'
#     plt.savefig(os.path.join(output_dir, filename))
#
# plt.figure(figsize=(8,6))
# for i in dtt_ddos.columns[8:13]:
#     plot_cdf(dtt_ddos[i])
#     plt.title('Azure one year downtime times simulation PMF')
#     filename = 'Azure one year downtime times simulation PDF.png'
#     plt.savefig(os.path.join(output_dir, filename))
#
# plt.figure(figsize=(8,6))
# for i in dtt_ddos.columns[13:15]:
#     plot_cdf(dtt_ddos[i])
#     plt.title('Oracle one year downtime times simulation PMF')
#     filename = 'Oracle one year downtime times simulation PDF.png'
#     plt.savefig(os.path.join(output_dir, filename))
#
# plt.figure(figsize=(8,6))
# for i in dtt_ddos.columns[15:]:
#     plot_cdf(dtt_ddos[i])
#     plt.title('Tencent one year downtime times simulation PMF')
#     filename = 'Tencent one year downtime times simulation PDF.png'
#     plt.savefig(os.path.join(output_dir, filename))

#
# def pert(a, b, c, size=1, lamb=4):  # a = min; b = most likely; c = max
#     r = c - a
#     alpha = 1 + lamb * (b - a) / r
#     betar = 1 + lamb * (c - b) / r
#
#     """
#     plt.hist(beta_we_need, 100, density=True, label=label)
#     plt.legend()
#     plt.show()"""
#     return a + np.random.beta(alpha, betar, size=size) * r
#
#
# fig = plt.figure()
# plot_cdf(pert(0, 2, 20, 100000))
# plt.title('conference fee in company A for one year')
# plt.figure()
# plot_cdf(pert(0, 83.6, 836, 100000))
# plt.title('conference fee in company B for one year')
# plt.figure()
# plot_cdf(pert(0, 6525, 65250, 100000))
# plt.title('conference fee in company C for one year')

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import beta

output_dir = '/Users/hsu/Downloads/論文pic'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file = '/Users/hsu/Fair_CyberRisk_v1/oneYearDowntimeTimes_exp_without_ddos'
# file = '/Users/hsu/Fair_CyberRisk_v1/oneYearDowntimeTimes_exp_without_ddos_120_CTL'
dtt_ddos = pd.read_csv(file)


def plot_cdf(data, title, filename, provider_location, save_fig=True):
    counts, bins = np.histogram(data, bins=40)
    counts = counts / counts.sum()
    plt.bar(bins[:-1], counts, width=(bins[1] - bins[0]), alpha=0.7, label=provider_location)
    plt.xlabel(f'times')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    plt.title(title)
    if save_fig:
        plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def pert(a, b, c, size=1, lamb=4):  # a = min; b = most likely; c = max
    r = c - a
    alpha = 1 + lamb * (b - a) / r
    betar = 1 + lamb * (c - b) / r
    return a + np.random.beta(alpha, betar, size=size) * r


def plot_company_fee(a, b, c, title, provider_location, save_fig=True):
    plt.figure()
    data = pert(a, b, c, 100000)
    params = beta.fit(data)
    x = np.linspace(min(data), max(data), 1000)
    cdf = beta.pdf(x, *params)
    x1 = np.linspace(beta.ppf(0.00001, *params), beta.ppf(0.99999, *params), 1000)
    plt.plot(x1, cdf, label=provider_location)
    plt.xlabel(f'Conference fee (thousand $)')
    plt.ylabel('Probability')
    plt.title(title)
    if save_fig:
        plt.savefig(os.path.join(output_dir, title + '.png'))


plot_company_fee(0, 2, 20, 'conference fee PDF in company A for one year', 'company A', save_fig=True)
plot_company_fee(0, 83.6, 836, 'conference fee PDf in company B for one year', 'company B', save_fig=True)
plot_company_fee(0, 6525, 65250, 'conference fee PDF in company C for one year', 'company C', save_fig=True)

# Define a dictionary to map column ranges to titles and filenames
plot_info = {
    (0, 5): (
        'Alibaba ECS one year downtime times simulation PMF', 'Alibaba ECS one year downtime times simulation PDF.png'),
    (5, 7): (
        'DigitalOcean one year downtime times simulation PMF',
        'DigitalOcean one year downtime times simulation PDF.png'),
    (7, 8): ('IBM Vitrual Private Cloud one year downtime times simulation PMF',
             'IBM Vitrual Private Cloud one year downtime times simulation PDF.png'),
    (8, 13): ('Azure one year downtime times simulation PMF', 'Azure one year downtime times simulation PDF.png'),
    (13, 15): ('Oracle one year downtime times simulation PMF', 'Oracle one year downtime times simulation PDF.png'),
    (15, None): (
        'Tencent one year downtime times simulation PMF', 'Tencent one year downtime times simulation PDF.png'),
}

plot_info_ctl = {
    (0, None): (
        'CenturyLink Cloud Servers one year downtime times simulation PMF',
        'CenturyLink Cloud Servers one year downtime times simulation PDF.png'),
}

# Loop over the plot_info dictionary and plot each range of columns
# for col_range, (title, filename) in plot_info.items():
#     start, end = col_range
#     plt.figure(figsize=(8, 6))
#     for i in dtt_ddos.columns[start:end]:
#         plot_cdf(dtt_ddos[i], title, filename, save_fig=False, provider_location=i)