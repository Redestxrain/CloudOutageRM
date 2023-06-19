import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import expon

dtt = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/oneYearDowntimeTimes_exp')
dtt.drop(dtt.columns[0], axis=1, inplace=True)
dtt2 = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/oneYearDowntimeTimes')
dt2 = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/OneYearDowntime')

dt = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/OneYearDowntime_exp')
dt.drop(dt.columns[0], axis=1, inplace=True)

dt_ddos = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/OneYearDowntime_exp_without_ddos')
dtt_ddos = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/oneYearDowntimeTimes_exp_without_ddos')

dt_hfdm = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/high_freq_downtime_exp_2')
dtt_hfdm = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/high_freq_downtime_times_exp_2')
#
dt_ctl = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/OneYearDowntime_exp_without_ddos_120_CTL')
dtt_ctl = pd.read_csv('/Users/hsu/Fair_CyberRisk_v1/oneYearDowntimeTimes_exp_without_ddos_120_CTL')


def pert(a, b, c, size=1, lamb=4):  # a = min; b = most likely; c = max
    r = c - a
    alpha = 1 + lamb * (b - a) / r
    betar = 1 + lamb * (c - b) / r

    """
    plt.hist(beta_we_need, 100, density=True, label=label)
    plt.legend()
    plt.show()"""
    return a + np.random.beta(alpha, betar, size=size) * r


class LostEstimate:

    def __init__(self, data_downtime, data_downtimes):
        self.data = data_downtimes
        self.Data = data_downtime

        self.data_l = pd.DataFrame()
        self.data_m = pd.DataFrame()
        self.data_s = pd.DataFrame()

        self.data_kf_s = pd.DataFrame()
        self.data_kf_m = pd.DataFrame()
        self.data_kf_l = pd.DataFrame()

        self.data__s = pd.DataFrame()
        self.data__m = pd.DataFrame()
        self.data__l = pd.DataFrame()

        self.second_loss_event_freq = pd.DataFrame()

    @staticmethod
    def df_pert(x, a, b, c):
        return pert(a, b, c, x).sum()

    def simulation(self):
        for i in self.data.columns:
            self.data_s[i] = self.data[i].apply(self.df_pert, args=(0, 2, 20))
            self.data_m[i] = self.data[i].apply(self.df_pert, args=(0, 83.6, 836))
            self.data_l[i] = self.data[i].apply(self.df_pert, args=(0, 6525, 65250))
            self.data_kf_s[i] = self.data[i].apply(self.df_pert, args=(0, 15, 150))
            self.data_kf_m[i] = self.data[i].apply(self.df_pert, args=(0, 480, 4800))
            self.data_kf_l[i] = self.data[i].apply(self.df_pert, args=(0, 4000, 40000))
            self.data__s[i] = self.data[i].apply(self.df_pert, args=(0, 105, 450))
            self.data__m[i] = self.data[i].apply(self.df_pert, args=(0, 191292.77, 819826.14))
            self.data__l[i] = self.data[i].apply(self.df_pert, args=(0, 4305765.05, 18453278.78))
            self.second_loss_event_freq[i] = pert(0, 0.001, 0.01, 1000)

        return self.data__s, self.data__m, self.data__l

    def Lost(self):
        second_lose_s = self.data_kf_s + self.data__s
        second_lose_m = self.data_kf_m + self.data__m
        second_lose_l = self.data_kf_l + self.data__l

        risk_s = self.Data * 0.02 + self.data_s + second_lose_s * self.second_loss_event_freq
        risk_m = self.Data * 0.472 + self.data_m + second_lose_m * self.second_loss_event_freq
        risk_l = self.Data * 54.2 + self.data_l + second_lose_l * self.second_loss_event_freq
        return risk_s, risk_m, risk_l


# a = LostEstimate(dt, dtt)
# a.simulation()
# risk = a.Lost()
# risk_df = pd.DataFrame()
# risk_df['small'] = risk[0].mean()
# risk_df['medium'] = risk[1].mean()
# risk_df['large'] = risk[2].mean()
# # risk_df.to_excel('/Users/hsu/Downloads/risk.xlsx')
#
# b = LostEstimate(dt2, dtt2)
# b.simulation()
# risk2 = b.Lost()
# risk_df2 = pd.DataFrame()
# risk_df2['small'] = risk2[0].mean()
# risk_df2['medium'] = risk2[1].mean()
# risk_df2['large'] = risk2[2].mean()
# risk_df2.to_excel('/Users/hsu/Downloads/risk2.xlsx')

c = LostEstimate(dt_ddos, dtt_ddos)
C = c.simulation()
risk3 = c.Lost()
risk_df3 = pd.DataFrame()
risk_df3['A'] = risk3[0].mean()
risk_df3['B'] = risk3[1].mean()
risk_df3['C'] = risk3[2].mean()
# # risk_df3.to_excel('/Users/hsu/Downloads/risk3.xlsx')

# d = LostEstimate(dt_hfdm, dtt_hfdm)
# d.simulation()
# risk4 = d.Lost()
# risk_df4 = pd.DataFrame()
# risk_df4['small'] = risk4[0].mean()
# risk_df4['medium'] = risk4[1].mean()
# risk_df4['large'] = risk4[2].mean()
# risk_df4.to_excel('/Users/hsu/Downloads/risk4.xlsx')

e = LostEstimate(dt_ctl, dtt_ctl)
E = e.simulation()
risk5 = e.Lost()
risk_df5 = pd.DataFrame()
risk_df5['A'] = risk5[0].mean()
risk_df5['B'] = risk5[1].mean()
risk_df5['C'] = risk5[2].mean()
# risk_df5.to_excel('/Users/hsu/Downloads/risk5.xlsx')





