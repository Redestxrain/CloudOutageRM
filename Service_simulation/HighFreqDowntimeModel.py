import math
import random
import numpy as np
import pandas as pd
from scipy.stats import expon, kstest, laplace
from itertools import chain
from tqdm import trange
import DowntimeInfo as dti


class bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


class HighFreqDowntimeModel(dti.Downtime):

    def __init__(self):
        super().__init__()
        self.sim_num = 8000

    def test(self):

        ServiceName = self.new_COD.groupby('Service Name')
        exp_shape_df = pd.DataFrame()
        one_year_downtime = pd.DataFrame()
        one_year_downtime_times = pd.DataFrame()

        S_df = pd.DataFrame()
        E_df = pd.DataFrame()
        for SPN in ServiceName.size().index:
            if SPN == 'CenturyLink Cloud Servers':

                CTL = ServiceName.get_group(SPN)
                CTL.downtime.sum()
                CTL_place = CTL.groupby('place')
                places = CTL_place.size().index

                for p in places:
                    P = CTL_place.get_group(p)
                    P = P.sort_values(by=['end time'])
                    provider_place = f'{SPN}: {p}'
                    P_end_life = P[P.downtime <= 120]
                    P_ddos = P[P.downtime > 120]
                    if len(P_end_life) < 5:
                        continue
                    dt_ratio = len(P_ddos) / len(P_end_life)

                    end_life_sim = []
                    downtime_sim = []
                    for P_df in [P_end_life, P_ddos]:  # this part is for outage time
                        if len(P_df) < 5:
                            continue

                        if P_df is P_end_life:
                            rand_num = int(self.sim_num * dt_ratio)
                        else:
                            rand_num = self.sim_num
                        params = expon.fit(P_df.downtime)
                        if params == (0, 0):
                            continue
                        e = expon(*params)
                        ks = kstest(P_df.downtime, e.cdf)

                        if ks.pvalue < 0.05:
                            laplace_params = laplace.fit(P_df.downtime)
                            l = laplace(*laplace_params)
                            ks = kstest(P_df.downtime, l.cdf)
                            if ks.pvalue > 0.05:
                                end_life_sim = laplace.ppf(np.random.rand(rand_num), *laplace_params)
                                # print(f'{bcolors.OK}{SPN} {p} downtime :{ks}{bcolors.RESET}')
                            else:
                                end_life_sim = expon.ppf(np.random.rand(rand_num), *params)
                        else:
                            end_life_sim = expon.ppf(np.random.rand(rand_num), *params)
                            # print(f'{bcolors.OK}{SPN} {p} downtime :{ks}{bcolors.RESET}')

                        outlier = 43200
                        end_life_sim[end_life_sim > outlier] = 43200  # clear outlier
                        downtime_sim = np.append(downtime_sim, end_life_sim)

                    random.shuffle(downtime_sim)
                    downtime_sim = downtime_sim[:self.sim_num]
                    S_df[provider_place] = downtime_sim

                    P_service_long = P[P.service_time.dt.total_seconds() / 60 > 120]
                    P_service_short = P[P.service_time.dt.total_seconds() / 60 <= 120]

                    ls_sim = []
                    service_time_sim = []
                    if len(P_service_short) < 1:
                        continue
                    ls_ratio = len(P_service_long) / len(P_service_short)

                    for P_df in [P_service_short, P_service_long]:  # this part is for outage time

                        if len(P_df) < 5:
                            continue

                        if P_df is P_service_short:
                            rand_num = int(self.sim_num * ls_ratio)
                        else:
                            rand_num = self.sim_num
                        params = expon.fit(P_df.service_time.dt.total_seconds() / 60)
                        if params == (0, 0):
                            continue
                        e = expon(*params)
                        ks = kstest(P_df.service_time.dt.total_seconds() / 60, e.cdf)

                        if ks.pvalue < 0.05:
                            laplace_params = laplace.fit(P_df.service_time.dt.total_seconds() / 60)
                            l = laplace(*laplace_params)
                            ks = kstest(P_df.service_time.dt.total_seconds() / 60, l.cdf)
                            if ks.pvalue > 0.05:
                                ls_sim = laplace.ppf(np.random.rand(rand_num), *laplace_params)
                                # print(f'{bcolors.OK}{SPN} {p} service time :{ks}{bcolors.RESET}')
                            else:
                                ls_sim = expon.ppf(np.random.rand(rand_num), *params)
                        else:
                            ls_sim = expon.ppf(np.random.rand(rand_num), *params)
                            # print(f'{bcolors.OK}{SPN} {p} service time :{ks}{bcolors.RESET}')

                        outlier = 43200
                        ls_sim[ls_sim > outlier] = 43200  # clear outlier

                        service_time_sim = np.append(service_time_sim, ls_sim)
                    random.shuffle(service_time_sim)
                    service_time_sim = service_time_sim[:self.sim_num]
                    E_df[provider_place] = service_time_sim  # service time

        return S_df, E_df  # downtime, service time

    def monte_carlo(self, *args):
        df1 = args[0]
        df2 = args[1]

        df = pd.merge(df1, df2, how='outer')
        df.dropna(axis=1, inplace=True)
        df_split = df.iloc[self.sim_num:]
        df = df.drop(df_split.index)

        time_set = 120
        df = df[df <= time_set]
        result = pd.DataFrame(list(chain.from_iterable(zip(df.values.tolist(), df_split.values.tolist()))))
        # 設置新 DataFrame 的列名
        result.columns = df.columns

        one_year_downtime_list = []
        one_year_downtime_times_list = []

        for i in df.columns:
            for j in range(self.sim_num):
                if result[i].iloc[:j + 1].sum() >= (60 * 24 * 365):
                    # print('ok')
                    down_place = math.floor((j + 1) / 2)
                    # print(int((j + 1) / 2))
                    one_year_downtime_list.append(df[i].iloc[:down_place].sum())
                    null_num = df[i].iloc[:down_place].isnull().sum()
                    one_year_downtime_times_list.append(down_place + 1 - null_num)
                    break
        return pd.DataFrame(one_year_downtime_list, index=df.columns).T, \
               pd.DataFrame(one_year_downtime_times_list, index=df.columns).T


if __name__ == '__main__':
    a = HighFreqDowntimeModel().test()
    b = HighFreqDowntimeModel().monte_carlo(*a)
    high_freq_downtime = b[0]
    high_freq_downtime_times = b[1]
    for i in trange(1000-1):
        a = HighFreqDowntimeModel().test()
        b = HighFreqDowntimeModel().monte_carlo(*a)
        high_freq_downtime = pd.concat([high_freq_downtime, b[0]], axis=0, ignore_index=True)
        high_freq_downtime_times = pd.concat([high_freq_downtime_times, b[1]], axis=0, ignore_index=True)

    high_freq_downtime_times.to_csv(f'/Users/hsu/Downloads/high_freq_downtime_times_exp_2', index=False)
    high_freq_downtime.to_csv(f'/Users/hsu/Downloads/high_freq_downtime_exp_2', index=False)