import glob, os
import pandas as pd
import pytz
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn import preprocessing
from tqdm import tqdm, trange
import math
from scipy.stats import pareto, kstest, expon

file_path = '//CloudOutageComputingData'
os.chdir(file_path)
files = [file for file in glob.glob('*.csv')]
files.reverse()


class bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


COD = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
COD = COD[COD.place != 'Loading']

TZINFOS = {
    'PDT': pytz.timezone('US/Pacific'),
    # ... add more to handle other timezones
    # (I wish pytz had a list of common abbreviations)
}
datetime_in_pdt = dateutil.parser.parse(COD['start time'][0], tzinfos=TZINFOS)
datetime_in_utc = datetime_in_pdt.astimezone(pytz.utc)
datetime_naive = datetime_in_utc.replace(tzinfo=None)
newCOD = COD.copy()

# Service_Name = COD.groupby('Service Name')
# service_size = Service_Name.size()
# service_downtime = [np.array(Service_Name.get_group(SEV)['downtime']) for SEV in service_size.index]
#
# n_bins = 20
# fig, ax = plt.subplots(nrows=1, ncols=1)
#
# ax.hist(service_downtime, n_bins, histtype='bar', label=service_size.index)
# ax.set_title(f'compute service downtime', fontsize='x-large', fontfamily='sans-serif', fontstyle='italic')
# ax.legend(prop={'size': 10})
# plt.semilogy(basey=10)
# ax.set_xlabel('Downtime $_{min}$', fontsize='x-large', fontfamily='sans-serif', fontstyle='italic')
# ax.set_ylabel('times', fontsize='x-large', fontfamily='sans-serif', fontstyle='italic')
# plt.show()


for i in COD.index:
    for j in ['start time', 'end time']:
        TZINFOS = {
            'PDT': pytz.timezone('US/Pacific'),
            # ... add more to handle other timezones
            # (I wish pytz had a list of common abbreviations)
        }
        try:
            datetime_in_pdt = dateutil.parser.parse(COD[j][i], tzinfos=TZINFOS)
            datetime_in_utc = datetime_in_pdt.astimezone(pytz.utc)
            datetime_naive = datetime_in_utc.replace(tzinfo=None)
            newCOD.loc[i, j] = datetime_naive
        except dateutil.parser._parser.ParserError:
            print(f'{bcolors.FAIL}'
                  f'dateutil.parser._parser.ParserError: Unknown string format: outages... Specific outage '
                  f'records are'
                  f'{bcolors.RESET}')

# mask_mistake = newCOD['Service Name'].str.startwith('o')

newCOD['start time'] = pd.to_datetime(newCOD['start time'])
newCOD['end time'] = pd.to_datetime(newCOD['end time'])
ServiceName = newCOD.groupby('Service Name')

std_mean_dict = {'std': [], 'mean': [], 'place': [], 'service provider': []}


# SPN: service provider name


def Split(x):
    return x.split(':')[0]


def Split2(x):
    return x.split(':')[1]


# when use expon than use this
shape_df = pd.DataFrame(index=['loc', 'scale'])
# shape_df = pd.DataFrame(index=['b', 'loc', 'scale'])
exp_shape_df = pd.DataFrame(index=['loc', 'scale'])
response = requests.get('https://cloudharmony.com/status-1week')
soup = BeautifulSoup(response.text, "html.parser")
data = soup.select('tbody tr')

SIM = False

if SIM is True:
    one_year_downtime = pd.DataFrame()
    one_year_downtime_times = pd.DataFrame()
    for time_set in [120, ]:
        for n in trange(1):

            S_df = pd.DataFrame()
            E_df = pd.DataFrame()
            for SPN in ServiceName.size().index:

                CTL = ServiceName.get_group(SPN)
                CTL.downtime.sum()
                CTL_place = CTL.groupby('place')
                places = CTL_place.size().index

                for p in places:
                    P = CTL_place.get_group(p)
                    P = P.sort_values(by=['end time'])
                    P['service_time'] = P['end time'].shift(-1) - P['end time']
                    P['service_time'] = P.service_time.fillna(
                        pd.to_datetime('2023-02-28 00:00:00') - P['end time'].iloc[-1])
                    # print(f'{p} mean {P.downtime.mean()}')

                    if len(P) < 3:
                        continue
                    std_mean_dict['mean'].append(P.downtime.mean())
                    std_mean_dict['std'].append(P.downtime.std())
                    std_mean_dict['place'].append(p)
                    std_mean_dict['service provider'].append(SPN)

                    if SPN != 'CenturyLink Cloud Servers':
                        provider_place = f'{SPN}: {p}'
                        params = expon.fit(P.downtime)
                        shape_df[provider_place] = params
                        shape_df.clip(lower=0, inplace=True)
                        # fig, ax = plt.subplots(1, 1)
                        # ax.hist(P.downtime, bins=50, density=True, alpha=0.6, color='g')
                        # x = np.linspace(P.downtime.min(), P.downtime.max(), 1000)
                        pareto_sim = expon.ppf(np.random.rand(4000), *params)
                        # clear outlier
                        outlier = 43200
                        pareto_sim[pareto_sim > outlier] = np.nan

                        S_df[provider_place] = pareto_sim
                        # ax.plot(x, pareto.pdf(x, *params), 'r-', lw=2, alpha=0.8, label='Pareto fit')
                        # plt.title(provider_place)
                        # ax.legend()
                        # plt.show()

                        exp_params = expon.fit(P.service_time.dt.total_seconds() / 60)
                        exp_shape_df[provider_place] = exp_params
                        exp_sim = expon.ppf(np.random.rand(4000), *exp_params)
                        E_df[provider_place] = exp_sim

            df1 = E_df
            df2 = S_df
            df2 = df2[df2 <= time_set]  # df2 是斷線時間

            df = pd.concat([df1, df2], axis=0)

            for i, row in enumerate(df2.iterrows()):
                df.iloc[2 * i + 1] = row[1]

            one_year_downtime_list = []
            one_year_downtime_times_list = []

            for i in df.columns:
                for j in range(8000):
                    if df[i].iloc[:j + 1].sum() >= (60 * 24 * 365):
                        down_place = math.floor((j + 1) / 2)
                        # print(int((j + 1) / 2))
                        one_year_downtime_list.append(df2[i].iloc[:down_place].sum())
                        null_num = df2[i].iloc[:down_place].isnull().sum()
                        one_year_downtime_times_list.append(down_place + 1 - null_num)
                        break
            one_year_downtime[f'{n}'] = one_year_downtime_list
            one_year_downtime_times[f'{n}'] = one_year_downtime_times_list

            # COD = pd.concat([COD, P])

        S_df.hist(bins=30,
                  density=True,
                  cumulative=True,
                  histtype=u'step'
                  )

        E_df.hist(bins=30,
                  density=True,
                  cumulative=True,
                  histtype=u'step'
                  )

        # std_mean_df = pd.DataFrame(std_mean_dict)
        # std_mean_df.dropna(inplace=True)
        # std_mean_df = std_mean_df.sort_values(by=['mean'])
        # std_mean_df = std_mean_df.sort_values(by=['service provider'])
        # ServiceName = COD.groupby('Service Name')
        # std_mean_df_latex = std_mean_df.to_latex(index=False)
        # std_mean_df = std_mean_df[std_mean_df['service provider'] != 'CenturyLink Cloud Servers']

        shape_df.dropna(inplace=True)
        shape_df = shape_df.transpose()
        shape_df.reset_index(inplace=True)
        shape_df = shape_df.sort_values(by=['index'])
        shape_df_latex = shape_df.to_latex(index=False)

        # exp_shape_df.dropna(inplace=True)
        # exp_shape_df = exp_shape_df.transpose()
        # exp_shape_df.reset_index(inplace=True)
        # exp_shape_df = exp_shape_df.sort_values(by=['index'])
        # exp_shape_df_latex = exp_shape_df.to_latex(index=False)
        #
        one_year_downtime.index = shape_df['index']
        one_year_downtime_times.index = shape_df['index']
        one_year_downtime_times = one_year_downtime_times.T
        one_year_downtime = one_year_downtime.T
        #
        # s_c = one_year_downtime * 0.01
        # m_c = one_year_downtime * 18.57
        # b_c = one_year_downtime * 365.72
        #
        # sc_df = pd.DataFrame(s_c.mean(), columns=['A-mean'])
        # mc_df = pd.DataFrame(m_c.mean(), columns=['B-mean'])
        # bc_df = pd.DataFrame(b_c.mean(), columns=['C-mean'])
        #
        # sc_df['A-std'] = s_c.std().to_list()
        # mc_df['B-std'] = m_c.std().to_list()
        # bc_df['C-std'] = b_c.std().to_list()
        #
        # sc_df_latex = sc_df.to_latex()
        # mc_df_latex = mc_df.to_latex()
        # bc_df_latex = bc_df.to_latex()
    one_year_downtime.to_csv(f'/Users/hsu/Downloads/OneYearDowntime_exp_without_ddos_{time_set}_CTL', index=False)
    one_year_downtime_times.to_csv(f'/Users/hsu/Downloads/oneYearDowntimeTimes_exp_without_ddos_{time_set}_CTL',
                                   index=False)
# S_list = []
# for i in std_mean_df.index:
#     std, mean = std_mean_df.loc[i]['std'], std_mean_df.loc[i]['mean']
#     # 計算 Pareto 分佈的形狀參數 alpha
#     alpha = (mean / std) ** 2
#
#     # 設定 Pareto 分佈的最小值
#     xmin = mean - (alpha * mean) / alpha
#
#     # 產生 Pareto 分佈資料
#     pp = pareto.rvs(alpha, loc=xmin, size=10000)
#     S_list.append(pp)
#
# std_mean_df['sim_data'] = S_list
# std_mean_df['service provider place'] = std_mean_df.apply(lambda x: x['service provider'] + ' ' + x['place'], axis=1)
# S_df = pd.DataFrame(S_list).transpose()
# S_Df = S_df.set_axis(std_mean_df['service provider place'].to_list(), axis=1, inplace=False)


AS_part = True

if AS_part is True:
    def to_value(x):
        return x.days * 60 + x.seconds / 3600


    def FH_index(a, b, asset: np.array) -> int:
        FH_try = np.linspace(a, b, 10001)
        diff_list = []
        temp = 0
        for FH in FH_try:
            a = 1 + asset / FH
            if np.all(a > 0):
                diff = np.mean(np.log(1 + (asset / FH))) - temp
                diff_list.append(diff)
                if abs(diff) < 0.1:
                    break
        if FH == b:
            plt.figure()
            plt.plot(diff_list)
        return FH


    def AS_index(a, b, asset: np.array) -> int:
        AS_try = np.linspace(a, b, 100001)
        diff_list2 = []
        for AS in AS_try:
            temp = np.mean(np.exp(-asset / AS))
            diff_list2.append(abs(temp - 1))
            if abs(temp - 1) < 0.01:
                break
        if AS == b:
            plt.figure()
            plt.plot(diff_list2)
        print(temp)
        return AS


    all_service = []
    for i in range(len(data)):
        try:
            if data[i].attrs:
                all_service.append(data[i].attrs['id'])
                q = i
            else:
                all_service.append(data[q].attrs['id'])
        except Exception as E:
            print(E)

    new_COD = pd.DataFrame()
    for SPN in ServiceName.size().index:

        CTL = ServiceName.get_group(SPN)
        CTL.downtime.sum()
        CTL_place = CTL.groupby('place')
        places = CTL_place.size().index

        for p in places:
            P = CTL_place.get_group(p)
            P = P.sort_values(by=['end time'])
            P['service_time'] = P['end time'].shift(-1) - P['end time']
            P['service_time'] = P.service_time.fillna(
                pd.to_datetime('2023-02-28 00:00:00') - P['end time'].iloc[-1])
            new_COD = pd.concat([new_COD, P], axis=0)

    all_service = pd.DataFrame(all_service, columns=['name'])
    all_service['func'] = all_service.name.apply(Split2)
    all_service['name'] = all_service.name.apply(Split)
    s = all_service[all_service.func == 'storage'].index[0]
    all_service = all_service[all_service.index < s]
    all_service = all_service.replace('aws', 'amazon')
    all_service = all_service.replace('azure', 'microsoft')
    N = all_service.groupby('name')

    ST_list = []
    service_provider = []

    ServiceName = new_COD.groupby('Service Name')

    AS_df_service = []
    AS_df_R = []
    plot_list= []
    label_list = []
    for SPN in ServiceName.size().index:
        SPN_server_num = N.size()[SPN.lower().split()[0]]
        CTL = ServiceName.get_group(SPN)
        CTL_place = CTL.groupby('place')
        down_server_num = len(CTL_place.size())
        CTL = CTL.sort_values(by=['service_time'])
        CTL['service_time'] = CTL.service_time.apply(to_value)
        ST = CTL['service_time'].tolist()
        notDown = SPN_server_num - down_server_num
        if notDown > 0:
            t = to_value(pd.to_datetime('2023-01-28 00:00:00') - pd.to_datetime('2022-07-31 00:00:00'))
            for i in range(notDown):
                ST.append(t)
        CTL = CTL.sort_values(by=['downtime'])
        CTL.downtime = -CTL.downtime / 60
        ST = np.array(ST + CTL['downtime'].tolist())
        print(f'downtime damage:{CTL.downtime.sum() * -365720}')
        ST.sort()
        # high = np.arange(1, len(ST)) / len(ST)
        # ST_diff = ST[1:] - ST[:-1]
        # New_ST = high * ST_diff
        # New_ST = np.cumsum(New_ST)
        # New_ST = New_ST / len(New_ST)
        # plt.plot(ST[:-1],New_ST, label=CTL['Service Name'].iloc[0])
        # plt.legend(loc='upper left')
        # plt.xlabel('Service Time and downtime (min)')
        # plt.ylabel('Cumulative Probability')
        # plt.title('one order stochastic dominance')

    #     a = plt.hist(ST,
    #                  bins=200,
    #                  density=True,
    #                  cumulative=True,
    #                  histtype=u'step',
    #                  label=CTL['Service Name'].iloc[0])
    #     plot_list.append(a)
    #     label_list.append(CTL['Service Name'].iloc[0])
    #
    #     plt.legend(loc='lower right')
    #     plt.xlabel('Service Time and downtime (min)')
    #     plt.ylabel('Cumulative Probability')
    #     plt.title('one order stochastic dominance')
    #
    # plt.figure()
    # for j,z in zip(plot_list,label_list):
    #     x = j[0]
    #     x = np.append(x, 1)
    #     plt.plot(j[1],x, label=z)
    #     plt.legend(loc='lower right')
    #     plt.xlabel('Service Time and downtime (min)')
    #     plt.ylabel('Cumulative Probability')
    #     plt.title('one order stochastic dominance')
        # print(f'{CTL["Service Name"].iloc[0]}: {FH_index(1e-60, 10, ST_new)}')
        R = AS_index(0.001, 25, ST)
        AS_df_service.append(CTL['Service Name'].iloc[0])
        AS_df_R.append(R*54.2)
        print(f'{CTL["Service Name"].iloc[0]}: {R}')
        print(f'alpha: {1 / R}')

    AS_df = pd.DataFrame()
    AS_df['service'] = AS_df_service
    AS_df['R'] = AS_df_R
