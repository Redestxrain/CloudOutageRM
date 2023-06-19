import time
import glob, os
from typing import Tuple
from tqdm import tqdm, trange
import pandas as pd
import pytz
import dateutil.parser
from pandas import DataFrame
from scipy.stats import poisson, expon, kstest, laplace
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import datetime


class bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


file_path = '//'
os.chdir(file_path)
files = [file for file in glob.glob('*.csv')]
files.reverse()


def main(PLOT=False, SP=2) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:

    # def plotting
    def distribution_plot(distribution, Scale=None, Loc=None):

        x = np.linspace(distribution.ppf(0.01, scale=Scale, loc=Loc),
                        distribution.ppf(0.99, scale=Scale, loc=Loc),
                        100)
        ax.plot(x,
                distribution.cdf(x, scale=Scale, loc=Loc),
                'r-', alpha=0.6,
                label='simulate service time')

        ax.set_xlabel(f'time (unit: hours)')
        ax.set_ylabel(f'probability')
        ax.legend()

        return distribution.cdf(x, scale=Scale, loc=Loc)

    COD = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

    TZINFOS = {
        'PDT': pytz.timezone('US/Pacific'),
        # ... add more to handle other timezones
        # (I wish pytz had a list of common abbreviations)
    }
    datetime_in_pdt = dateutil.parser.parse(COD['start time'][0], tzinfos=TZINFOS)
    datetime_in_utc = datetime_in_pdt.astimezone(pytz.utc)
    datetime_naive = datetime_in_utc.replace(tzinfo=None)
    newCOD = COD.copy()

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

    Place_df = pd.DataFrame()
    ServiceName = newCOD.groupby('Service Name')

    # SPN: service provider name
    SPN = ServiceName.size().index[SP]
    global CTL

    CTL = ServiceName.get_group(SPN)
    CTL_place = CTL.groupby('place')

    # place: where the service is
    places = CTL_place.size().index

    downtime_stat = {'place': [], 'scale': [], 'Loc': [], 'times': [], 'laplace': []}
    downtime_stat_short = {'place': [], 'scale': [], 'Loc': [], 'times': [], 'laplace': []}
    service_time_stat_low = {'place': [], 'scale': [], 'Loc': [], 'times': []}
    service_time_stat_high = {'place': [], 'scale': [], 'Loc': [], 'times': []}

    for place in places:
        for i in range(2):
            if i == 0:
                Place = CTL_place.get_group(place)
                Place = Place.sort_values("start time", ascending=True).drop_duplicates(keep='first')
                temp = Place['start time'] - Place['end time'].shift(1)
                Place = Place[Place.downtime >= 30]
            if i == 1:
                Place = CTL_place.get_group(place)
                Place = Place.sort_values("start time", ascending=True).drop_duplicates(keep='first')
                temp = Place['start time'] - Place['end time'].shift(1)
                Place = Place[Place.downtime < 30]
                print(Place)
            Place_df = temp
            Place_df = Place_df.dropna()

            # test for downtime
            if len(Place) > 5:  # 調整樣本數量
                Place_downtime = Place.downtime
                outlier_downtime = Place_downtime.mean() + 2 * Place_downtime.std()
                Place_downtime = Place_downtime[Place_downtime < outlier_downtime]
                loc, scale = expon.fit(Place_downtime)
                e = expon(loc=loc, scale=scale)
                KS_result = kstest(Place_downtime, e.cdf)
                if i == 0:
                    if KS_result.pvalue > 0.05:
                        downtime_stat['place'].append(place)
                        downtime_stat['scale'].append(scale)
                        downtime_stat['Loc'].append(loc)
                        downtime_stat['times'].append(len(Place_downtime))
                        downtime_stat['laplace'].append(False)
                        print(f'{bcolors.OK}{SPN} {place} for downtime pass:{KS_result}{bcolors.RESET}')
                        if PLOT is True:
                            fig, ax = plt.subplots(1, 1)
                            Place_downtime.hist(bins=200,
                                                density=True,
                                                cumulative=True,
                                                histtype=u'step',
                                                label='actual service time')
                            distribution_plot(expon, Scale=scale, Loc=loc)
                            ax.set_title(f'{SPN} {place} low frequency outages time CDF (follow expon distribution)')

                    if KS_result.pvalue < 0.05:
                        print(f'{bcolors.FAIL}{SPN} {place} for downtime not pass:{KS_result}{bcolors.RESET}')
                        if PLOT is True:
                            Place_downtime.hist(bins=200,
                                                density=True,
                                                cumulative=True,
                                                histtype=u'step',
                                                label='actual service time')
                        # those not follow expon, try double expon
                        loc, scale = laplace.fit(Place_downtime)
                        fig, ax = plt.subplots(1, 1)
                        l = laplace(loc=loc, scale=scale)
                        x = np.linspace(l.ppf(0.01),
                                        l.ppf(0.99),
                                        100)
                        ax.plot(x,
                                l.cdf(x),
                                'r-', alpha=0.6,
                                label='downtime')
                        KS_result_laplace = kstest(Place_downtime, l.cdf)
                        if KS_result_laplace.pvalue >= 0.05:
                            print(f'{bcolors.OK}{SPN} {place} for downtime (laplace):'
                                  f'{KS_result_laplace}{bcolors.RESET}')
                            downtime_stat['place'].append(f'{place}')
                            downtime_stat['scale'].append(scale)
                            downtime_stat['Loc'].append(loc)
                            downtime_stat['times'].append(len(Place_downtime))
                            downtime_stat['laplace'].append(True)
                        else:
                            print(f'{bcolors.FAIL}{SPN} {place} for downtime (laplace):'
                                  f'{KS_result_laplace}{bcolors.RESET}')
                            loc, scale = expon.fit(Place_downtime)
                            downtime_stat['place'].append(f'{place}')
                            downtime_stat['scale'].append(scale)
                            downtime_stat['Loc'].append(loc)
                            downtime_stat['times'].append(len(Place_downtime))
                            downtime_stat['laplace'].append(False)

                elif i == 1:
                    if KS_result.pvalue > 0.05:
                        downtime_stat_short['place'].append(place)
                        downtime_stat_short['scale'].append(scale)
                        downtime_stat_short['Loc'].append(loc)
                        downtime_stat_short['times'].append(len(Place_downtime))
                        downtime_stat_short['laplace'].append(False)
                        if PLOT is True:
                            fig, ax = plt.subplots(1, 1)
                            Place_downtime.hist(bins=200,
                                                density=True,
                                                cumulative=True,
                                                histtype=u'step',
                                                label='actual service time')
                            distribution_plot(expon, Scale=scale, Loc=loc)
                            ax.set_title(f'{SPN} {place} low frequency outages time CDF (follow expon distribution)')

                        print(f'{bcolors.OK}{SPN} {place} for downtime pass short:{KS_result}{bcolors.RESET}')

                    if KS_result.pvalue <= 0.05:

                        print(f'{bcolors.FAIL}{SPN} {place} for downtime not pass short:{KS_result}{bcolors.RESET}')

                        if PLOT is True:
                            Place_downtime.hist(bins=200,
                                                density=True,
                                                cumulative=True,
                                                histtype=u'step',
                                                label='actual service time')
                        # those not follow expon, try double expon
                        loc, scale = laplace.fit(Place_downtime)
                        l = laplace(loc=loc, scale=scale)
                        KS_result_laplace = kstest(Place_downtime, l.cdf)

                        if KS_result_laplace.pvalue >= 0.05:

                            print(
                                f'{bcolors.OK}{SPN} {place} for downtime (laplace) short:'
                                f'{KS_result_laplace}{bcolors.RESET}')
                            downtime_stat_short['place'].append(f'{place}')
                            downtime_stat_short['scale'].append(scale)
                            downtime_stat_short['Loc'].append(loc)
                            downtime_stat_short['times'].append(len(Place_downtime))
                            downtime_stat_short['laplace'].append(True)

                        else:

                            print(
                                f'{bcolors.FAIL}{SPN} {place} for downtime (laplace):'
                                f'{KS_result_laplace}{bcolors.RESET}')
                            loc, scale = expon.fit(Place_downtime)
                            downtime_stat['place'].append(f'{place}')
                            downtime_stat['scale'].append(scale)
                            downtime_stat['Loc'].append(loc)
                            downtime_stat['times'].append(len(Place_downtime))
                            downtime_stat['laplace'].append(False)

        # test for downtime interval
        for i in Place_df.index:
            days, seconds = Place_df.loc[i].days, Place_df.loc[i].seconds
            hours = days * 24 + seconds / 3600
            day = hours / 24
            Place_df.loc[i] = day

        Place_df = Place_df.astype(float)
        Place_df.sort_values(ascending=True)

        # data cleaning
        # data for low frequency occurrence
        Place_df_high_frequency = Place_df[Place_df < 240 / 1440] * 24 * 60
        # data for low freq outages
        Place_df_low_frequency = Place_df[Place_df > 240 / 1440]
        outlier = Place_df_low_frequency.mean() + 2 * Place_df_low_frequency.std()
        Place_df_low_frequency = Place_df_low_frequency[Place_df_low_frequency < outlier] * 24 * 60

        if len(Place_df_low_frequency) > 5:  # 調整樣本數量
            # KS test
            loc, scale = expon.fit(Place_df_low_frequency)
            e = expon(loc=loc, scale=scale)
            p = poisson(loc=loc, mu=max(Place_df_high_frequency) / scale)
            KS_result = kstest(Place_df_low_frequency, e.cdf)
            print(KS_result.pvalue)
            if KS_result.pvalue >0.05:
                service_time_stat_low['place'].append(place)
                service_time_stat_low['scale'].append(scale)
                service_time_stat_low['Loc'].append(loc)
                service_time_stat_low['times'].append(len(Place_df_low_frequency))
                print(f'{bcolors.OK}{SPN} {place} service time low:{KS_result}{bcolors.RESET}')
            else:
                print(f'{bcolors.FAIL}{SPN} {place} not pass:{KS_result}{bcolors.RESET}')

            if PLOT is True:
                fig, ax = plt.subplots(1, 1)
                Place_df_low_frequency.hist(bins=200,
                                            density=True,
                                            cumulative=True,
                                            histtype=u'step',
                                            label='actual service time')

                # this plot is for expon distribution
                distribution_plot(expon, Scale=scale, Loc=loc)
                ax.set_title(f'{SPN} {place} low frequency outages time CDF (follow expon distribution)')
                fig.savefig(f'/Users/hsu/PycharmProjects/OperationSystem/CyberRisk_Fig/{SPN} {place} low '
                            f'frequency outages time CDF (follow expon distribution).png')

            # this plot is for poisson distribution
            mu = 240 / scale
            mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
            x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))

            if PLOT is True:
                fig2, ax2 = plt.subplots(1, 1)
                ax2.plot(x,
                         poisson.pmf(x, mu),
                         'bo',
                         ms=8,
                         label='poisson pmf')

                ax2.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
                rv = poisson(mu)
                ax2.vlines(x,
                           0,
                           rv.pmf(x),
                           colors='k',
                           linestyles='-',
                           lw=1,
                           label='frozen pmf')

                ax2.set_xlabel(f'Occurrence')
                ax2.set_ylabel(f'probability')
                ax2.set_title(f'{SPN} {place} occurrence of low frequency outage probability')
                ax2.grid()
                ax2.legend(loc='best')
                plt.show()
                fig2.savefig(
                    f'/Users/hsu/PycharmProjects/OperationSystem/CyberRisk_Fig/{SPN} {place} occurrence of low '
                    f'frequency outages probability.png')

        # here is for high frequency occurrence
        if len(Place_df_high_frequency) > 5:  # 調整樣本數量
            # KS test
            loc, scale = expon.fit(Place_df_high_frequency)
            e = expon(loc=loc, scale=scale)
            KS_result = kstest(Place_df_high_frequency, e.cdf)
            print(KS_result.pvalue)
            if KS_result.pvalue > 0.05:
                service_time_stat_high['place'].append(place)
                service_time_stat_high['scale'].append(scale)
                service_time_stat_high['Loc'].append(loc)
                service_time_stat_high['times'].append(len(Place_df_high_frequency))
                print(f'{bcolors.OK}{SPN} {place} service time high:{KS_result}{bcolors.RESET}')
            else:
                print(f'{bcolors.FAIL}{SPN} {place}not pass service time high:{KS_result}{bcolors.RESET}')

            if PLOT is True:
                fig, ax = plt.subplots(1, 1)
                Place_df_high_frequency.hist(bins=200,
                                             density=True,
                                             cumulative=True,
                                             histtype=u'step',
                                             label='actual service time')

                # this plot is for poisson distribution
                distribution_plot(expon, Scale=scale, Loc=loc)
                ax.set_title(f'{SPN} {place} high frequency outages time CDF (follow exponential distribution)')
                fig.savefig(f'/Users/hsu/PycharmProjects/OperationSystem/CyberRisk_Fig/{SPN} {place} high '
                            f'frequency outages time CDF (follow expon distribution).png')

            mu = 40 / scale
            mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
            x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))

            if PLOT is True:
                fig2, ax2 = plt.subplots(1, 1)
                ax2.plot(x,
                         poisson.pmf(x, mu),
                         'bo',
                         ms=8,
                         label='poisson pmf')

                ax2.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
                rv = poisson(mu)
                ax2.vlines(x,
                           0,
                           rv.pmf(x),
                           colors='k',
                           linestyles='-',
                           lw=1,
                           label='frozen pmf')

                ax2.set_xlabel(f'Occurrence')
                ax2.set_ylabel(f'probability')
                ax2.grid()
                ax2.legend(loc='best')
                ax2.set_title(f'{SPN} {place} occurrence of high frequency outage probability')
                plt.show()
                fig2.savefig(
                    f'/Users/hsu/PycharmProjects/OperationSystem/CyberRisk_Fig/{SPN} {place} occurrence of high '
                    f'frequency outages probability.png')

    return pd.DataFrame(service_time_stat_low), pd.DataFrame(service_time_stat_high), pd.DataFrame(downtime_stat), \
           pd.DataFrame(downtime_stat_short)


class simulation:
    """
            in this part, we need to create a process follow poisson process.
            The poisson process try to describe the outage time.  The outage time follow exponential distribution
            and the outage frequent follow poisson distribution.
    """

    def __init__(self, service_high, service_low, downtime, downtime_short):
        self.H = service_high
        self.L = service_low
        self.d = downtime
        self.d_s = downtime_short
        pass

    def Data_enough(self, place: str, service=True) -> bool:
        if service:
            if not (self.H[self.H.place == place].empty or self.L[self.L.place == place].empty):
                return True
            return False
        else:
            if not (self.d[self.d.place == place].empty or self.d_s[self.d_s.place == place].empty):
                return True
            return False

    def prob(self, place: str, service=True) -> float:
        if service:
            all_freq = self.H[self.H.place == place].times.iloc[0] + \
                       self.L[self.L.place == place].times.iloc[0]

            p = self.H[self.H.place == place].times.iloc[0] / all_freq
        else:
            all_freq = self.d_s[self.d_s.place == place].times.iloc[0] + \
                       self.d[self.d.place == place].times.iloc[0]

            p = self.d_s[self.d_s.place == place].times.iloc[0] / all_freq
        return p

    @staticmethod
    def filter(Service_data: pd.DataFrame, pLace) -> pd.Series:
        F = Service_data.place == pLace
        return F

    @staticmethod
    def process_simulate(st, dt, name):
        fig, ax = plt.subplots(1, 1)
        initial = 0
        for i, j in zip(st, dt):
            ax.plot((initial, initial + i), (0, 0), c='b')
            initial += (i + j)
            ax.plot((initial - j, initial), (j, j), c='r')
        ax.plot((initial, initial + i), (0, 0), c='b', label='service time')
        ax.plot((initial - j, initial), (j, j), c='r', label='downtime')
        ax.set_title(f'{name} downtime simulation')
        ax.set_ylabel('downtime')
        ax.set_xlabel('time series')
        ax.legend(loc='best')
        fig.savefig(f'/Users/hsu/PycharmProjects/OperationSystem/CyberRisk_Fig/CTL {PLA} '
                    f'downtime poisson process simulation.png')

    @staticmethod
    def bernoulli(p0: float) -> bool:
        """
        param p0: float
        :return: bool
        """
        if p0 < np.random.rand():
            return True
        return False


if __name__ == '__main__':
    SD = main(PLOT=False, SP=2)

    sim = simulation(*SD)

    # get longer dataframe as base to search
    long_df = SD[0] if len(SD[0]) > len(SD[1]) else SD[1]
    stat_dict = {'laplace': np.random.laplace}
    process = {}

    for Place in long_df.place:

        ST = []
        Downtime = []
        num = 1000

        # progress = tqdm(range(num), desc=f'{Place}')
        for i in tqdm(range(num), desc=f'{Place}'):
            # service time simulate
            if sim.Data_enough(Place):
                '''for those data enough'''
                # if the data is enough in here, we want to generate the Time and downtime simulation.

                P0 = sim.prob(Place)

                if sim.bernoulli(P0):
                    temp_x = expon.ppf(np.random.rand(), scale=SD[1][sim.filter(SD[1], Place)].scale.iloc[0],
                                       loc=SD[1][sim.filter(SD[1], Place)].Loc.iloc[0])

                else:
                    temp_x = expon.ppf(np.random.rand(), scale=SD[0][sim.filter(SD[0], Place)].scale.iloc[0],
                                       loc=SD[0][sim.filter(SD[0], Place)].Loc.iloc[0])

                ST.append(temp_x)

            else:
                '''for those data not enough'''
                # currently, I have no idea to evaluate the params
                pass

            # outage time simulate
            if sim.Data_enough(Place, service=True):

                p1 = sim.prob(Place, service=True)
                if sim.bernoulli(p1):

                    if SD[3][SD[3].place == Place].laplace.iloc[0]:
                        temp_x = laplace.ppf(np.random.rand(), scale=SD[3][sim.filter(SD[3], Place)].scale.iloc[0],
                                             loc=SD[3][sim.filter(SD[3], Place)].Loc.iloc[0])

                        down = 0 if temp_x < 0 else temp_x

                    else:
                        down = expon.ppf(np.random.rand(), scale=SD[3][sim.filter(SD[3], Place)].scale.iloc[0],
                                         loc=SD[3][sim.filter(SD[3], Place)].Loc.iloc[0])

                else:
                    try:
                        if SD[2][SD[2].place == Place].laplace.iloc[0]:
                            temp_x = laplace.ppf(np.random.rand(), scale=SD[2][sim.filter(SD[2], Place)].scale.iloc[0],
                                                 loc=SD[2][sim.filter(SD[2], Place)].Loc.iloc[0])

                            down = 0 if temp_x < 0 else temp_x

                        else:
                            down = expon.ppf(np.random.rand(), scale=SD[2][sim.filter(SD[2], Place)].scale.iloc[0],
                                             loc=SD[2][sim.filter(SD[2], Place)].Loc.iloc[0])
                    except Exception as e:
                        e
                Downtime.append(down)
            # progress.update(1)

        process[f'{Place} downtime'] = Downtime
        process[f'{Place} service'] = ST

    for Plac in list(process):
        if not process[f'{Plac}']:
            del process[f'{Plac}']

    process_df = pd.DataFrame(process)
    for PLA in tqdm(long_df.place):
        sim.process_simulate(process_df[f'{PLA} service'], process_df[f'{PLA} downtime'], PLA)

    process_df.hist(bins=200,
                    density=True,
                    cumulative=True,
                    histtype=u'step')

    downtime_df = pd.DataFrame()
    for i in range(len(process_df.columns)):
        if i % 2 == 0:
            downtime_df[f'{process_df.columns[i]}'] = process_df[process_df.columns[i]] + process_df[
                process_df.columns[i + 1]]

    for i in downtime_df.columns:
        for j in range(1000):
            if downtime_df[i].iloc[:j + 1].sum() >= 216000:
                print(process_df[i].iloc[0:j + 1].sum())
                break
