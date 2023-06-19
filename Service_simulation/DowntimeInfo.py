import glob, os
import pandas as pd
import pytz
import dateutil.parser
import warnings

# 停止顯示指定類型的警告訊息
warnings.filterwarnings("ignore", category=Warning)



class bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


class Downtime:

    def __init__(self, file_path: str = None, file_name: str = None, file_type: str = None):
        file_path = '//'
        os.chdir(file_path)
        files = [file for file in glob.glob('*.csv')]
        files.reverse()

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

        self.new_COD = pd.DataFrame()
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
                self.new_COD = pd.concat([self.new_COD, P], axis=0)

    def data_info(self,):
        return self.new_COD