import time
import datetime
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

driver = webdriver.Safari()
driver.get("https://cloudharmony.com/status-of-compute")
driver.maximize_window()
Header = driver.find_elements_by_class_name('header')

actions = ActionChains(driver)

for outage in Header:
    if outage.text == 'Outages':
        driver.execute_script('arguments[0].click();', outage)

outage_time_elements = driver.find_elements(By.XPATH, '//a[@style="text-decoration:underline"]')

outages_df = pd.DataFrame()
for outage_time in outage_time_elements[1:]:
    driver.execute_script('arguments[0].click();', outage_time)

    try:
        WebDriverWait(driver, 5, 0.5).until(

            expected_conditions.presence_of_element_located((By.XPATH, "//tr[@style='display: table-row;']"))

        )
        time.sleep(1.5)
        outages_element = driver.find_elements(By.XPATH, "//tr[@style='display: table-row;']")
        outages = [i.text.split() for i in outages_element]

        print(outages)

    finally:
        pass

    for i in range(len(outages)):
        temp = []
        for j in range(1, 6):
            if j == 5:
                temp.append(f'{outages[i][j]}')
            else:
                temp.append(f'{outages[i][j]} ')
        content = "".join(temp)
        outages[i][j] = content

        temp = []
        for j in range(6, 11):
            if j == 10:
                temp.append(f'{outages[i][j]}')
            else:
                temp.append(f'{outages[i][j]} ')

        content = "".join(temp)
        outages[i][j] = content

    outages_df = outages_df.append(outages)
    print(outages_df)

    close_element = driver.find_elements(By.XPATH, '//div[@class="modal_close icon-round-close"]')

outages_df = outages_df.reset_index().drop(['index'], axis=1)
outages_df[15] = outages_df[outages_df[12] == 'secs'][11].astype(float) / 60
outages_df[16] = outages_df[outages_df[12] == 'hours'][11].astype(float) * 60
outages_df = outages_df.replace(np.nan, 0)
for i in range(len(outages_df[15])):
    if outages_df[15][i] == 0 and outages_df[12][i] == 'mins':
        outages_df[15][i] = outages_df[11][i]
    elif outages_df[15][i] == 0 and outages_df[12][i] == 'hours':
        outages_df[15][i] = outages_df[16][i]

outages_df = outages_df.drop([1, 2, 3, 4, 6, 7, 8, 9, 16], axis=1).set_axis(

    ['place', 'start time', 'end time', 'original downtime', 'unit',

     'Ripe Atlas Confirmation Ratio', 'notes', 'downtime'], axis=1, inplace=False)


information_stage = {}
for i in range(len(outage_time_elements)):
    information_stage.update({

        driver.find_element_by_xpath(f'//tr[{i + 1}]/td[2]').text:
        [
            int(driver.find_element_by_xpath(f'//tr[{i + 1}]/td[5]/a').text),
            driver.find_element_by_xpath(f'//tr[{i + 1}]/td[1]/a').text
        ]
    })


# information_stage2 = {}
# for i in range(len(outage_time_elements)):
#     if information_stage2.get(driver.find_element_by_xpath(f'//tr[{i + 1}]/td[1]/a').text) is None:
#         information_stage2.update({
#             driver.find_element_by_xpath(f'//tr[{i + 1}]/td[1]/a').text:
#                 [driver.find_element_by_xpath(f'//tr[{i + 1}]/td[2]').text]
#         })
#     else:
#         information_stage2[driver.find_element_by_xpath(f'//tr[{i + 1}]/td[1]/a').text].append(
#             driver.find_element_by_xpath(f'//tr[{i + 1}]/td[2]').text
#         )

# Service_dict = {'Service_Name': []}
# for place_name in outages_df['place']:
#     try:
#         Service_dict['Service_Name'].append(information_stage.get(place_name)[1])
#     except TypeError:
#         print('\n Warning! missing value in the cloud outage data, check the dataframe')


outages_df['Service Name'] = None
num = 0
for i in outages_df['place']:
    try:
        outages_df['Service Name'][num] = information_stage.get(i)[1]
    except TypeError:
        pass
        print('Warning! NoneType object is not subscribtable')
    num += 1

Service_Name = outages_df.groupby('Service Name')
service_size = Service_Name.size()
service_downtime = [np.array(Service_Name.get_group(SEV)['downtime']) for SEV in service_size.index]


n_bins = 20
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.hist(service_downtime, n_bins, histtype='bar', label=service_size.index)
ax.set_title(f'compute service downtime', fontsize='x-large', fontfamily='sans-serif', fontstyle='italic')
ax.legend(prop={'size': 10})
plt.semilogy(basey=10)
ax.set_xlabel('Downtime $_{min}$', fontsize='x-large', fontfamily='sans-serif', fontstyle='italic')
ax.set_ylabel('times', fontsize='x-large', fontfamily='sans-serif', fontstyle='italic')
plt.show()
outages_df.to_csv(f"/Users/hsu/PycharmProjects/OperationSystem/cloud_outages_dataset_{datetime.date.today()}.csv",
                  encoding='utf-8', index=False)
