from modules import *


if __name__ == '__main__':

    file1 = 'C:/temp/decid/gee_mack_data_extract_250_v2019_06_14T03_55_15_250_0.csv'

    list_dicts = Handler(file1).read_from_csv(return_dicts=True)

    print('Number of dictionaries: {}'.format(str(
        len(list_dicts))))

    count = dict()

    for dict_ in list_dicts:
        fire_count = 0

        for k, v in dict_.items():

            if 'burn_year' in k:

                if 0 < int(v) < 2000:

                    fire_count += 1

        if fire_count in count:
            count[fire_count] += 1
        else:
            count[fire_count] = 1

    for k, v in count.items():
        print('{} burns : {} pixels'.format(str(k), str(v)))








