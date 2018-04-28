from modules import *
from sys import argv


if __name__ == '__main__':

    script, year, folder, regionfile, wrsfile = argv

    # find the wrs2 path and row that intersect with region boundary
    path_row, _ = find_intersecting_tiles(regionfile, wrsfile)

    # prepare list of tuples based on the path, row, and year
    tile_list = list()
    for pr in path_row:
        tile_list.append((pr[0], pr[1], year))

    # get server name
    server = TCserver
    print(server)

    # create ftp handle and connect
    ftp = FTPHandler(ftpserv=server,
                     dirname=folder)
    ftp.connect()

    # get a list of tile links on the ftp
    ftp.ftpfilepath = list()
    for tile_param in tile_list:
        tile_link = get_TCdata_filepath(*tile_param)['filestr']
        ftp.ftpfilepath.append(tile_link)

    # download all the ftp tiles in list
    ftp.getfiles()

    # disconnect the ftp
    ftp.disconnect()

    print('Done!')
