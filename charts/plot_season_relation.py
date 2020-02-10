from modules.plots import Plot
from modules import Handler, Opt, Sublist
from sys import argv


if __name__ == '__main__':

    # script, in_dir = argv

    in_dir = 'c:/temp/check2/'
    Handler(dirname=in_dir).dir_create()

    csv_files = Handler(dirname=in_dir).find_all('.csv')

    for csv_file in csv_files:

        Opt.cprint('Reading file : {}'.format(csv_file))

        val_dicts = Handler(csv_file).read_from_csv(return_dicts=True,
                                                    read_random=True,
                                                    line_limit=None,
                                                    percent_random=10.0)

        bandname = Handler(csv_file).basename.split('.csv')[0]

        plot_file = in_dir + "{}_21.png".format(bandname)

        xvar = '{}_2'.format(bandname)
        xlabel = 'Season 2 {}'.format(bandname.upper())

        yvar = '{}_1'.format(bandname)
        ylabel = 'Season 1 {}'.format(bandname.upper())

        pts = list()

        for i in range(0, len(val_dicts)):
            try:
                pts.append((float(val_dicts[i][xvar]), float(val_dicts[i][yvar])))
            except ValueError:
                pass

        maxx = Sublist.percentile(list(elem[0] for elem in pts), 99.995)
        maxy = Sublist.percentile(list(elem[1] for elem in pts), 99.995)

        max_elem = max(maxx, maxy)

        Opt.cprint('X and Y limits: {:3.3f} {:3.3f}'.format(maxx, maxy))

        plot_heatmap = {
            'type': 'rheatmap2',
            'points': pts,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'color_bar_label': 'Data-points per bin',
            'plotfile': plot_file,
            'xlim': (0, max_elem),
            'ylim': (0, max_elem),
            'line': True,
            'legend': True
        }

        heatmap = Plot(plot_heatmap)
        heatmap.draw()

        Opt.cprint('Plot file : {}'.format(plot_file))

        plot_file = in_dir + "{}_23.png".format(bandname)

        xvar = '{}_2'.format(bandname)
        xlabel = 'Season 2 {}'.format(bandname.upper())

        yvar = '{}_3'.format(bandname)
        ylabel = 'Season 3 {}'.format(bandname.upper())

        pts = list()

        for i in range(0, len(val_dicts)):
            try:
                pts.append((float(val_dicts[i][xvar]), float(val_dicts[i][yvar])))
            except ValueError:
                pass

        maxx = Sublist.percentile(list(elem[0] for elem in pts), 99.995)
        maxy = Sublist.percentile(list(elem[1] for elem in pts), 99.995)

        max_elem = max(maxx, maxy)

        Opt.cprint('X and Y limits: {:3.3f} {:3.3f}'.format(maxx, maxy))

        plot_heatmap = {
            'type': 'rheatmap2',
            'points': pts,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'color_bar_label': 'Data-points per bin',
            'plotfile': plot_file,
            'xlim': (0, max_elem),
            'ylim': (0, max_elem),
            'line': True,
            'legend': True
        }
        heatmap = Plot(plot_heatmap)
        heatmap.draw()

        print('Plot file : {}'.format(plot_file))
