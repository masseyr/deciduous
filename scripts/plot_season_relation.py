from modules.plots import Plot
from modules import Handler, Opt
from sys import argv


if __name__ == '__main__':

    script, in_dir = argv

    csv_files = Handler(dirname=in_dir).find_all('.csv')

    for csv_file in csv_files:

        Opt.cprint('Reading file : {}'.format(csv_file))

        val_dicts = Handler(csv_file).read_from_csv(return_dicts=True)

        bandname = Handler(csv_file).basename.split('.csv')[0]

        plot_file = in_dir + "{}_21.png".format(bandname)

        xvar = '{}_2'.format(bandname)
        xlabel = 'Season 2 {}'.format(bandname.upper())

        yvar = '{}_1'.format(bandname)
        ylabel = 'Season 1 {}'.format(bandname.upper())

        x = list(int(elem[xvar].strip()) for elem in val_dicts)
        y = list(int(elem[yvar].strip()) for elem in val_dicts)

        pts = [(x[i], y[i]) for i in range(0, len(x))]

        Opt.cprint(len(pts))

        plot_heatmap = {
            'type': 'rheatmap',
            'points': pts,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'color_bar_label': 'Data-points per bin',
            'plotfile': plot_file,
            'xlim': (0, 10000),
            'ylim': (0, 10000),
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

        x = list(int(elem[xvar].strip()) for elem in val_dicts)
        y = list(int(elem[yvar].strip()) for elem in val_dicts)

        pts = [(x[i], y[i]) for i in range(0, len(x))]

        print(len(pts))

        plot_heatmap = {
            'type': 'rheatmap',
            'points': pts,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'color_bar_label': 'Data-points per bin',
            'plotfile': plot_file,
            'xlim': (0, 10000),
            'ylim': (0, 10000),
            'line': True,
            'legend': True
        }

        heatmap = Plot(plot_heatmap)
        heatmap.draw()

        print('Plot file : {}'.format(plot_file))
