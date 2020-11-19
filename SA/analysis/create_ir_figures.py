"""Plots for RAWR paper"""
import pickle
import argparse 
import pandas as pd
import matplotlib
matplotlib.use('agg')
from plotnine import (
    ggplot, aes, facet_wrap,
    geom_histogram, geom_text, geom_segment, geom_density,
    scale_color_manual, scale_fill_manual, scale_linetype_manual,
    element_blank, element_text, element_rect, element_line,
    theme_light,
    xlim, ylim, xlab, ylab, theme
)


class theme_fs(theme_light):
    """
    A theme similar to :class:`theme_linedraw` but with light grey
    lines and axes to direct more attention towards the data.
    Parameters
    ----------
    base_size : int, optional
        Base font size. All text sizes are a scaled versions of
        the base font size. Default is 11.
    base_family : str, optional
        Base font family.
    """

    def __init__(self, base_size=11, base_family='DejaVu Sans'):
        theme_light.__init__(self, base_size, base_family)
        self.add_theme(theme(
            axis_ticks=element_line(color='#DDDDDD', size=0.5),
            panel_border=element_rect(fill='None', color='#838383',
                                      size=1),
            strip_background=element_rect(
                fill='#DDDDDD', color='#838383', size=1),
            strip_text_x=element_text(color='black'),
            strip_text_y=element_text(color='black', angle=-90)
        ), inplace=True)


COLORS = [
    '#49afcd', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
FIGURES_STATS_FILE = "data/figure_stats.pkl"


def figfile(s):
    return 'auto_fig/' + str(s)


def create_df(nums, task, method):
    df = pd.DataFrame({'x': nums})
    df['Task'] = task
    df['Method'] = method
    return df


def load_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    print("data", data)

    len_rows = [
        create_df(data[0], "SA Baseline", "Original"),
        create_df(data[1], "SA Baseline", "Reduced"),
        create_df(data[2], "SA Combined", "Original"),
        create_df(data[3], "SA Combined", "Reduced")
    ]

    # count = 0
    # for val in data[1]:
    #     if val == 1:
    #         count += 1
    # print("PERCENT", count/872)
    # exit(0)

    # len_rows = [
    #     create_df(data[0], 'SNLI', 'Original'),
    #     create_df(data[1], 'SNLI', 'Reduced'),
    #     create_df(data[4], 'SQuAD', 'Original'),
    #     create_df(data[5], 'SQuAD', 'Reduced'),
    #     create_df(data[10], 'VQA', 'Original'),
    #     create_df(data[11], 'VQA', 'Reduced')
    # ]

    # conf_rows = [
    #     create_df(data[2], 'SNLI', 'Original'),
    #     create_df(data[3], 'SNLI', 'Reduced'),
    #     create_df(data[6], 'SQuAD Start', 'Original'),
    #     create_df(data[7], 'SQuAD End', 'Original'),
    #     create_df(data[8], 'SQuAD Start', 'Reduced'),
    #     create_df(data[9], 'SQuAD End', 'Reduced'),
    #     create_df(data[12], 'VQA', 'Original'),
    #     create_df(data[13], 'VQA', 'Reduced')
    # ]

    len_df = pd.concat(len_rows)
    # len_df['Task'] = len_df['Task'].astype('category').cat.reorder_categories([
    #     'SQuAD', 'SNLI', 'VQA'
    # ])

    len_df['Task'] = len_df['Task'].astype('category')
    len_df['Method'] = len_df['Method'].astype('category')

    # conf_df = pd.concat(conf_rows)
    # conf_df['Task'] = conf_df['Task'].astype('category').cat.reorder_categories([
    #     'SQuAD Start', 'SQuAD End', 'SNLI', 'VQA'
    # ])
    # conf_df['Method'] = conf_df['Method'].astype('category')

    # return len_df, conf_df

    return len_df, None 


def create_length_plot(len_df, legend_position='right', legend_box='vertical'):
    mean_len_df = len_df.groupby(['Task', 'Method']).mean().reset_index()

    mean_len_df[' '] = 'Mean Length'
    print(len_df)

    plt = (
        ggplot(len_df)
        + aes(x='x', fill='Method', y='..density..')
        + geom_histogram(binwidth=2, position='identity', alpha=0.6)
        + geom_text(
            aes(x='x', y=.22, label='x', color='Method'),
            mean_len_df,
            inherit_aes=False,
            format_string='{:.1f}',
            show_legend=False
        )
        + geom_segment(
            aes(x='x', xend='x', y=0, yend=.205, linetype=' '),
            mean_len_df,
            inherit_aes=False, color='black'
        )
        + scale_linetype_manual(['dashed'])
        + facet_wrap('Task')
        # + xlim(0, 40) 
        + ylim(0, 0.35)
        + xlab('Example Length') + ylab('Frequency')
        + scale_color_manual(values=COLORS)
        + scale_fill_manual(values=COLORS)
        + theme_fs()
        + theme(
            aspect_ratio=1,
            legend_title=element_blank(),
            legend_position=legend_position,
            legend_box=legend_box,
        )
    )

    return plt


def create_confidence_plot(conf_df):
    plt = (
        ggplot(conf_df)
        + aes(x='x', color='Method', fill='Method')
        + geom_density(alpha=.45)
        + facet_wrap('Task', nrow=4)
        + xlab('Confidence')
        + scale_color_manual(values=COLORS)
        + scale_fill_manual(values=COLORS)
        + theme_fs()
        + theme(
            axis_text_y=element_blank(),
            axis_ticks_major_y=element_blank(),
            axis_title_y=element_blank(),
            legend_title=element_blank(),
            legend_position='top',
            legend_box='horizontal',
        )
    )
    return plt


def main():
    args = argument_parsing()

    print('Loading data from: ' + 'data/input_reduction_stats_{}.pkl'.format(args.id))
    len_df, conf_df = load_data('data/input_reduction_stats_{}.pkl'.format(args.id))
    print(len_df)

    print('Generating length histogram plot...')
    len_plt = create_length_plot(len_df)

    len_output_file = figfile('input_reduction_length_histogram_{}.pdf'.format(args.id))
    print('Saving to: ' + str({len_output_file}))
    len_plt.save(len_output_file)


def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--id', type=int, help='ID number')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()