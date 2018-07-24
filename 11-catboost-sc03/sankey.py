import collections
import colorsys

import matplotlib.pyplot
import matplotlib.path
import matplotlib.patches


def get_distinct_colors(n):
    """
    https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python/answer/Karthik-Kumar-Viswanathan
    """
    hue_partition = 1 / (n + 1)
    colors = [colorsys.hsv_to_rgb(hue_partition * value, 1.0, 1.0)
              for value in range(0, n)]
    return colors[::2] + colors[1::2]


def text_width(fig, ax, text, fontsize):
    text = ax.text(-100, 0, text, fontsize=fontsize)
    text_bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    text_bb = text_bb.transformed(fig.dpi_scale_trans.inverted())
    return text_bb.width


class Sankey:
    def __init__(self, x, y,
                 plot_width=8,
                 plot_height=8,
                 gap=0.1,
                 alpha=0.3,
                 fontsize='small'):
        self.X = x
        self.Y = y
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.gap = gap
        self.alpha = alpha
        self.fontsize = fontsize

        self.init_figure()

        self.flows = collections.Counter(zip(x, y))
        self.init_nodes()

        self.init_widths()
        # inches per 1 item in x and y
        self.resolution = (plot_height - gap * (len(self.left_nodes) - 1)) / len(x)
        self.colors = {
            name: colour
            for name, colour
            in zip(self.left_nodes.keys(),
                   get_distinct_colors(len(self.left_nodes)))
        }

        self.init_offsets()

    def init_figure(self):
        self.fig = matplotlib.pyplot.figure()
        self.ax = matplotlib.pyplot.Axes(self.fig, [0, 0, 1, 1])
        self.fig.add_axes(self.ax)

    def init_nodes(self):
        left_nodes = {}
        right_nodes = {}
        left_offset = 0
        for (left, right), flow in sorted(self.flows.items()):
            if left in left_nodes:
                left_nodes[left] += flow
            else:
                left_nodes[left] = flow
            if right in right_nodes:
                node = right_nodes[right]
                node[0] += flow
                if flow > node[2]:
                    node[1] = left
                    node[2] = flow
            else:
                right_nodes[right] = [flow, left, flow]

        self.left_nodes = collections.OrderedDict()
        for name, flow in sorted(
            left_nodes.items(),
            key=lambda pair: -pair[1]
        ):
            self.left_nodes[name] = flow

        left_names = list(self.left_nodes.keys())
        self.right_nodes = collections.OrderedDict()
        for name, node in sorted(
            right_nodes.items(),
            key=lambda pair: (left_names.index(pair[1][1]), -pair[1][2])
        ):
            self.right_nodes[name] = node[0]

    def init_widths(self):
        self.left_width = max(
            (text_width(self.fig,
                        self.ax,
                        node,
                        self.fontsize) for node in self.left_nodes)
        )
        self.right_width = max(
            (text_width(self.fig,
                        self.ax,
                        node,
                        self.fontsize) for node in self.right_nodes)
        )

        self.right_stop = self.plot_width - self.left_width - self.right_width
        self.middle1_stop = self.right_stop * 9 / 20
        self.middle2_stop = self.right_stop * 11 / 20

    def init_offsets(self):
        self.offsets_l = {}
        self.offsets_r = {}

        offset = 0
        for name, flow in self.left_nodes.items():
            self.offsets_l[name] = offset
            offset += flow * self.resolution + self.gap

        offset = 0
        for name, flow in self.right_nodes.items():
            self.offsets_r[name] = offset
            offset += flow * self.resolution + self.gap

    def draw_flow(self, left, right, flow, node_offsets_l, node_offsets_r):
        P = matplotlib.path.Path

        flow *= self.resolution
        left_y = self.offsets_l[left] + node_offsets_l[left]
        right_y = self.offsets_r[right] + node_offsets_r[right]
        node_offsets_l[left] += flow
        node_offsets_r[right] += flow

        path_data = [
            (P.MOVETO, (0, -left_y)),
            (P.LINETO, (0, -left_y - flow)),
            (P.CURVE4, (self.middle1_stop, -left_y - flow)),
            (P.CURVE4, (self.middle2_stop, -right_y - flow)),
            (P.CURVE4, (self.right_stop, -right_y - flow)),
            (P.LINETO, (self.right_stop, -right_y)),
            (P.CURVE4, (self.middle2_stop, -right_y)),
            (P.CURVE4, (self.middle1_stop, -left_y)),
            (P.CURVE4, (0, -left_y)),
            (P.CLOSEPOLY, (0, -left_y)),
        ]
        codes, verts = zip(*path_data)
        path = P(verts, codes)
        patch = matplotlib.patches.PathPatch(
            path,
            facecolor=self.colors[left],
            alpha=1 if flow < .02 else self.alpha,
            edgecolor='none',
        )
        self.ax.add_patch(patch)

    def draw_label(self, label, is_left):
        nodes = self.left_nodes if is_left else self.right_nodes
        offsets = self.offsets_l if is_left else self.offsets_r
        y = offsets[label] + nodes[label] * self.resolution / 2

        self.ax.text(
            -.1 if is_left else self.right_stop + .1,
            -y,
            label,
            horizontalalignment='right' if is_left else 'left',
            verticalalignment='center',
            fontsize=self.fontsize,
        )

    def draw(self):
        node_offsets_l = collections.Counter()
        node_offsets_r = collections.Counter()

        for (left, right), flow in sorted(
            self.flows.items(),
            key=lambda pair: -pair[1]
        ):
            self.draw_flow(left, right, flow, node_offsets_l, node_offsets_r)

        for name in self.left_nodes:
            self.draw_label(name, True)
        for name in self.right_nodes:
            self.draw_label(name, False)

        self.ax.axis('equal')
        self.ax.set_xlim(-self.left_width - self.gap,
                         self.right_stop + self.gap + self.right_width)
        matplotlib.pyplot.axis('off')
        self.fig.set_figheight(self.plot_height)
        self.fig.set_figwidth(self.plot_width)
        #matplotlib.pyplot.tight_layout()


def sankey(x, y, **kwargs):
    diag = Sankey(x, y, **kwargs)
    diag.draw()
    return diag.fig