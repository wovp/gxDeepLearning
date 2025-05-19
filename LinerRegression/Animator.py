from IPython import display, get_ipython
from matplotlib import pyplot as plt

def _is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter Notebook 或 QtConsole
        elif shell == 'TerminalInteractiveShell':
            return False  # 终端 IPython
        else:
            return False  # 其他环境
    except NameError:
        return False  # 标准 Python


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        def config_axes(ax_idx=0):
            ax = self.axes[ax_idx]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            if legend:
                ax.legend(legend)

        self.config_axes = config_axes
        self.X, self.Y, self.fmts = None, None, fmts
        self.is_jupyter = _is_jupyter()

    def add(self, x, y, ax_idx=0):
        # 处理输入数据
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n

        # 初始化数据存储
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        # 追加新数据点
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        # 检查格式数量是否足够
        if len(self.X) > len(self.fmts):
            raise ValueError(f"数据系列数({len(self.X)})超过格式数({len(self.fmts)})")

        # 更新图表
        ax = self.axes[ax_idx]
        if not hasattr(self, 'lines'):
            self.lines = []
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                line, = ax.plot(x, y, fmt)
                self.lines.append(line)
        else:
            for line, x, y in zip(self.lines, self.X, self.Y):
                line.set_data(x, y)

        ax.relim()
        ax.autoscale_view()
        self.config_axes(ax_idx)

        # 显示更新
        if self.is_jupyter:
            display.display(self.fig)
            display.clear_output(wait=True)
        else:
            plt.draw()
            plt.pause(0.001)
