import paddle
from matplotlib import pyplot as plt
from paddle.vision import transforms


def picCatTrainSet_show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)  # 修改这里
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if paddle.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = paddle.vision.datasets.FashionMNIST(mode="train",
                                                      transform=trans)
    mnist_test = paddle.vision.datasets.FashionMNIST(mode="test",
                                                     transform=trans)
    return (paddle.io.DataLoader(dataset=mnist_train,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 return_list=True,
                                 num_workers=get_dataloader_workers()),
            paddle.io.DataLoader(dataset=mnist_test,
                                 batch_size=batch_size,
                                 return_list=True,
                                 shuffle=True,
                                 num_workers=get_dataloader_workers()))
