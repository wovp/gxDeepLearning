import paddle


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with paddle.no_grad():
        for i, param in enumerate(params):
            param -= lr * params[i].grad / batch_size
            params[i].set_value(param)
            params[i].clear_gradient()