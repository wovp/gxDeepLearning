def load_array(data_arrays, batch_size, is_train=True):
    """构造一个Paddle数据迭代器"""
    dataset = paddle.io.TensorDataset(data_arrays)
    return paddle.io.DataLoader(dataset, batch_size=batch_size,
                                shuffle=is_train,
                                return_list=True)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))
