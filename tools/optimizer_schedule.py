
def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False,
                       warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter / warmup_step))
    else:
        # 最开始的学习率大，慢慢变小
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr  # 10x LR
