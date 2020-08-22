import torch

def onlyLasso(y,p):

    print(p[y])

    LOSSES = []
    CLASSES = []

    """
    # Get the ones and zeros as ragged tensors
    bool_y = tf.cast(y, tf.bool)
    ones = tf.ragged.boolean_mask(p, bool_y)

    inverted_labels_bool = tf.math.logical_not(bool_y)
    zeros = tf.ragged.boolean_mask(p, inverted_labels_bool)

    # Stack corresponding ones and zeros on top of each other
    one_zero_pairs = tf.ragged.stack([ones,zeros], axis=1)

    print(f"one_zero_pairs: {one_zero_pairs}")
    print(f"one_zero_pairs.shape: {one_zero_pairs.shape}")
    print(f"one_zero_pairs[0]: {one_zero_pairs[0]}")
    print(f"one_zero_pairs[0].shape: {one_zero_pairs[0].shape}")

    indVar = 0
    LOSSES = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    CLASSES = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    CLASSES = []

    # Fix this if you can!
    for one_zero_pair in one_zero_pairs:
        # Calculate missing class errors
        onez = tf.expand_dims(one_zero_pair[0], axis=0)
        zeroz = tf.expand_dims(one_zero_pair[1], axis=1)
        k = tf.math.subtract(zeroz, onez)
        k = tf.math.multiply(k, 2)
        k = tf.math.add(k, 1)
        k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
        errors_miss = tf.math.square(k)

        # Calculate extra class errors
        onez = tf.expand_dims(one_zero_pair[0], axis=1)
        zeroz = tf.expand_dims(one_zero_pair[1], axis=0)
        k = tf.math.subtract(zeroz, onez)
        k = tf.math.multiply(k, 2)
        k = tf.math.add(k, 1)
        k = tf.math.maximum(tf.constant(0, dtype=tf.float32), k)
        errors_extra = tf.math.square(k)

        # Calculate the loss_miss
        k = tf.reduce_sum(errors_miss, axis=1)
        groups_miss = tf.math.sqrt(k)
        loss_miss = tf.reduce_sum(groups_miss)

        # Calculate the extra_miss
        k = tf.reduce_sum(errors_extra, axis=1)
        groups_extra = tf.math.sqrt(k)
        loss_extra = tf.reduce_sum(groups_extra)

        # Enter the loss for the sample
        LOSSES = LOSSES.write(indVar, (loss_miss + loss_extra) / 2.0)

        indVar += 1
    """

    return LOSSES, CLASSES


def test():
    y1 = torch.tensor([[1,0,1,0,1,1,0,0,1]]) # 9 CLASSES
    #noisy_y1 = torch.tensor([[1,0,0,0,1,0,0,0,1]]) # Missing classes at indexes 2 and 5
    #noisy_y1 = torch.tensor([[1,0,1,1,1,1,0,1,1]]) # Extra classes at indexes 3 and 7
    noisy_y1 = torch.tensor([[1,0,0,0,1,1,0,1,1]]) # Missing class at index 2 and extra class at index 7
    double_noisy_y1 = torch.tensor([[1,0,0,0,1,0,0,1,1],[1,0,0,0,1,1,0,1,1]]) # Missing class at index 2 and extra class at index 7
    p1 = torch.tensor([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]])
    p2 = torch.tensor([[0.78,0.34,0.72,0.2,0.86,0.62,0.4,0.17,0.95]])
    p3 = torch.tensor([[0.53,0.47,0.58,0.39,0.51,0.54,0.49,0.46,0.61]])
    p4 = torch.tensor([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.78,0.34,0.72,0.2,0.86,0.62,0.4,0.17,0.95]], dtype=np.float32)
    losses, classes = onlyFasterLasso(double_noisy_y1,p4)

    print(f"losses: {losses}")
    print(f"classes: {classes}")

    '''
    results = groupLasso(double_noisy_y1,p4)
    print(f"LOSS1: {results[0][0]}")
    print(f"change_class: {results[0][1]}")
    print(f"missing_class: {results[0][2]}")
    print(f"extra_class: {results[0][3]}")
    print(f"loss_miss: {results[0][4]}")
    print(f"groups_miss: {results[0][5]}")
    print(f"errors_miss: {results[0][6]}")
    print(f"loss_extra: {results[0][7]}")
    print(f"groups_extra: {results[0][8]}")
    print(f"errors_extra: {results[0][9]}")
    print(f"missing_classes: {results[0][10]}")
    print(f"extra_classes: {results[0][11]}")
    '''

def bucakTest():
    y1 = torch.tensor([1,-1,1,-1,1,1,-1,-1,1]) # 9 CLASSES
    noisy_y1 = torch.tensor([1,-1,-1,-1,1,-1,-1,-1,1]) # Missing classes at indexes 2 and 5
    #noisy_y1 = torch.tensor([1,-1,1,1,1,1,-1,1,1]) # Extra classes at indexes 3 and 7
    #noisy_y1 = torch.tensor([1,-1,-1,-1,1,1,-1,1,1]) # Missing class at index 2 and extra class at index 7
    p1 = torch.tensor([0.91,-0.88,0.87,-0.83,0.92,0.83,-0.93,-0.91,0.95])  # loss = 3.1990626168794223
    p2 = torch.tensor([0.78,-0.68,0.72,-0.8,0.86,0.62,-0.32,-0.56,0.95])   # loss = 2.804749814873396
    p3 = torch.tensor([0.23,-0.32,0.42,-0.52,0.27,0.37,-0.32,-0.13,0.19])   # loss = 7.14685765636982
    p4 = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.])   # loss = 10.392304845413262 
    p5 = torch.tensor([1.,-1.,1.,-1.,1.,1.,-1.,-1.,1.])   # loss = 3.4641016151377544
    p6 = torch.tensor([1.,-1.,-1.,-1.,1.,-1.,-1.,-1.,1.])   # loss = 0.0
    loss_miss1, groups_miss1, errors_miss1, missing_classes1 = bucakLasso(noisy_y1,p6)
    print(f"loss_miss1: {loss_miss1}")
    print(f"groups_miss1: {groups_miss1}")
    print(f"errors_miss1: {errors_miss1}")
    print(f"missing_classes1: {missing_classes1}")

test()
#bucakTest()
