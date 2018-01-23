def data():
    train_dir = '/home/fytroo/Tuner/test/standard_datasets/5a66e0f7b037cb29b22f3106/auged' 
    test_dir = '/home/fytroo/Tuner/test/standard_datasets/5a66e0f7b037cb29b22f3106/validation'
    resize = 96 
    rescale = 1 
    df = load_data.df_fromdir_classed(train_dir)
    x_train, y_train = load_data.load_fromdf(df, resize=resize, rescale=rescale)
    df = load_data.df_fromdir_classed(test_dir)
    x_test, y_test = load_data.load_fromdf(df, resize=resize, rescale=rescale)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test
