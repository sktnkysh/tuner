def data():
    train_dir = 'standard_datasets/5a5f131db037cb8223459336/train' 
    test_dir = 'standard_datasets/5a5f131db037cb8223459336/validation'
    resize = 96 
    rescale = 1 
    df = load_data.df_fromdir(train_dir)
    x_train, y_train = load_data.load_fromdf(df, resize=resize, rescale=rescale)
    df = load_data.df_fromdir(test_dir)
    x_test, y_test = load_data.load_fromdf(df, resize=resize, rescale=rescale)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test
