filter_type = getenv('filter_type')
filter_count = str2double(getenv('filters'))
times = str2double(getenv('times'))
dataset = str2double(getenv('dataset'))

run_mnist_times(dataset, filter_type, filter_count, times)
