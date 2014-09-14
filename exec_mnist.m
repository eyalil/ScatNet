filter_type = getenv('filter_type')
filter_count = str2double(getenv('filters'))
times = str2double(getenv('times'))

run_mnist_times(filter_type, filter_count,times)
