class CustomCounter:
    def __init__(self):
        self.counter_dict = {}

    def increment(self, key, amount=1):
        if key not in self.counter_dict:
            self.counter_dict[key] = 0
            print('counter creating key: {}'.format(key))
        self.counter_dict[key] += amount

    def report(self):
        keys = list(self.counter_dict.keys())
        keys.sort()
        max_str_length = max([len(key) for key in keys])

        for key in keys:
            print("{0:{1:}s}: {2:}".format(key, max_str_length, self.counter_dict[key]))

