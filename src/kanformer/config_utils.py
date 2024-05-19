import functools


def register_to_config(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {
            **{
                arg: value
                for arg, value in zip(
                    func.__code__.co_varnames[1 : len(args) + 1], args
                )
            },
            **kwargs,
        }
        func(self, *args, **kwargs)

    return wrapper
