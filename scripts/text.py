import time
from functools import wraps


class Decos:
    def __init__(self, func):
        Decos.func = func

    @classmethod
    def third_level_timing(cls, g):
        def time_it(func):

            """
            Outputs the time a function takes
            to execute.
            """
            @wraps(func)
            def wrapper(*args, **kwargs):
                if g < 0:
                    return "HAHA! No Way!"
                else:
                    t1 = time.time()
                    func(*args, **kwargs)
                    t2 = time.time()
                    return "Time it took to run the function: " + str((t2 - t1)) + "\n"

            return wrapper
        return time_it


@Decos.third_level_timing(g=0)
def my_function2(st_num, en_num, g=0):
    num_list = []
    for num in (range(st_num, en_num + g)):
        num_list.append(num)
    math_sum = sum(num_list)

    print("\nSum of all the numbers: " + str(math_sum))


class Nothing:
    def __init__(self, k):
        self.k = k

    @Decos.third_level_timing(g=-2)
    def my_function(self, st_num, en_num, g=0):
        num_list = []
        for num in (range(st_num, en_num + g)):
            num_list.append(num)

        if self.k >= 1000:
            math_sum = sum(num_list) + self.k
        else:
            math_sum = sum(num_list)

        print("\nSum of all the numbers: " + str(math_sum))


if __name__ == '__main__':

    space = Nothing(k=1024)

    val1 = space.my_function(12, 300000)
    print(val1)

    val2 = my_function2(13, 244440)
    print(val2)



