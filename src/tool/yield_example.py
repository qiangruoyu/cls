def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
        print(a)

gen = fibonacci_generator()
for _ in range(10):
    print(next(gen))  # 输出前10个斐波那契数