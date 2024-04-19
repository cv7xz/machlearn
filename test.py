from threading import Thread

threads = [None] * 10
results = [None] * 10

def foo(bar, result, index):
    result[index] = f"foo-{index}"

for i in range(len(threads)):
    threads[i] = Thread(target=foo, args=('world!', results, i))
    threads[i].start()

for i in range(len(threads)):
    threads[i].join()

print (" ".join(results))