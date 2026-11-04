CC     ?= gcc
CFLAGS  = -std=c11 -O3 -Wall -Wextra -Wpedantic -Iinclude
SRC     = $(wildcard src/*.c)

.PHONY: test debug

# Linux 개발용
libfoodcv.so: $(SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $^ -lm
# Windows 배포용 (Python이 64비트이므로 DLL도 64비트)
foodcv.dll: $(SRC)
	x86_64-w64-mingw32-gcc $(CFLAGS) -shared -o $@ $^ -lm
test:
	$(CC) $(CFLAGS) -DFCV_DEBUG_DUMP -o run_tests tests/*.c $(SRC) -lm && ./run_tests
debug:
	$(CC) -std=c11 -O0 -g -Iinclude -DFCV_DEBUG_DUMP -fsanitize=address,undefined -o run_tests tests/*.c $(SRC) -lm && ./run_tests
