libknf.so: knf.o
	$(CC) -o $@ $^ `pkg-config fftw3 fftw3f --libs` -shared
