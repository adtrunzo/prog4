FLAGS = -DDEBUG -arch=compute_35
LIBS = -lm
ALWAYS_REBUILD = makefile

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody2.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<

compute.o: compute2.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<

clean:
	rm -f *.o nbody

