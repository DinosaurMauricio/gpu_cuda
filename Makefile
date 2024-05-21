NVCC = nvcc
CC = gcc

LIB_HOME = .
LIBS = -L$(LIB_HOME)/lib64
INCLUDE = -Isrc
#OPT = -std=c++14 -g

MAIN = transpose.cu

BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/gpu_transpose

ifndef DATA_TYPE
DATA_TYPE = int
else
$(info DATA_TYPE is $(DATA_TYPE))
endif

ifeq ($(DATA_TYPE), int)
FORMAT_SPECIFIER = \\t%d
else
FORMAT_SPECIFIER = \\t%f
endif

ifndef BANDWIDTH_PERFORMANCE
BANDWIDTH_PERFORMANCE = O0
else
$(info BANDWIDTH_PERFORMANCE is $(BANDWIDTH_PERFORMANCE))
endif

OBJECTS = $(BUILDDIR)/my_library.o

$(TARGETDIR)/gpu_transpose: ${MAIN} $(OBJECTS)
	mkdir -p $(@D)
	$(NVCC) $< $(OBJECTS) -DDATA_TYPE=$(DATA_TYPE) -DFORMAT_SPECIFIER='"$(FORMAT_SPECIFIER)"' -o $@ $(INCLUDE) $(LIBS) $(OPT) -$(BANDWIDTH_PERFORMANCE)

$(BUILDDIR)/my_library.o: my_library.c
	mkdir -p $(BUILDDIR) $(TARGETDIR)
	$(CC) -c -DDATA_TYPE=$(DATA_TYPE) -DFORMAT_SPECIFIER='"$(FORMAT_SPECIFIER)"' -o $@ $(INCLUDE) my_library.c $(OPT) -$(BANDWIDTH_PERFORMANCE)


clean:
	rm -rf $(BUILDDIR)/*.o $(TARGETDIR)/*