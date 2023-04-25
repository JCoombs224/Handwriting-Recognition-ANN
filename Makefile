# Makefile for main.cpp using the sqlite3 library

# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -std=c++11

# Library flags
LIBS = -lsqlite3

# Source files
SRC = main.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Executable name
EXE = main

all: $(EXE)

$(EXE): $(OBJ)
	$(CC) $(CFLAGS) -o $(EXE) $(OBJ) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXE)