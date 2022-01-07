rm *.o
gcc -c BinarySearch.c
gcc -c SearchMain.c
gcc -o search_exe BinarySearch.o SearchMain.o
./search_exe 