#include "search.h"
//A function whih does binary search on a given array
int search(int arr[], int size, int num)
{
    int start_in=0;
    int end_in=size-1;

    while (start_in <= end_in){
    
      int middle = start_in + (end_in- start_in )/2;
    
      if (arr[middle] == num)
         return middle;
    
      else if (arr[middle] < num)
         start_in = middle + 1;
    
      else
         end_in = middle - 1;
   }
   return -1;//returns -1 if the given no. doesn't belong to the given array.
}