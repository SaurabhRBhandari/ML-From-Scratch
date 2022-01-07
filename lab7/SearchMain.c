#include <stdio.h>
#include "search.h"

int main()
{
int arr[]={10,15,14,9,8,4,6,8,4,68,45};//Just a random array
int size_of_arr=sizeof(arr)/sizeof(arr[0]);
printf("The index is %d \n",search(arr,size_of_arr,68));
return 0;
}