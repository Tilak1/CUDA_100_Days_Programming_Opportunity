#include <iostream>
using namespace std;


int main()
{

int array[5];
int * p;
p = array;

*p = 10;
p++; *p = 20; // just incrementing base pointer address
p = &array[2]; *p = 30; // using the same base address with the right index

p = array+3; *p = 40; // incrementing base address to the desired pos
p = array; *(p+4)=50; // inc base to desired pos  and dereference operator to assign value then

for(int i = 0; i<5; i++)
{
        cout<<array[i]<<",";
}

return 0; }