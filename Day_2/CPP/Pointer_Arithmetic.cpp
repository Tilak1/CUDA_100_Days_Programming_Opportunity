/******************************************************************************

Welcome to GDB Online.
  GDB online is an online compiler and debugger tool for C, C++, Python, PHP, Ruby, 
  C#, OCaml, VB, Perl, Swift, Prolog, Javascript, Pascal, COBOL, HTML, CSS, JS
  Code, Compile, Run and Debug online from anywhere in world.

*******************************************************************************/
#include <stdio.h>

int main()
{

/*
Use ++x when you only want to increment x.

Use x++ when you want the original value of x and want to also increment x as a side effect.

Generally, use notation Whatever when you want the Whatever effect.
*/
    
int toys[] = {10, 20, 30};
int* p = toys;

int a = *p++;// Same as: a = *(p++); 
//As per precedence does postfix inc. But, the postfix 
// returns the old value / assigns the previous value of p to a. So, still pointing to 10. 


printf("a = %d\n", a);     // What is a?
printf("*p = %d\n", *p);   // Where is p now? postfix 
// o/p: a = 10 | *p = 20 

int b = ++*p; // both prefix and * are of same precedence. So R -> L. * will get 1st prefix

printf("b = %d\n", b);     // What is b?
printf("*p = %d\n", *p);   // Where is p now?

// o/p: b = 21, *p = 21 

int c = *++p; // R -> L. Prefix gets 1st pref and then the dereferencing


printf("c = %d\n", c);     // What is c?
printf("*p = %d\n", *p);   // Where is p now?


int d = (*p)++; // because of () *p gets preference and then the value gets inc 


printf("d = %d\n", d);     // What is d?
printf("*p = %d\n", *p);   // Where is p now?


}