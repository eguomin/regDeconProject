#include <stdio.h>
// tool functions
extern "C" {
#include "libapi.h"
}

int main(int argc, char* argv[])
{
	// *** print GPU devices information
	printf("Checking GPU device information ...\n");
	queryDevice();
	getchar();
	return 0;
}