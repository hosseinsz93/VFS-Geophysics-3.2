#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;

double x1, x2, z1, z2, y_h;
int ntx, ntz;
int Layout_type;

int main(int argc, char **argv) 
{  


	cout << "x begin coordinate: ";
	cin >> x1;

        cout << "x end coordinate: ";
        cin >> x2;

        cout << "z begin coordinate: ";
        cin >> z1;

        cout << "z end coordinate: ";
        cin >> z2;

        cout << "hub coordinate: ";
        cin >> y_h;

        cout << "Layout type align (1) stagger (2): ";
        cin >> Layout_type;

        cout << "num of turbine in x direction: ";
        cin >> ntx;

        cout << "num of turbine in z direction: ";
        cin >> ntz;


        double xt_loc[ntx][ntz];
        double zt_loc[ntx][ntz];



	double Sx = (x2-x1)/(double) ntx;
	double Sz = (z2-z1)/(double) ntz;

	int i, k;
	for (k=0; k<ntz; k++) 
	for (i=0; i<ntx; i++) {
		xt_loc[i][k] = x1 + 0.5*Sx + (double)i*Sx;
		zt_loc[i][k] = z1 + 0.5*Sz + (double)k*Sz;
	}

	if (Layout_type == 2) {

		for (k=0; k<ntz; k=k+2)
		for (i=0; i<ntx; i++) {
			xt_loc[i][k] = x1 + 0.25*Sx + (double)i*Sx;
			xt_loc[i][k+1] = x1 + 0.75*Sx + (double)i*Sx;
		}

	} 



        char filen[80];
        FILE * pFile;


        sprintf(filen, "CenterWT.dat");
        pFile = fopen(filen, "w");
        for (k=0; k<ntz; k++) 
        for (i=0; i<ntx; i++) {

        	fprintf(pFile, "%le %le %le \n", xt_loc[i][k], y_h, zt_loc[i][k]);
        }
        fclose(pFile);

}
