#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;

int imax = 121;
double xl = 9.0;
int ntype = 2;
double h_wave = 0.016;
double lamda = 1.0;
double pi = 3.14159265358979323846;
int main(int argc, char **argv) 
{  
	int i;
	double xb[imax], yb[imax];
	double x, ybot; 
	double dx = xl / (double(imax)-1.0);

 	for(i=0; i<imax; i++) { 
    		xb[i]=(double(i))*dx;
	}
  	
  
  	for(i=0; i<imax; i++) {
//		if (ntype == 1) yb[i]=-h_wave*sin(0.5*pi+4*pi* xb[i]/xl);
		if (ntype == 1) yb[i]=h_wave*cos(2.0*pi*xb[i]/lamda);

   		if (ntype == 2) {
      			x=xb[i]*28.0;
      			if ((x >= 0.0) && (x < 9.0)) {
        			ybot=min(28.0,2.800000000000E+01 
             			+0.000000000000E+00*x 
             			+6.775070969851E-03*pow(x,2.0) 
             			-2.124527775800E-03*pow(x,3.0)); 
      			} else if ((x >= 9.0) && (x < 14.0)) {
        			ybot=(2.507355893131E+01 
             			+9.754803562315E-01*x 
             			-1.016116352781E-01*pow(x,2.0) 
             			+1.889794677828E-03*pow(x,3.0));
      			} else if ((x >= 14.0) && (x < 20.0)) {
        			ybot= (2.579601052357E+01 
              			+8.206693007457E-01*x 
              			-9.055370274339E-02*pow(x,2.0) 
              			+1.626510569859E-03*pow(x,3.0));
      			} else if ((x >= 20.0) && (x < 30.0)) {
        			ybot= (4.046435022819E+01 
             			-1.379581654948E+00*x 
             			+1.945884504128E-02*pow(x,2.0) 
             			-2.070318932190E-04*pow(x,3.0));
      			} else if ((x >= 30.0) && (x < 40.0)) {
        			ybot= (1.792461334664E+01 
             			+8.743920332081E-01*x 
             			-5.567361123058E-02*pow(x,2.0) 
             			+6.277731764683E-04*pow(x,3.0));
      			} else if ((x >= 40.0) && (x < 54.0)) {
        			ybot=max(0.,5.639011190988E+01 
             			-2.010520359035E+00*x 
             			+1.644919857549E-02*pow(x,2.0) 
             			+2.674976141766E-05*pow(x,3.0));
      			} else {
        			ybot=0.0;
      			}
    
      			yb[i]=ybot/28.0;
		}
	
	}

  	if (ntype == 2) {
    		for (i=imax-1; i>0.5*imax; i--) {
	  		yb[i]=yb[imax-i];
		}
  	}

        char filen[80];
        FILE * pFile;


        sprintf(filen, "hill.dat");
        pFile = fopen(filen, "w");

 	fprintf(pFile, "%d \n", imax );
       	for (i=0; i<imax; i++) {
        	fprintf(pFile, "%le %le %le \n", 0.0, yb[i], xb[i] );
        }
        fclose(pFile);
/*
c.....nmax is the number of segments
c.....ni(n) is the number of points on segment n
c.....imax is the number of points
integer nmax
integer ni(nmax)
real x(imax), y(imax), z(imax)
do n = 1, nmax
write(1,*) ni(n)
do i = 1, ni(n)
write(1,*) x(i,n), y(i,n), z(i,n)
end do
end do
*/

}
