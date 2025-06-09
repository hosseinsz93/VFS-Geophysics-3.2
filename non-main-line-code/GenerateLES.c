
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string>
#include <unistd.h>
#include <fftw3.h>
#include <time.h>

#include "petscvec.h"
//#include "petscda.h"
#include "petscksp.h"
#include "petscsnes.h"



#ifdef TECIO
#include "TECIO.h"
#endif

using namespace std;


/* 
  format of plot3d
        1
        3        3        1
  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  5.00000000000e-01
  5.00000000000e-01  5.00000000000e-01  1.00000000000e+00  1.00000000000e+00
  1.00000000000e+00
  0.00000000000e+00  5.00000000000e-01  1.00000000000e+00  0.00000000000e+00
  5.00000000000e-01  1.00000000000e+00  0.00000000000e+00  5.00000000000e-01
  1.00000000000e+00
  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00
  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00
  0.00000000000e+00
*/

int Nx=64, Ny=64, Nz=64; // The number of grid for synthetic turbulence

double Lx=5.0, Ly=2.0, Lz=1.0;

double Gamma=3.9;//3.9;

double ustar=0.05;

double L_coef=0.59;

double Disp_coef=3.2;

double Y_loc=0.5;

double Karman_const=0.4;

double Beta;


int Nt_WRF=13;
double dt_WRF =600.0; // seconds
int Nx_WRF=11, Ny_WRF=20, Nz_WRF=1;
double PI=3.14159265359;
double R_earth = 6371000;

int Nt_LES=28800; //28800;
double dt_LES =0.25; //0.25; // seconds
int Nx_LES=102, Ny_LES=82, Nz_LES=2;

int save_inflow_period=100;

double L_ref=1000.0;
double V_ref=10.0;
double Tmprt_ref=291.0;

char path[256]; 

double Uz_Convective;

typedef struct {
        PetscScalar x, y, z;
} Cmpnts;


extern double randn_notrig(); 

//PetscReal Time_WRF, Longitude_WRF, Latitude_WRF, Altitude_WRF, X_WRF, Y_WRF, Z_WRF, U_WRF, V_WRF, W_WRF, Qvapor_WRF, Pressure_WRF, Temperature_WRF;
//PetscReal U_WRFp1, V_WRFp1, W_WRFp1, Qvapor_WRFp1, Pressure_WRFp1, Temperature_WRFp1;
double *Time_LES;
double ***Longitude_LES, ***Latitude_LES, ***Altitude_LES, ***X_LES, ***Y_LES, ***Z_LES, ***U_LES, ***V_LES, ***W_LES, ***Qvapor_LES, ***Pressure_LES, ***Temperature_LES;
//PetscReal X_LES2, Y_LES2, Z_LES2;
//int Loc_intp_i, Loc_intp_j;

void GenerateTurb(); 

int main(int argc, char **argv) 
{
  
	int t,k,j,i;

	PetscReal Time_WRF[Nt_WRF], Longitude_WRF[Nz_WRF][Ny_WRF][Nx_WRF], Latitude_WRF[Nz_WRF][Ny_WRF][Nx_WRF], Altitude_WRF[Nz_WRF][Ny_WRF][Nx_WRF], X_WRF[Nz_WRF][Ny_WRF][Nx_WRF], Y_WRF[Nz_WRF][Ny_WRF][Nx_WRF], Z_WRF[Nz_WRF][Ny_WRF][Nx_WRF], U_WRF[Nz_WRF][Ny_WRF][Nx_WRF], V_WRF[Nz_WRF][Ny_WRF][Nx_WRF], W_WRF[Nz_WRF][Ny_WRF][Nx_WRF], Qvapor_WRF[Nz_WRF][Ny_WRF][Nx_WRF], Pressure_WRF[Nz_WRF][Ny_WRF][Nx_WRF], Temperature_WRF[Nz_WRF][Ny_WRF][Nx_WRF];
	PetscReal U_WRFp1[Nz_WRF][Ny_WRF][Nx_WRF], V_WRFp1[Nz_WRF][Ny_WRF][Nx_WRF], W_WRFp1[Nz_WRF][Ny_WRF][Nx_WRF], Qvapor_WRFp1[Nz_WRF][Ny_WRF][Nx_WRF], Pressure_WRFp1[Nz_WRF][Ny_WRF][Nx_WRF], Temperature_WRFp1[Nz_WRF][Ny_WRF][Nx_WRF];
//        PetscReal Time_LES[Nt_LES], Longitude_LES[Nz_LES][Ny_LES][Nx_LES], Latitude_LES[Nz_LES][Ny_LES][Nx_LES], Altitude_LES[Nz_LES][Ny_LES][Nx_LES], X_LES[Nz_LES][Ny_LES][Nx_LES], Y_LES[Nz_LES][Ny_LES][Nx_LES], Z_LES[Nz_LES][Ny_LES][Nx_LES], U_LES[Nz_LES][Ny_LES][Nx_LES], V_LES[Nz_LES][Ny_LES][Nx_LES], W_LES[Nz_LES][Ny_LES][Nx_LES], Qvapor_LES[Nz_LES][Ny_LES][Nx_LES], Pressure_LES[Nz_LES][Ny_LES][Nx_LES], Temperature_LES[Nz_LES][Ny_LES][Nx_LES];
        PetscReal X_LES2[Nz_LES][Ny_LES][Nx_LES], Y_LES2[Nz_LES][Ny_LES][Nx_LES], Z_LES2[Nz_LES][Ny_LES][Nx_LES];

	int Loc_intp_i[Nz_LES][Ny_LES][Nx_LES], Loc_intp_j[Nz_LES][Ny_LES][Nx_LES];

	Time_LES = (double*) malloc(sizeof(double)*Nt_LES);

	Longitude_LES = (double***) malloc(sizeof(double)*Nz_LES);
	Latitude_LES = (double***) malloc(sizeof(double)*Nz_LES);
	Altitude_LES = (double***) malloc(sizeof(double)*Nz_LES);
	X_LES = (double***) malloc(sizeof(double)*Nz_LES);
	Y_LES = (double***) malloc(sizeof(double)*Nz_LES);
	Z_LES = (double***) malloc(sizeof(double)*Nz_LES);
	U_LES = (double***) malloc(sizeof(double)*Nz_LES);
	V_LES = (double***) malloc(sizeof(double)*Nz_LES);
	W_LES = (double***) malloc(sizeof(double)*Nz_LES);
	Qvapor_LES = (double***) malloc(sizeof(double)*Nz_LES);
	Pressure_LES = (double***) malloc(sizeof(double)*Nz_LES);
	Temperature_LES = (double***) malloc(sizeof(double)*Nz_LES);

	for(k=0;k<Nz_LES;k++) {
		Longitude_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		Latitude_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		Altitude_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		X_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		Y_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		Z_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		U_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		V_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		W_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		Qvapor_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		Pressure_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
		Temperature_LES[k] = (double**) malloc(sizeof(double)*Ny_LES);
	}


	for(k=0;k<Nz_LES;k++) 
	for(j=0;j<Ny_LES;j++) {
		Longitude_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		Latitude_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		Altitude_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		X_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		Y_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		Z_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		U_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		V_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		W_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		Qvapor_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		Pressure_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
		Temperature_LES[k][j] = (double*) malloc(sizeof(double)*Nx_LES);
	}



	sprintf(path, "/home/xyang/MowerCounty_WRF-CURVIB/NoTurb_inflow");

        Cmpnts **ucat, **ucat1;

        ucat = (Cmpnts **)malloc( sizeof(Cmpnts *) * Ny_LES );

        for(j=0; j<Ny_LES; j++) {
                ucat[j] = (Cmpnts *)malloc( sizeof(Cmpnts) * Nx_LES );
        }

        ucat1 = (Cmpnts **)malloc( sizeof(Cmpnts *) * Ny_LES );

        for(j=0; j<Ny_LES; j++) {
                ucat1[j] = (Cmpnts *)malloc( sizeof(Cmpnts) * Nx_LES );
        }

	double T_ref=L_ref/V_ref;
	double dt_LES_scaled=dt_LES/T_ref;

	double fac=1.0/double(Nt_LES*Nx_LES*Nz_LES);


	double Uavg[Ny_LES], Vavg[Ny_LES], Wavg[Ny_LES], Qavg[Ny_LES], Pavg[Ny_LES], Tavg[Ny_LES], Yavg[Ny_LES];

	for (j=0; j<Ny_LES; j++) {
		Uavg[j]=0.0;
		Vavg[j]=0.0;
		Wavg[j]=0.0;
		Pavg[j]=0.0;
		Qavg[j]=0.0;
		Tavg[j]=0.0;
		Yavg[j]=0.0;
	}



	for (t=0;t<Nt_LES;t++) {
		Time_LES[t]=dt_LES*t;
	} 

        for (t=0;t<Nt_WRF;t++) {
                Time_WRF[t]=dt_WRF*t;
        }


    	FILE *fd;
    	char filen[80];  
 
        sprintf(filen, "grid_WRF.grd");
        fd = fopen(filen, "r");

	int Num_face, Nx_WRFg, Ny_WRFg, Nz_WRFg;
 	fscanf(fd, "%d \n", &Num_face );
 	fscanf(fd, "%d %d %d \n", &Nx_WRFg, &Ny_WRFg, &Nz_WRFg );

	if (Nx_WRF!=Nx_WRFg || Ny_WRF!=Ny_WRFg || Nz_WRF!=Nz_WRFg) {
		printf("the dimension of the grid is inconsistant the specified \n");
		exit(0);
	}

	printf("read WRF grid \n");
       	for (k=0; k<Nz_WRF; k++)  
       	for (j=0; j<Ny_WRF; j++)  
       	for (i=0; i<Nx_WRF; i++) { 
        	fscanf(fd, "%le ", &X_WRF[k][j][i] );
        }
        fscanf(fd, "\n");
       
       	for (k=0; k<Nz_WRF; k++)  
        for (j=0; j<Ny_WRF; j++) 
 	for (i=0; i<Nx_WRF; i++) {
        	fscanf(fd, "%le ", &Y_WRF[k][j][i] );
        }
        fscanf(fd, "\n");

       	for (k=0; k<Nz_WRF; k++)  
        for (j=0; j<Ny_WRF; j++) 
        for (i=0; i<Nx_WRF; i++) {
        	fscanf(fd, "%le ", &Z_WRF[k][j][i] );
        }

        fclose(fd);
	
	int Nx_LESg, Ny_LESg, Nz_LESg;
  
        sprintf(filen, "grid_LES_FromWRF.grd");
        fd = fopen(filen, "r");

//	int Nx_LESg, Ny_LESg, Nz_LESg;
 	fscanf(fd, "%d \n", &Num_face );
 	fscanf(fd, "%d %d %d \n", &Nx_LESg, &Ny_LESg, &Nz_LESg );

	if (Nx_LES-1!=Nx_LESg || Ny_LES-1!=Ny_LESg || Nz_LES-1!=Nz_LESg) {
		printf("the dimension of the grid is inconsistant the specified \n");
		exit(0);
	}

	printf("read the LES grid from WRF\n");
       	for (k=1; k<Nz_LES; k++)  
       	for (j=1; j<Ny_LES; j++)  
       	for (i=1; i<Nx_LES; i++) { 
        	fscanf(fd, "%le ", &X_LES2[k][j][i] );
        }
       
       	for (k=1; k<Nz_LES; k++)  
 	for (j=1; j<Ny_LES; j++) 
        for (i=1; i<Nx_LES; i++) {
        	fscanf(fd, "%le ", &Y_LES2[k][j][i] );
        }

       	for (k=1; k<Nz_LES-1; k++)  
        for (j=1; j<Ny_LES-1; j++) 
        for (i=1; i<Nx_LES-1; i++) {
        	fscanf(fd, "%le ", &Z_LES2[k][j][i] );
        }

        fclose(fd);
	
        sprintf(filen, "grid_LES.grd");
        fd = fopen(filen, "r");

 	fscanf(fd, "%d \n", &Num_face );
 	fscanf(fd, "%d %d %d \n", &Nx_LESg, &Ny_LESg, &Nz_LESg );

	if (Nx_LES-1!=Nx_LESg || Ny_LES-1!=Ny_LESg || Nz_LES-1!=Nz_LESg) {
		printf("the dimension of the grid is inconsistant the specified \n");
		exit(0);
	}

	printf("read LES grid \n");
       	for (k=1; k<Nz_LES; k++)  
       	for (j=1; j<Ny_LES; j++)  
       	for (i=1; i<Nx_LES; i++) { 
        	fscanf(fd, "%le ", &X_LES[k][j][i] );
        }
       
       	for (k=1; k<Nz_LES; k++)  
        for (j=1; j<Ny_LES; j++) 
 	for (i=1; i<Nx_LES; i++) {
        	fscanf(fd, "%le ", &Y_LES[k][j][i] );
        }

       	for (k=1; k<Nz_LES; k++)  
        for (j=1; j<Ny_LES; j++) 
        for (i=1; i<Nx_LES; i++) {
        	fscanf(fd, "%le ", &Z_LES[k][j][i] );
        }

        fclose(fd);
	
	// only for Cartesian grid
	printf("Identify interpolation points \n");
	int i_1, j_1, k_1;

        for (k=1; k<Nz_LES; k++)
        for (j=1; j<Ny_LES; j++)
        for (i=1; i<Nx_LES; i++) {
//		printf("Identify interpolation points %d %d\n", j, i);
	        k_1=0;
        	j_1=0;
	        for (i_1=0; i_1<Nx_WRF-1; i_1++) {
			if ((X_LES[k][j][i]-X_WRF[k_1][j_1][i_1])>-1e-9 && (X_LES[k][j][i]-X_WRF[k_1][j_1][i_1+1])<1e-9 ) Loc_intp_i[k][j][i]=i_1;	
		}

                k_1=0;
                i_1=0;
                for (j_1=0; j_1<Ny_WRF-1; j_1++) {
                        if ((Y_LES[k][j][i]-Y_WRF[k_1][j_1][i_1])>-1e-9 && (Y_LES[k][j][i]-Y_WRF[k_1][j_1+1][i_1])<1e-9)  Loc_intp_j[k][j][i]=j_1;           
                }

	}

	// Generating Inflow Turbulence
//	GenerateTurb();


	int t_LES, t_WRF, tWRF_intp;
	
	for(t_LES=0;t_LES<Nt_LES;t_LES++) {

		printf("t_LES %d \n", t_LES);
		for(t_WRF=0;t_WRF<Nt_WRF-1;t_WRF++) {
			if((Time_WRF[t_WRF]-Time_LES[t_LES])<1e-9 && (Time_WRF[t_WRF+1]-Time_LES[t_LES])>-1e-9) tWRF_intp = t_WRF;
			
		}

		printf("Read WRF Inflow data at time %d \n", tWRF_intp);
                char fname[256];

                sprintf(fname, "./inflow_fromWRF_%06d_dt=%gSec.dat", tWRF_intp, dt_WRF);

                FILE *fp=fopen(fname, "rb");
		for(k=0;k<Nz_WRF;k++)
                for(j=0;j<Ny_WRF;j++) {
			fread(&U_WRF[k][j][0], sizeof(double), Nx_WRF, fp);
			fread(&V_WRF[k][j][0], sizeof(double), Nx_WRF, fp);
			fread(&W_WRF[k][j][0], sizeof(double), Nx_WRF, fp);
		}
                fclose(fp);


		printf("Read WRF Inflow data at time %d \n", tWRF_intp+1);
                sprintf(fname, "./inflow_fromWRF_%06d_dt=%gSec.dat", tWRF_intp+1, dt_WRF);

                fp=fopen(fname, "rb");
		for(k=0;k<Nz_WRF;k++)
                for(j=0;j<Ny_WRF;j++) {
			fread(&U_WRFp1[k][j][0], sizeof(double), Nx_WRF, fp);
			fread(&V_WRFp1[k][j][0], sizeof(double), Nx_WRF, fp);
			fread(&W_WRFp1[k][j][0], sizeof(double), Nx_WRF, fp);
		}
                fclose(fp);

		printf("Read WRF Inflow Temperature data at time %d \n", tWRF_intp);

                sprintf(fname, "./inflowTmprt_fromWRF_%06d_dt=%gSec.dat", tWRF_intp, dt_WRF);

                fp=fopen(fname, "rb");
		for(k=0;k<Nz_WRF;k++)
                for(j=0;j<Ny_WRF;j++) {
			fread(&Temperature_WRF[k][j][0], sizeof(double), Nx_WRF, fp);
		}
                fclose(fp);


		printf("Read WRF Inflow Temperature data at time %d \n", tWRF_intp+1);
                sprintf(fname, "./inflowTmprt_fromWRF_%06d_dt=%gSec.dat", tWRF_intp+1, dt_WRF);

                fp=fopen(fname, "rb");
		for(k=0;k<Nz_WRF;k++)
                for(j=0;j<Ny_WRF;j++) {
			fread(&Temperature_WRFp1[k][j][0], sizeof(double), Nx_WRF, fp);
		}
                fclose(fp);


		printf("Interpolate over time \n");
		double fac1=(Time_WRF[tWRF_intp+1]-Time_LES[t_LES])/(Time_WRF[tWRF_intp+1]-Time_WRF[tWRF_intp]);
		double fac2=(-Time_WRF[tWRF_intp]+Time_LES[t_LES])/(Time_WRF[tWRF_intp+1]-Time_WRF[tWRF_intp]);
//		printf("TIme %le  %le %le  \n", Time_LES[t_LES], Time_WRF[tWRF_intp+1], Time_WRF[tWRF_intp]);
		for(k=0;k<Nz_WRF;k++)
                for(j=0;j<Ny_WRF;j++) 
		for(i=0;i<Nx_WRF;i++) {
			U_WRF[k][j][i]=fac1*U_WRF[k][j][i]+fac2*U_WRFp1[k][j][i];
			V_WRF[k][j][i]=fac1*V_WRF[k][j][i]+fac2*V_WRFp1[k][j][i];
			W_WRF[k][j][i]=fac1*W_WRF[k][j][i]+fac2*W_WRFp1[k][j][i];
			Temperature_WRF[k][j][i]=fac1*Temperature_WRF[k][j][i]+fac2*Temperature_WRFp1[k][j][i];
		}

		printf("Interpolation to LES grid \n");
		for(k=1;k<Nz_LES;k++)
		for(j=1;j<Ny_LES;j++)
		for(i=1;i<Nx_LES;i++) {

			int i_WRF = Loc_intp_i[k][j][i];
			int j_WRF = Loc_intp_j[k][j][i];

			double fac1_x = (X_WRF[0][j_WRF][i_WRF+1]-X_LES[k][j][i])/(X_WRF[0][j_WRF][i_WRF+1]-X_WRF[0][j_WRF][i_WRF]);
			double fac2_x = (-X_WRF[0][j_WRF][i_WRF]+X_LES[k][j][i])/(X_WRF[0][j_WRF][i_WRF+1]-X_WRF[0][j_WRF][i_WRF]);
                        double fac1_y = (Y_WRF[0][j_WRF+1][i_WRF]-Y_LES[k][j][i])/(Y_WRF[0][j_WRF+1][i_WRF]-Y_WRF[0][j_WRF][i_WRF]);
                        double fac2_y = (-Y_WRF[0][j_WRF][i_WRF]+Y_LES[k][j][i])/(Y_WRF[0][j_WRF+1][i_WRF]-Y_WRF[0][j_WRF][i_WRF]);

			U_LES[k][j][i]=fac1_x*(fac1_y*U_WRF[0][j_WRF][i_WRF]+fac2_y*U_WRF[0][j_WRF+1][i_WRF]) + fac2_x*(fac1_y*U_WRF[0][j_WRF][i_WRF+1]+fac2_y*U_WRF[0][j_WRF+1][i_WRF+1]);

			V_LES[k][j][i]=fac1_x*(fac1_y*V_WRF[0][j_WRF][i_WRF]+fac2_y*V_WRF[0][j_WRF+1][i_WRF]) + fac2_x*(fac1_y*V_WRF[0][j_WRF][i_WRF+1]+fac2_y*V_WRF[0][j_WRF+1][i_WRF+1]);

			W_LES[k][j][i]=fac1_x*(fac1_y*W_WRF[0][j_WRF][i_WRF]+fac2_y*W_WRF[0][j_WRF+1][i_WRF]) + fac2_x*(fac1_y*W_WRF[0][j_WRF][i_WRF+1]+fac2_y*W_WRF[0][j_WRF+1][i_WRF+1]);

			Temperature_LES[k][j][i]=fac1_x*(fac1_y*Temperature_WRF[0][j_WRF][i_WRF]+fac2_y*Temperature_WRF[0][j_WRF+1][i_WRF]) + fac2_x*(fac1_y*Temperature_WRF[0][j_WRF][i_WRF+1]+fac2_y*Temperature_WRF[0][j_WRF+1][i_WRF+1]);

		}

                for(j=Ny_LES-1;j>=1;j--)
                for(i=1;i<Nx_LES;i++) {
			if (Y_LES[1][j][i]<=Y_LES2[1][0][i]) {
				double fac1=(Y_LES[1][j+1][i]-Y_LES[1][j][i])/(Y_LES[1][j+1][i]-Y_LES[1][j+2][i]+1.e-9);
				double fac2=(Y_LES[1][j][i]-Y_LES[1][j+2][i])/(Y_LES[1][j+1][i]-Y_LES[1][j+2][i]+1.e-9);

				U_LES[1][j][i]=fac1*U_LES[1][j+2][i]+fac2*U_LES[1][j+1][i];
				V_LES[1][j][i]=fac1*V_LES[1][j+2][i]+fac2*V_LES[1][j+1][i];
				W_LES[1][j][i]=fac1*W_LES[1][j+2][i]+fac2*W_LES[1][j+1][i];
				Temperature_LES[1][j][i]=fac1*Temperature_LES[1][j+2][i]+fac2*Temperature_LES[1][j+1][i];
			}
		}

		double Z_intp=Uz_Convective*dt_LES_scaled;



                for(j=1;j<Ny_LES;j++)
                for(i=1;i<Nx_LES;i++) {

//			double Umag=sqrt(pow(U_LES[1][j][i],2)+pow(V_LES[1][j][i],2)+pow(W_LES[1][j][i],2));
			double Umag=sqrt(pow(W_LES[1][j][i],2));
			ucat[j][i].x=0.0; //0.1*Umag*randn_notrig()/V_ref; //(U_LES[1][j][i]+0.1*Umag*randn_notrig())/V_ref;
			ucat[j][i].y=0.0; //0.1*Umag*randn_notrig()/V_ref; //(V_LES[1][j][i]+0.1*Umag*randn_notrig())/V_ref;
			ucat[j][i].z=W_LES[1][j][i]/V_ref; //+0.1*Umag*randn_notrig())/V_ref;
			Temperature_LES[1][j][i]=Temperature_LES[1][j][i]-Tmprt_ref; //+0.1*(Temperature_LES[0][j][i]-Tmprt_ref)*randn_notrig();
		}

                for(j=1;j<Ny_LES;j++)
                for(i=1;i<Nx_LES;i++) {
			U_LES[1][j][i]=ucat[j][i].x;
			V_LES[1][j][i]=ucat[j][i].y;
			W_LES[1][j][i]=ucat[j][i].z;
		}




		printf("Write LES Inflow \n");
                char fname_LES[256];
                int ti_name = ( (t_LES-1) / save_inflow_period ) * save_inflow_period + 1;

                sprintf(fname_LES, "%s/inflow_%06d_dt=%g.dat", path, ti_name, dt_LES_scaled);
                if(t_LES==0 || t_LES%save_inflow_period==1) unlink(fname_LES);    // delete existing file

                FILE *fp_LES=fopen(fname_LES, "ab");
                if(!fp) printf("\n******************* Cannot open %s ! *******************\n", fname_LES),exit(0);

                for(j=0; j<Ny_LES; j++) fwrite(&ucat[j][0], sizeof(Cmpnts), Nx_LES, fp_LES);
                fclose(fp_LES);



		printf("Write Temperature Inflow \n");
                char fname_tp[256];
               	ti_name = ( (t_LES-1) / save_inflow_period ) * save_inflow_period + 1;

                sprintf(fname_tp, "%s/inflow_t_%06d_dt=%g.dat", path, ti_name, dt_LES_scaled);

	        if(t_LES==0 || t_LES%save_inflow_period==1) unlink(fname_tp);    // delete existing file

        	FILE *fp_tp=fopen(fname_tp, "ab");
	        if(!fp) printf("\n******************* Cannot open %s ! *******************\n", fname_tp),exit(0);

      	        for(j=0; j<Ny_LES; j++) fwrite(&Temperature_LES[1][j][0], sizeof(double), Nx_LES, fp_tp);
                fclose(fp_tp);



		if(t_LES%save_inflow_period==0) {
//		if(t_LES%1==0) {
      		printf("output tecplot file\n");
    		FILE *f;
    		char filen_tec[80];
    		sprintf(filen_tec, "FlowField_LES%06d.dat",t_LES);
    		f = fopen(filen_tec, "w");
    		fprintf(f, "Variables=x,y,z,U,V,W, T\n");

		fprintf(f, "ZONE T='QUADRILATERAL', N=%d, E=%d, F=FEBLOCK, ET=QUADRILATERAL, VARLOCATION=([1-7]=NODAL)\n", (Nx_LES-1)*(Ny_LES-1), (Nx_LES-2)*(Ny_LES-2));

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
      			fprintf(f, "%le\n", X_LES[k][j][i]);
		}

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", Y_LES[k][j][i]);
                }

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                       	fprintf(f, "%le\n", Z_LES[k][j][i]);
                }


                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", U_LES[k][j][i]);
                }

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", V_LES[k][j][i]);
                }

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", W_LES[k][j][i]);
                }

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", Temperature_LES[k][j][i]);
                }



                for (j=1;j<Ny_LES-1;j++)
                for (i=1;i<Nx_LES-1;i++) {
		     	fprintf(f, "%d %d %d %d\n", (j-1)*(Nx_LES-1)+i, (j-1)*(Nx_LES-1)+i+1, (j)*(Nx_LES-1)+i+1, (j)*(Nx_LES-1)+i);
		}
    
	    	fclose(f);
	
		}

		for (k=1; k<Nz_LES; k++)
		for (j=1; j<Ny_LES; j++)
		for (i=1; i<Nx_LES; i++) {
			Uavg[j] += fac*U_LES[k][j][i];
			Vavg[j] += fac*V_LES[k][j][i];
			Wavg[j] += fac*W_LES[k][j][i];
			Tavg[j] += fac*Temperature_LES[k][j][i];
			Yavg[j] += fac*Y_LES[k][j][i];
		}



	}


        double **tmprt_plane;

        tmprt_plane = (double **)malloc( sizeof(double *) * Ny_LES );

        for(j=0; j<Ny_LES; j++) {
                tmprt_plane[j] = (double *)malloc( sizeof(double) * Nx_LES );
        }

	
	int tistart=0;
	for(t_LES=0;t_LES<Nt_LES;t_LES++) {
	        char fname[256];
	        int ti2=t_LES;
        
	        if(ti2==0) ti2=1;
                
	        int ti_name = ( (ti2-1) / save_inflow_period ) * save_inflow_period + 1;

		FILE *fp_LES2;
	      	sprintf(fname, "%s/inflow_%06d_dt=%g.dat", path, ti_name, dt_LES_scaled);

           	if(t_LES==tistart || (t_LES>tistart+90 && ti2%save_inflow_period==1) ) {
                        if(t_LES!=tistart) fclose(fp_LES2);
                	fp_LES2=fopen(fname, "rb");
                        if(!fp_LES2) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(0);
                        
                        if(t_LES==tistart) {       // move position; 100616
                                for(int it=0; it<(ti2-1)%save_inflow_period; it++) {
                                        for(j=0; j<Ny_LES; j++) fread(&ucat1[j][0], sizeof(Cmpnts), Nx_LES, fp_LES2);
                                }
                        }
                }
                if(tistart==0 && t_LES==1) {}
                else for(j=0; j<Ny_LES; j++) fread(&ucat1[j][0], sizeof(Cmpnts), Nx_LES, fp_LES2);

                for(j=1;j<Ny_LES;j++)
                for(i=1;i<Nx_LES;i++) {
			U_LES[1][j][i]=ucat1[j][i].x;
			V_LES[1][j][i]=ucat1[j][i].y;
			W_LES[1][j][i]=ucat1[j][i].z;
		}

	        char fname_T[256];
		FILE *fp_LES2_T;
		sprintf(fname_T, "%s/inflow_t_%06d_dt=%g.dat", path, ti_name, dt_LES_scaled);
			
		if(t_LES==tistart || (t_LES>tistart+90 && ti2%save_inflow_period==1) ) {
				
			if(t_LES!=tistart) fclose(fp_LES2_T);
			fp_LES2_T=fopen(fname_T, "rb");
			if(!fp_LES2_T) printf("\n******************* Cannot open %s ! *******************\n", fname_T),exit(0);
					
			if(t_LES==tistart) {	// move position; 100616
				for(int it=0; it<(ti2-1)%save_inflow_period; it++) {
					for(j=0; j<Ny_LES; j++) fread(&tmprt_plane[j][0], sizeof(double), Nx_LES, fp_LES2_T);
				}
			}
		}
			
		for(j=0; j<Ny_LES; j++) fread(&tmprt_plane[j][0], sizeof(double), Nx_LES, fp_LES2_T);
			
		printf("\nRead inflow data from %s ! \n", fname);

                for(j=1;j<Ny_LES;j++)
                for(i=1;i<Nx_LES;i++) {
			Temperature_LES[1][j][i]=tmprt_plane[j][i];
		}


		if(t_LES%save_inflow_period==0) {
//		if(t_LES%1==0) {
      		printf("output tecplot file\n");
    		FILE *f;
    		char filen_tec[80];
    		sprintf(filen_tec, "TESTFlowField_LES%06d.dat",t_LES);
    		f = fopen(filen_tec, "w");
    		fprintf(f, "Variables=x,y,z,U,V,W, T\n");

		fprintf(f, "ZONE T='QUADRILATERAL', N=%d, E=%d, F=FEBLOCK, ET=QUADRILATERAL, VARLOCATION=([1-7]=NODAL)\n", (Nx_LES-1)*(Ny_LES-1), (Nx_LES-2)*(Ny_LES-2));

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
      			fprintf(f, "%le\n", X_LES[k][j][i]);
		}

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", Y_LES[k][j][i]);
                }

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                       	fprintf(f, "%le\n", Z_LES[k][j][i]);
                }


                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", U_LES[k][j][i]);
                }

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", V_LES[k][j][i]);
                }

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", W_LES[k][j][i]);
                }

                for (k=1;k<Nz_LES;k++)
                for (j=1;j<Ny_LES;j++)
                for (i=1;i<Nx_LES;i++) {
                        fprintf(f, "%le\n", Temperature_LES[k][j][i]);
                }



                for (j=1;j<Ny_LES-1;j++)
                for (i=1;i<Nx_LES-1;i++) {
		     	fprintf(f, "%d %d %d %d\n", (j-1)*(Nx_LES-1)+i, (j-1)*(Nx_LES-1)+i+1, (j)*(Nx_LES-1)+i+1, (j)*(Nx_LES-1)+i);
		}
    
	    	fclose(f);
	
		}

		for (k=1; k<Nz_LES; k++)
		for (j=1; j<Ny_LES; j++)
		for (i=1; i<Nx_LES; i++) {
			Uavg[j] += fac*U_LES[k][j][i];
			Vavg[j] += fac*V_LES[k][j][i];
			Wavg[j] += fac*W_LES[k][j][i];
			Tavg[j] += fac*Temperature_LES[k][j][i];
			Yavg[j] += fac*Y_LES[k][j][i];
		}



	}

/*
        sprintf(filen, "grid_LES3.grd");
        fd = fopen(filen, "w");

 	fprintf(fd, "%d \n", 1 );
 	fprintf(fd, "%d %d %d \n", Nx_LES, Ny_LES, Nz_LES );
       	for (k=0; k<Nz_LES-1; k++)  
       	for (j=0; j<Ny_LES-1; j++)  
       	for (i=0; i<Nx_LES-1; i++) { 
        	fprintf(fd, "%le ", X_LES2[k][j][i] );
        }
        fprintf(fd, "\n");
       
       	for (k=0; k<Nz_LES-1; k++)  
        for (j=0; j<Ny_LES-1; j++) 
 	for (i=0; i<Nx_LES-1; i++) {
        	fprintf(fd, "%le ", Y_LES2[k][j][i] );
        }
        fprintf(fd, "\n");

       	for (k=0; k<Nz_LES-1; k++)  
        for (j=0; j<Ny_LES-1; j++) 
        for (i=0; i<Nx_LES-1; i++) {
        	fprintf(fd, "%le ", Z_LES2[k][j][i] );
        }

        fclose(fd);
*/	

        sprintf(filen, "Profile_avg_LES.plt");
        fd = fopen(filen, "w");

 	fprintf(fd, "Variables = Y, U, V, W, Tmprt \n");
	for (j=1; j<Ny_LES; j++) fprintf(fd, "%le %le %le %le %le  \n", Yavg[j], Uavg[j], Vavg[j], Wavg[j], Tavg[j]);

        fclose(fd);
 
	for(k=0;k<Nz_LES;k++) 
	for(j=0;j<Ny_LES;j++) {
		free(Longitude_LES[k][j]);
		free(Latitude_LES[k][j]);
		free(Altitude_LES[k][j]);
		free(X_LES[k][j]);
		free(Y_LES[k][j]);
		free(Z_LES[k][j]);
		free(U_LES[k][j]);
		free(V_LES[k][j]);
		free(W_LES[k][j]);
		free(Qvapor_LES[k][j]);
		free(Pressure_LES[k][j]);
		free(Temperature_LES[k][j]);
	}


        for(j=0; j<Ny_LES; j++) {
                free(ucat[j]);
                free(ucat1[j]);
        }


}


void GenerateTurb() 
{
  
	int t,k,j,i;

	double TC11, TC12, TC13, TC21, TC22, TC23, TC31, TC32, TC33;

	double *Kx, *Ky, *Kz;
	double *X, *Y, *Z;

	double C11, C12, C13, C21, C22, C23, C31, C32, C33;
	double ***U, ***V, ***W;
	double ***U_i, ***V_i, ***W_i;
	double ***n_Gauss; 


	Ly=fabs(X_LES[1][1][Nx_LES-1]-X_LES[1][1][1]);
	Lz=fabs(Y_LES[1][Ny_LES-1][1]-Y_LES[1][1][1]);
//	Lx=fabs(Z_LES[Nz_LES-1][1][1]-Z_LES[1][1][1]);
	Lx=4.0*Ly;

//	Nx=Nz_LES-2;
	Ny=Nx_LES-2;
	Nz=Ny_LES-2;
	Nx=4*Ny;

	
	Kx = (double*) malloc(sizeof(double)*Nx);
	Ky = (double*) malloc(sizeof(double)*Ny);
	Kz = (double*) malloc(sizeof(double)*Nz);

	X = (double*) malloc(sizeof(double)*Nx);
	Y = (double*) malloc(sizeof(double)*Ny);
	Z = (double*) malloc(sizeof(double)*Nz);

	U = (double***) malloc(sizeof(double)*Nz);
	V = (double***) malloc(sizeof(double)*Nz);
	W = (double***) malloc(sizeof(double)*Nz);

	U_i = (double***) malloc(sizeof(double)*Nz);
	V_i = (double***) malloc(sizeof(double)*Nz);
	W_i = (double***) malloc(sizeof(double)*Nz);

	n_Gauss = (double***) malloc(sizeof(double)*Nz);

	for(k=0;k<Nz;k++) {
		U[k] = (double**) malloc(sizeof(double)*Ny);
		V[k] = (double**) malloc(sizeof(double)*Ny);
		W[k] = (double**) malloc(sizeof(double)*Ny);

		U_i[k] = (double**) malloc(sizeof(double)*Ny);
		V_i[k] = (double**) malloc(sizeof(double)*Ny);
		W_i[k] = (double**) malloc(sizeof(double)*Ny);

		n_Gauss[k] = (double**) malloc(sizeof(double)*Ny);
	}


	for(k=0;k<Nz;k++) 
	for(j=0;j<Ny;j++) {
		U[k][j] = (double*) malloc(sizeof(double)*Nx);
		V[k][j] = (double*) malloc(sizeof(double)*Nx);
		W[k][j] = (double*) malloc(sizeof(double)*Nx);

		U_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		V_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		W_i[k][j] = (double*) malloc(sizeof(double)*Nx);

		n_Gauss[k][j] = (double*) malloc(sizeof(double)*Nx);
	}
/*
	for(i=0;i<Nx;i++) {
		Kx[i]=i; //(i)*2.0*PI/Nx;
	}

	for(j=0;j<Ny;j++) {
		Ky[j]=j; //(j)*2.0*PI/Ny;
	}

	for(k=0;k<Nz;k++) {
		Kz[k]=k; //(k)*2.0*PI/Nz;
	}
*/


	for(i=0;i<Nx/2+1;i++) {
		Kx[i]=i*2.0*PI/Lx;
	}

	for(j=0;j<Ny/2+1;j++) {
		Ky[j]=j*2.0*PI/Ly;
	}

	for(k=0;k<Nz/2+1;k++) {
		Kz[k]=k*2.0*PI/Lz;
	}

	for(i=Nx/2+1;i<Nx;i++) {
		Kx[i]=(i-Nx)*2.0*PI/Lx;
	}

	for(j=Ny/2+1;j<Ny;j++) {
		Ky[j]=(j-Ny)*2.0*PI/Ly;
	}

	for(k=Nz/2+1;k<Nz;k++) {
		Kz[k]=(k-Nz)*2.0*PI/Lz;
	}




	double dy=Ly/(Ny-1);

	for(j=0;j<Ny;j++) {
		Y[j]=double(j)*dy;
	}

	double dx=Lx/(Nx-1);

	for(i=0;i<Nx;i++) {
		X[i]=double(i)*dx;
	}

	double dz=Lz/(Nz-1);

	for(k=0;k<Nz;k++) {
		Z[k]=double(k)*dz;
	}


        FILE *f_tmp;
        char fname_tmp[80];  
        sprintf(fname_tmp, "E_k.dat");
        f_tmp = fopen(fname_tmp, "w");
	j=0; k=0;
	for (i=0;i<Nx;i++) {

		double K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+Kz[k]*Kz[k]);
		double fac=1.0/(sqrt(4.0*PI)*K*K+1.e-9);
		double fac1 = 3.2*ustar*ustar/pow(Y_loc,2.0/3.0);
		double L = 0.59*Y_loc;
		double fac2 = pow(L,5.0/3.0);
		double fac3 = pow(L,4)*pow(K,4)/pow((1+pow(L,2)*pow(K,2)),17.0/6.0);

		double E=  fac1 * fac2 * fac3;   // alpha * e^2/3 *  L^5/3 *  L^4 k^4 / (1+L^2K^2)^(17/6) 
		
		double FF=E;
        	fprintf( f_tmp, "%le %le \n", K, FF );
	}


	fclose(f_tmp);

	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		n_Gauss[k][j][i] = randn_notrig();
	}
	

	fftw_complex *CKx, *CKy, *CKz;
	fftw_complex *CKx_o, *CKy_o, *CKz_o;

	CKx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx);
	CKx_o = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx);

	CKy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny);
	CKy_o = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny);

	CKz = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nz);
	CKz_o = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nz);

	fftw_plan px;
	px = fftw_plan_dft_1d(Nx, CKx, CKx_o, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan py;
	py = fftw_plan_dft_1d(Ny, CKy, CKy_o, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan pz;
	pz = fftw_plan_dft_1d(Nz, CKz, CKz_o, FFTW_FORWARD, FFTW_ESTIMATE);

	printf("%le %le %le \n", randn_notrig(), randn_notrig(), randn_notrig() );


	// u
	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) {
		for(i=0;i<Nx;i++) {

			double K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+Kz[k]*Kz[k]);
			double fac=1.0/(sqrt(4.0*PI)*K*K+1.e-9);
			double fac1 = 3.2*ustar*ustar/pow(Y_loc,2.0/3.0);
			double L = 0.59*Y_loc;
			double fac2 = pow(L,5.0/3.0);
			double fac3 = pow(L,4)*pow(K,4)/pow((1+pow(L,2)*pow(K,2)),17.0/6.0);

			double E=  fac1 * fac2 * fac3;   // alpha * e^2/3 *  L^5/3 *  L^4 k^4 / (1+L^2K^2)^(17/6) 

			double dkx=2*PI/Lx, dky=2*PI/Ly, dkz=2*PI/Lz;
		
			double fac4=sqrt(dkx*dky*dkz);

			double facc = fac4*sqrt(E)*fac;

			C11=0.0; C22=0.0; C33=0.0; 

			C12=facc * Kz[k] ; C13=-facc * Ky[j] ; 
			C21=-facc * Kz[k] ; C23=facc * Kx[i] ; 
			C31=facc * Ky[j] ; C32=-facc * Kx[i] ; 

			Beta=Gamma*(ustar/Karman_const/Y_loc)*pow((K*L+1.e-9),-2.0/3.0);

			double K30=Kz[k]+Beta*Kx[i];
			double K0 = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+K30*K30);

			double C1=Beta*pow(Kx[i],2)*(pow(K0,2)-2*pow(K30,2)+Beta*Kx[i]*K30)/(pow(K,2)*(pow(Kx[i],2)+pow(Ky[j],2))+1.e-9);

			double tmp=atan(Beta*Kx[i]*sqrt(pow(Kx[i],2)+pow(Ky[i],2))/(pow(K0,2)-K30*Kx[i]*Beta+1.e-9));
			double C2=Ky[j]*pow(K0,2)*tmp/(pow(pow(Kx[i],2)+pow(Ky[j],2),3.0/2.0)+1.e-9);

			double zeta1=C1-Ky[j]*C2/(Kx[i]+1.e-9);

			double zeta2=Ky[j]*C1/(Kx[i]+1.e-9)+C2;

			TC11=1.0;
			TC12=0.0;
			TC13=zeta1;

			TC21=0.0;
			TC22=1.0;
			TC23=zeta2;

			TC31=0.0;
			TC32=0.0;
			TC33=pow(K0,2)/(pow(K,2)+1.e-9);

			double C11_tmp=C11, C12_tmp=C12, C13_tmp=C13;
			double C21_tmp=C21, C22_tmp=C22, C23_tmp=C23;
			double C31_tmp=C31, C32_tmp=C32, C33_tmp=C33;

			C11=TC11*C11_tmp+TC12*C21_tmp+TC13*C31_tmp;
			C12=TC11*C12_tmp+TC12*C22_tmp+TC13*C32_tmp;
			C13=TC11*C13_tmp+TC12*C23_tmp+TC13*C33_tmp;

			C21=TC21*C11_tmp+TC22*C21_tmp+TC23*C31_tmp;
			C22=TC21*C12_tmp+TC22*C22_tmp+TC23*C32_tmp;
			C23=TC21*C13_tmp+TC22*C23_tmp+TC23*C33_tmp;

			C31=TC31*C11_tmp+TC32*C21_tmp+TC33*C31_tmp;
			C32=TC31*C12_tmp+TC32*C22_tmp+TC33*C32_tmp;
			C33=TC31*C13_tmp+TC32*C23_tmp+TC33*C33_tmp;

			CKx[i][0]=C11*n_Gauss[k][j][i]+C12*n_Gauss[k][j][i]+C13*n_Gauss[k][j][i];

			CKx[i][1]=0.0;
		}
		fftw_execute(px); 
		for(i=0;i<Nx;i++) {
			U[k][j][i]=CKx_o[i][0];			
			U_i[k][j][i]=CKx_o[i][1];			
		}
	}

	for(i=0;i<Nx;i++)
	for(k=0;k<Nz;k++) {
		for(j=0;j<Ny;j++) {
			CKy[j][0]=U[k][j][i];
			CKy[j][1]=U_i[k][j][i];
		}
		fftw_execute(py); 
		for(j=0;j<Ny;j++) {
			U[k][j][i]=CKy_o[j][0];	
			U_i[k][j][i]=CKy_o[j][1];	
		}
	}

	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		for(k=0;k<Nz;k++) {
			CKz[k][0]=U[k][j][i];
			CKz[k][1]=U_i[k][j][i];
		}
		fftw_execute(pz); 
		for(k=0;k<Nz;k++) {
			U[k][j][i]=CKz_o[k][0];	
			U_i[k][j][i]=CKz_o[k][1];	
		}
	}

	// v
	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) {
		for(i=0;i<Nx;i++) {

			double K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+Kz[k]*Kz[k]);
			double fac=1.0/(sqrt(4.0*PI)*K*K+1.e-9);
			double fac1 = 3.2*ustar*ustar/pow(Y_loc,2.0/3.0);
			double L = 0.59*Y_loc;
			double fac2 = pow(L,5.0/3.0);
			double fac3 = pow(L,4)*pow(K,4)/pow((1+pow(L,2)*pow(K,2)),17.0/6.0);

			double E=  fac1 * fac2 * fac3;   // alpha * e^2/3 *  L^5/3 *  L^4 k^4 / (1+L^2K^2)^(17/6) 

			double dkx=2*PI/Lx, dky=2*PI/Ly, dkz=2*PI/Lz;
		
			double fac4=sqrt(dkx*dky*dkz);

			double facc = fac4*sqrt(E)*fac;

			C11=0.0; C22=0.0; C33=0.0; 

			C12=facc * Kz[k] ; C13=-facc * Ky[j] ; 
			C21=-facc * Kz[k] ; C23=facc * Kx[i] ; 
			C31=facc * Ky[j] ; C32=-facc * Kx[i] ; 

			Beta=Gamma*(ustar/Karman_const/Y_loc)*pow((K*L+1.e-9),-2.0/3.0);

			double K30=Kz[k]+Beta*Kx[i];
			double K0 = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+K30*K30);

			double C1=Beta*pow(Kx[i],2)*(pow(K0,2)-2*pow(K30,2)+Beta*Kx[i]*K30)/(pow(K,2)*(pow(Kx[i],2)+pow(Ky[j],2))+1.e-9);

			double tmp=atan(Beta*Kx[i]*sqrt(pow(Kx[i],2)+pow(Ky[i],2))/(pow(K0,2)-K30*Kx[i]*Beta+1.e-9));
			double C2=Ky[j]*pow(K0,2)*tmp/(pow(pow(Kx[i],2)+pow(Ky[j],2),3.0/2.0)+1.e-9);

			double zeta1=C1-Ky[j]*C2/(Kx[i]+1.e-9);

			double zeta2=Ky[j]*C1/(Kx[i]+1.e-9)+C2;

			TC11=1.0;
			TC12=0.0;
			TC13=zeta1;

			TC21=0.0;
			TC22=1.0;
			TC23=zeta2;

			TC31=0.0;
			TC32=0.0;
			TC33=pow(K0,2)/(pow(K,2)+1.e-9);

			double C11_tmp=C11, C12_tmp=C12, C13_tmp=C13;
			double C21_tmp=C21, C22_tmp=C22, C23_tmp=C23;
			double C31_tmp=C31, C32_tmp=C32, C33_tmp=C33;

			C11=TC11*C11_tmp+TC12*C21_tmp+TC13*C31_tmp;
			C12=TC11*C12_tmp+TC12*C22_tmp+TC13*C32_tmp;
			C13=TC11*C13_tmp+TC12*C23_tmp+TC13*C33_tmp;

			C21=TC21*C11_tmp+TC22*C21_tmp+TC23*C31_tmp;
			C22=TC21*C12_tmp+TC22*C22_tmp+TC23*C32_tmp;
			C23=TC21*C13_tmp+TC22*C23_tmp+TC23*C33_tmp;

			C31=TC31*C11_tmp+TC32*C21_tmp+TC33*C31_tmp;
			C32=TC31*C12_tmp+TC32*C22_tmp+TC33*C32_tmp;
			C33=TC31*C13_tmp+TC32*C23_tmp+TC33*C33_tmp;

			CKx[i][0]=C21*n_Gauss[k][j][i]+C22*n_Gauss[k][j][i]+C23*n_Gauss[k][j][i];
			CKx[i][1]=0.0;
		}
		fftw_execute(px); 
		for(i=0;i<Nx;i++) {
			V[k][j][i]=CKx_o[i][0];	
			V_i[k][j][i]=CKx_o[i][1];	
		}
	}

	for(i=0;i<Nx;i++)
	for(k=0;k<Nz;k++) {
		for(j=0;j<Ny;j++) {
			CKy[j][0]=V[k][j][i];
			CKy[j][1]=V_i[k][j][i];
		}
		fftw_execute(py); 
		for(j=0;j<Ny;j++) {
			V[k][j][i]=CKy_o[j][0];	
			V_i[k][j][i]=CKy_o[j][1];	
		}
	}

	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		for(k=0;k<Nz;k++) {
			CKz[k][0]=V[k][j][i];
			CKz[k][1]=V_i[k][j][i];
		}
		fftw_execute(pz); 
		for(k=0;k<Nz;k++) {
			V[k][j][i]=CKz_o[k][0];	
			V_i[k][j][i]=CKz_o[k][1];	
		}
	}


	// w
	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) {
		for(i=0;i<Nx;i++) {


			double K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+Kz[k]*Kz[k]);
			double fac=1.0/(sqrt(4.0*PI)*K*K+1.e-9);
			double fac1 = 3.2*ustar*ustar/pow(Y_loc,2.0/3.0);
			double L = 0.59*Y_loc;
			double fac2 = pow(L,5.0/3.0);
			double fac3 = pow(L,4)*pow(K,4)/pow((1+pow(L,2)*pow(K,2)),17.0/6.0);

			double E=  fac1 * fac2 * fac3;   // alpha * e^2/3 *  L^5/3 *  L^4 k^4 / (1+L^2K^2)^(17/6) 

			double dkx=2*PI/Lx, dky=2*PI/Ly, dkz=2*PI/Lz;
		
			double fac4=sqrt(dkx*dky*dkz);

			double facc = fac4*sqrt(E)*fac;

			C11=0.0; C22=0.0; C33=0.0; 

			C12=facc * Kz[k] ; C13=-facc * Ky[j] ; 
			C21=-facc * Kz[k] ; C23=facc * Kx[i] ; 
			C31=facc * Ky[j] ; C32=-facc * Kx[i] ; 

			Beta=Gamma*(ustar/Karman_const/Y_loc)*pow((K*L+1.e-9),-2.0/3.0);

			double K30=Kz[k]+Beta*Kx[i];
			double K0 = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+K30*K30);

			double C1=Beta*pow(Kx[i],2)*(pow(K0,2)-2*pow(K30,2)+Beta*Kx[i]*K30)/(pow(K,2)*(pow(Kx[i],2)+pow(Ky[j],2))+1.e-9);

			double tmp=atan(Beta*Kx[i]*sqrt(pow(Kx[i],2)+pow(Ky[i],2))/(pow(K0,2)-K30*Kx[i]*Beta+1.e-9));
			double C2=Ky[j]*pow(K0,2)*tmp/(pow(pow(Kx[i],2)+pow(Ky[j],2),3.0/2.0)+1.e-9);

			double zeta1=C1-Ky[j]*C2/(Kx[i]+1.e-9);

			double zeta2=Ky[j]*C1/(Kx[i]+1.e-9)+C2;

			TC11=1.0;
			TC12=0.0;
			TC13=zeta1;

			TC21=0.0;
			TC22=1.0;
			TC23=zeta2;

			TC31=0.0;
			TC32=0.0;
			TC33=pow(K0,2)/(pow(K,2)+1.e-9);

			double C11_tmp=C11, C12_tmp=C12, C13_tmp=C13;
			double C21_tmp=C21, C22_tmp=C22, C23_tmp=C23;
			double C31_tmp=C31, C32_tmp=C32, C33_tmp=C33;

			C11=TC11*C11_tmp+TC12*C21_tmp+TC13*C31_tmp;
			C12=TC11*C12_tmp+TC12*C22_tmp+TC13*C32_tmp;
			C13=TC11*C13_tmp+TC12*C23_tmp+TC13*C33_tmp;

			C21=TC21*C11_tmp+TC22*C21_tmp+TC23*C31_tmp;
			C22=TC21*C12_tmp+TC22*C22_tmp+TC23*C32_tmp;
			C23=TC21*C13_tmp+TC22*C23_tmp+TC23*C33_tmp;

			C31=TC31*C11_tmp+TC32*C21_tmp+TC33*C31_tmp;
			C32=TC31*C12_tmp+TC32*C22_tmp+TC33*C32_tmp;
			C33=TC31*C13_tmp+TC32*C23_tmp+TC33*C33_tmp;

			CKx[i][0]=C31*n_Gauss[k][j][i]+C32*n_Gauss[k][j][i]+C33*n_Gauss[k][j][i];
			CKx[i][1]=0.0;
		}
		fftw_execute(px); 
		for(i=0;i<Nx;i++) {
			W[k][j][i]=CKx_o[i][0];	
			W_i[k][j][i]=CKx_o[i][1];	
		}
	}

	for(i=0;i<Nx;i++)
	for(k=0;k<Nz;k++) {
		for(j=0;j<Ny;j++) {
			CKy[j][0]=W[k][j][i];
			CKy[j][1]=W_i[k][j][i];
		}
		fftw_execute(py); 
		for(j=0;j<Ny;j++) {
			W[k][j][i]=CKy_o[j][0];	
			W_i[k][j][i]=CKy_o[j][1];	
		}
	}

	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		for(k=0;k<Nz;k++) {
			CKz[k][0]=W[k][j][i];
			CKz[k][1]=W_i[k][j][i];
		}
		fftw_execute(pz); 
		for(k=0;k<Nz;k++) {
			W[k][j][i]=CKz_o[k][0];			
			W_i[k][j][i]=CKz_o[k][1];			
		}
	}


	fftw_destroy_plan(px);
	fftw_destroy_plan(py);
	fftw_destroy_plan(pz);
	fftw_free(CKx); fftw_free(CKx_o);
	fftw_free(CKy); fftw_free(CKy_o);
	fftw_free(CKz); fftw_free(CKz_o);

	char filen[80];
	sprintf(filen, "Turbulence%06d.plt", 0);

	INTEGER4 I, Debug, VIsDouble, DIsDouble, III, IMax, JMax, KMax;
	VIsDouble = 0;
	DIsDouble = 0;
	Debug = 0;

	double SolTime = 0.0;
	INTEGER4 StrandID = 0; /* StaticZone */	
	INTEGER4 ParentZn = 0;	
	INTEGER4 TotalNumFaceNodes = 1; /* Not used for FEQuad zones*/
	INTEGER4 NumConnectedBoundaryFaces = 1; /* Not used for FEQuad zones*/
	INTEGER4 TotalNumBoundaryConnections = 1; /* Not used for FEQuad zones*/
	INTEGER4 FileType = 0;


	
	I = TECINI112((char*)"Flow", (char*)"X Y Z U V W U_i V_i W_i", filen, (char*)".", &FileType, &Debug, &VIsDouble);

	INTEGER4	ZoneType=0, ICellMax=0, JCellMax=0, KCellMax=0;
	INTEGER4	IsBlock=1, NumFaceConnections=0, FaceNeighborMode=0;
	INTEGER4    ShareConnectivityFromZone=0;
	INTEGER4	LOC[40] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}; /* 1 is cell-centered 0 is node centered */
		

	I = TECZNE112((char*)"Zone 1",
                        &ZoneType,      /* Ordered zone */
                        &Nx,
                        &Ny,
                        &Nz,
                        &ICellMax,
                        &JCellMax,
                        &KCellMax,
                        &SolTime,
                        &StrandID,
			&ParentZn,
			&IsBlock,
			&NumFaceConnections,
			&FaceNeighborMode,
			&TotalNumFaceNodes,
			&NumConnectedBoundaryFaces,
			&TotalNumBoundaryConnections,
			NULL,
			LOC,
			NULL,
			&ShareConnectivityFromZone);

	III = Nx*Ny*Nz;
	
	float *x;
	x = new float [III];
		
	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = X[i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = Y[j];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = Z[k];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = U[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = V[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = W[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = U_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = V_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = W_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);



	I = TECEND112();

/*
	int N=32;
	fftw_complex *in, *out;
	fftw_plan p;
//	...
	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	in[0][0]=1;
	in[0][1]=0;
	for(i=1;i<N;i++) {
		in[i][0]=0; //cos(i*2.0*PI/(N-1));
		in[i][1]=0; //-sin(i*2.0*PI/(N-1));
	}	

        FILE *f_fft;
        char fname_fft[80];  
        sprintf(fname_fft, "fft_in.dat");
        f_fft = fopen(fname_fft, "w");
    	for( i = 0 ; i < N ; i++ ) {
        	fprintf( f_fft, "%d %le %le\n", i, in[i][0], in[i][1] );
    	}
	fclose(f_fft);

	fftw_execute(p); // repeat as needed 

        sprintf(fname_fft, "fft_out.dat");
        f_fft = fopen(fname_fft, "w");
        for( i = 0 ; i < N ; i++ ) {
                fprintf( f_fft, "%d %le %le \n", i, out[i][0], out[i][1] );
        }
        fclose(f_fft);


	fftw_destroy_plan(p);
	fftw_free(in); fftw_free(out);

*/

	free(Kx);
	free(Ky);
	free(Kz);
	free(X);
	free(Y);
	free(Z);

	for(k=0;k<Nz;k++) 
	for(j=0;j<Ny;j++) {
		free(U[k][j]);
		free(V[k][j]);
		free(W[k][j]);

		free(U_i[k][j]);
		free(V_i[k][j]);
		free(W_i[k][j]);

		free(n_Gauss[k][j]);
	}


}


double randn_notrig() {


	static bool deviateAvailable=false;   
	static float storedDeviate;
	double polar, rsquared, var1, var2;
	double mu=0.0; double sigma=1.0;
	if (!deviateAvailable) {
		do {
			var1=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
			var2=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
			rsquared=var1*var1+var2*var2;
		} while ( rsquared>=1.0 || rsquared == 0.0);

		polar=sqrt(-2.0*log(rsquared)/rsquared);
		storedDeviate=var1*polar;
		deviateAvailable=true;
		return var2*polar*sigma + mu;
	}
	else {
		deviateAvailable=false;
		return storedDeviate*sigma + mu;
	}
}


