#include "variables.h"

extern	PetscInt ti, tiout;

PetscErrorCode preprocess_ib(UserCtx *user, IBMNodes *ibm)
{
  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo     info;
  	int        xs, xe, ys, ye, zs, ze; // Local grid information
  	int        mx, my, mz; // Dimensions in three directions
  	int        i, j, k, l, ibi;
  	int	   lxs, lxe, lys, lye, lzs, lze;
  	double	   xb_min, xb_max, yb_min, yb_max, zb_min, zb_max;

  	Cmpnts		***coor;

  	Vec		Coor;

  	DMDAGetLocalInfo(da, &info);
  	mx = info.mx; my = info.my; mz = info.mz;
  	xs = info.xs; xe = xs + info.xm;
  	ys = info.ys; ye = ys + info.ym;
  	zs = info.zs; ze = zs + info.zm;


  	lxs = xs; lxe = xe;
  	lys = ys; lye = ye;
  	lzs = zs; lze = ze;

  	if (xs==0) lxs = xs+1;
  	if (ys==0) lys = ys+1;
  	if (zs==0) lzs = zs+1;

  	if (xe==mx) lxe = xe-1;
  	if (ye==my) lye = ye-1;
  	if (ze==mz) lze = ze-1;


  	DMDAGetGhostedCoordinates(da, &Coor);
  	DMDAVecGetArray(fda, Coor, &coor);

	Cmpnts		***cent;
        DMDAVecGetArray(fda, user->lCent,  &cent);

	double 	hmax;	


  	for (ibi=0; ibi<NumberOfBodies; ibi++) {

		double *count, *count_sum;
        	double *u, *u_sum;
		u = (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		u_sum = (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		count = (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		count_sum = (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		int *ib, *ib_sum;
		ib = (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		ib_sum = (int *) malloc(ibm[ibi].n_elmt*sizeof(int));


	    	for (l=0; l<ibm[ibi].n_elmt; l++) {
        	        ibm[ibi].xb_elmt[l] = 0.0; 
	                ibm[ibi].yb_elmt[l] = 0.0; 
	                ibm[ibi].zb_elmt[l] = 0.0; 
			count[l] = 0.0;		
			ibm[ibi].count[l]=0;
			for (k=lzs; k<lze; k++) 
			for (j=lys; j<lye; j++) 
			for (i=lxs; i<lxe; i++){

				if ( ibm[ibi].cent_x[l] >= cent[k][j][i-1].x && ibm[ibi].cent_x[l] < cent[k][j][i].x && 
				     ibm[ibi].cent_y[l] >= cent[k][j-1][i].y && ibm[ibi].cent_y[l] < cent[k][j][i].y &&
				     ibm[ibi].cent_z[l] >= cent[k-1][j][i].z && ibm[ibi].cent_z[l] < cent[k][j][i].z ){
					hmax = max(fabs(cent[k][j][i].x- cent[k][j][i-1].x), fabs(cent[k][j][i].y- cent[k][j-1][i].y));
					hmax = max(hmax, fabs(cent[k][j][i].z- cent[k-1][j][i].z));

					ibm[ibi].xb_elmt[l] = ibm[ibi].cent_x[l] + 2.0*hmax*ibm[ibi].nf_x[l]; 
					ibm[ibi].yb_elmt[l] = ibm[ibi].cent_y[l] + 2.0*hmax*ibm[ibi].nf_y[l];
					ibm[ibi].zb_elmt[l] = ibm[ibi].cent_z[l] + 2.0*hmax*ibm[ibi].nf_z[l];	
				 	count[l] = 1.0;
				}
			} 
	    	}


        	for (l=0; l<ibm[ibi].n_elmt; l++) u[l]=ibm[ibi].xb_elmt[l];
		MPI_Allreduce (&u[0], &u_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&count[0], &count_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
        	for (l=0; l<ibm[ibi].n_elmt; l++) {
			if (count_sum[0]>0.9) {
				ibm[ibi].xb_elmt[l]=u_sum[l]/count_sum[l];
				ibm[ibi].count[l]=1;
			}
		}


        	for (l=0; l<ibm[ibi].n_elmt; l++) u[l]=ibm[ibi].yb_elmt[l];
		MPI_Allreduce (&u[0], &u_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&count[0], &count_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
                for (l=0; l<ibm[ibi].n_elmt; l++) {
                        if (count_sum[0]>0.9) {
                                ibm[ibi].yb_elmt[l]=u_sum[l]/count_sum[l];
                                ibm[ibi].count[l]=1;
                        }
                }


        	for (l=0; l<ibm[ibi].n_elmt; l++) u[l]=ibm[ibi].zb_elmt[l];
		MPI_Allreduce (&u[0], &u_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&count[0], &count_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
                for (l=0; l<ibm[ibi].n_elmt; l++) {
                        if (count_sum[0]>0.9) {
                                ibm[ibi].zb_elmt[l]=u_sum[l]/count_sum[l];
                                ibm[ibi].count[l]=1;
                        }
                }


	        for (l=0; l<ibm[ibi].n_elmt; l++) {

        	        ibm[ibi].ib_elmt[l] = -100;
	                ibm[ibi].jb_elmt[l] = -100;
	                ibm[ibi].kb_elmt[l] = -100;
			count[l] = 0;

	                for (k=lzs; k<lze; k++)
	                for (j=lys; j<lye; j++)
	                for (i=lxs; i<lxe; i++){
        	                if ( ibm[ibi].xb_elmt[l] >= cent[k][j][i-1].x && ibm[ibi].xb_elmt[l] < cent[k][j][i].x &&
                	             ibm[ibi].yb_elmt[l] >= cent[k][j-1][i].y && ibm[ibi].yb_elmt[l] < cent[k][j][i].y &&
                        	     ibm[ibi].zb_elmt[l] >= cent[k-1][j][i].z && ibm[ibi].zb_elmt[l] < cent[k][j][i].z ){

	                                ibm[ibi].ib_elmt[l] = i-1;
	                                ibm[ibi].jb_elmt[l] = j-1;
	                                ibm[ibi].kb_elmt[l] = k-1;
					count[l] = 1;

	                        }
        	        }
	        }
 

        	for (l=0; l<ibm[ibi].n_elmt; l++) ib[l]=ibm[ibi].ib_elmt[l];
		MPI_Allreduce (&ib[0], &ib_sum[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&count[0], &count_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
        	for (l=0; l<ibm[ibi].n_elmt; l++) {
			if (count_sum[0]>0.9) {
				ibm[ibi].ib_elmt[l]=ib_sum[l]/(int)count_sum[l];
			}
		}


        	for (l=0; l<ibm[ibi].n_elmt; l++) ib[l]=ibm[ibi].jb_elmt[l];
		MPI_Allreduce (&ib[0], &ib_sum[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&count[0], &count_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
        	for (l=0; l<ibm[ibi].n_elmt; l++) {
			if (count_sum[0]>0.9) {
				ibm[ibi].jb_elmt[l]=ib_sum[l]/(int)count_sum[l];
			}
		}

        	for (l=0; l<ibm[ibi].n_elmt; l++) ib[l]=ibm[ibi].kb_elmt[l];
		MPI_Allreduce (&ib[0], &ib_sum[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&count[0], &count_sum[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
        	for (l=0; l<ibm[ibi].n_elmt; l++) {
			if (count_sum[0]>0.9) {
				ibm[ibi].kb_elmt[l]=ib_sum[l]/(int)count_sum[l];
			}
		}


		free(u);
		free(u_sum);
		free(count);
		free(count_sum);
		free(ib);
		free(ib_sum);


	}



        DAVecRestoreArray(fda, Coor, &coor);
        DAVecRestoreArray(fda, user->lCent,  &cent);

	return(0);
}










