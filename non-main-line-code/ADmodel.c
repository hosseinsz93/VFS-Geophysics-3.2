/*Modeling the rotor blades */

#include "variables.h"

#include <algorithm>
using namespace std;

extern	double dfunc_2h(double r);
extern	double dfunc_4h(double r);
extern  double dfunc_s3h(double r);
extern  double dfunc_s4h(double r);

extern	PetscErrorCode Calc_U_lagr(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects); 

extern Cmpnts ArbitraryRotate(Cmpnts p,double theta,Cmpnts r);
double  coef_cr1 = 2.5, coef_cr2 = 1.0;

int Itpwidth=4;


PetscErrorCode disk_read_ucd(IBMNodes *ibm, int ibi, FSInfo *fsi, int OneDmZ, char fname[80])
{
  	int	rank;
  	int	n_v , n_elmt ;
  	PetscReal	*x_bp , *y_bp , *z_bp ;
  	int	*nv1 , *nv2 , *nv3 ;
  	PetscReal	*nf_x, *nf_y, *nf_z;
  	int	i,ii;
  	int	n1e, n2e, n3e;
  	PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  	PetscReal     dr;
  	PetscReal     *dA ;//area
  	PetscReal	*nt_x, *nt_y, *nt_z;
  	PetscReal	*ns_x, *ns_y, *ns_z;

  	char   ss[20];
  	char string[128];

  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  	if(!rank) { // root processor read in the data
    		FILE *fd;
    		PetscPrintf(PETSC_COMM_SELF, "READ %s\n", fname);
    		char filen[160];  
    		sprintf(filen,"%s/%s" ,path, fname);
 
    		fd = fopen(filen, "r"); 
    		if (!fd) printf("Cannot open %s !!\n", filen),exit(0);
    		else printf("Opened %s !\n", filen);
    		n_v =0;

    		if (fd) {
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      
      			fscanf(fd, "%i %i %i %i %i",&n_v,&n_elmt,&ii,&ii,&ii);
      			PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements %d %d\n",n_v, n_elmt);
      
      			ibm->n_v = n_v;
      			ibm->n_elmt = n_elmt;      
      
      			MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
      
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
      			x_bp = ibm->x_bp;	// seokkoo
      			y_bp = ibm->y_bp;
      			z_bp = ibm->z_bp;

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
      
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
 
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

      			for (i=0; i<n_v; i++) {
				fscanf(fd, "%i %le %le %le", &ii, &x_bp[i], &y_bp[i], &z_bp[i]);//, &t, &t, &t);
	
				x_bp[i]=x_bp[i]/reflength_wt;
				y_bp[i]=y_bp[i]/reflength_wt;
				z_bp[i]=z_bp[i]/reflength_wt;

				// rotate the axis. Assume the one from gridgen is (0,0,1)
				/*
				double n_rot[3];
				n_rot[0]=-fsi->ny_tb; n_rot[1]=fsi->nx_tb; n_rot[2]=0.0;
				double ux=n_rot[0], uy=n_rot[1], uz=n_rot[2];

				double ang_rot;
				double rrr=fsi->nz_tb;
				ang_rot=acos(rrr);
				double cos_=cos(ang_rot), sin_=sin(ang_rot);
				*/

				Cmpnts p, r, q;
				p.x=x_bp[i]; p.y=y_bp[i]; p.z=z_bp[i];
				r.x=-fsi->ny_tb; r.y=fsi->nx_tb; r.z=0.0;
				double theta=acos(fsi->nz_tb);
				q=ArbitraryRotate(p,theta,r);
				x_bp[i]=q.x;	
				y_bp[i]=q.y;	
				z_bp[i]=q.z;	
	
				/*
				double mat_rot[3][3];

				mat_rot[0][0]=cos_+pow(ux,2)*(1.0-cos_);
				mat_rot[0][1]=ux*uy*(1.0-cos_)-uz*sin_;
				mat_rot[0][2]=ux*uz*(1.0-cos_)+uy*sin_;

				mat_rot[1][0]=uy*ux*(1.0-cos_)+uz*sin_;
				mat_rot[1][1]=cos_+pow(uy,2)*(1.0-cos_);
				mat_rot[1][2]=uy*uz*(1.0-cos_)-ux*sin_;

				mat_rot[2][0]=uz*ux*(1.0-cos_)-uy*sin_;
				mat_rot[2][1]=uz*uy*(1.0-cos_)+ux*sin_;
				mat_rot[2][2]=cos_+pow(uz,2)*(1.0-cos_);
				*/

				// double xb=x_bp[i], yb=y_bp[i], zb=z_bp[i];

				/*
				x_bp[i]=mat_rot[0][0]*xb+mat_rot[0][1]*yb+mat_rot[0][2]*zb;
				y_bp[i]=mat_rot[1][0]*xb+mat_rot[1][1]*yb+mat_rot[1][2]*zb;
				z_bp[i]=mat_rot[2][0]*xb+mat_rot[2][1]*yb+mat_rot[2][2]*zb;
				*/

				/*
				double wCv[3];
				wCv[0]=uy*zb-uz*yb; wCv[1]=uz*xb-ux*zb; wCv[2]=ux*yb-uy*xb;
				double wPv=ux*xb+uy*yb+uz*zb;
				x_bp[i]=xb*cos_+wCv[0]*sin_+wPv*ux*(1.0-cos_);
				y_bp[i]=yb*cos_+wCv[1]*sin_+wPv*uy*(1.0-cos_);
				z_bp[i]=zb*cos_+wCv[2]*sin_+wPv*uz*(1.0-cos_);
				*/

			        ibm->x_bp_i[i] = x_bp[i];
			        ibm->y_bp_i[i] = y_bp[i];
			        ibm->z_bp_i[i] = z_bp[i];

			        x_bp[i] += fsi->x_c;
			        y_bp[i] += fsi->y_c;
			        z_bp[i] += fsi->z_c;

			        ibm->x_bp0[i] = x_bp[i];
			        ibm->y_bp0[i] = y_bp[i];
			        ibm->z_bp0[i] = z_bp[i];


				if (OneDmZ) {
					double rr = 2.0*r_rotor/reflength_wt;
					x_bp[i]-=rr*fsi->nx_tb; 
					y_bp[i]-=rr*fsi->ny_tb; 
					z_bp[i]-=rr*fsi->nz_tb; 
				}
	
				ibm->x_bp[i] = x_bp[i];
				ibm->y_bp[i] = y_bp[i];
				ibm->z_bp[i] = z_bp[i];

				ibm->x_bp_o[i] = x_bp[i];
				ibm->y_bp_o[i] = y_bp[i];
				ibm->z_bp_o[i] = z_bp[i];

				ibm->u[i].x = 0.;
				ibm->u[i].y = 0.;
				ibm->u[i].z = 0.;

				ibm->uold[i].x = 0.;
				ibm->uold[i].y = 0.;
				ibm->uold[i].z = 0.;

				ibm->urm1[i].x = 0.;
				ibm->urm1[i].y = 0.;
				ibm->urm1[i].z = 0.;
			}
      			i=0;
		      	PetscPrintf(PETSC_COMM_WORLD, "xyz_bp %le %le %le\n", x_bp[i], y_bp[i], z_bp[i]);

      			MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
		      	MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

      			PetscMalloc(n_elmt*sizeof(int), &nv1);
      			PetscMalloc(n_elmt*sizeof(int), &nv2);
      			PetscMalloc(n_elmt*sizeof(int), &nv3);
      
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      
      			PetscMalloc(n_elmt*sizeof(PetscReal), &dA); //Area

      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
      
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));

			// rotor model

        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

		        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
		        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
		        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));


		        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
		        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
		        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

		        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
		        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
		        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
       		        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
                	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));


        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_attack));
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_pitch));
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->chord_blade));

                       	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CD));
                       	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CL));


      			for (i=0; i<n_elmt; i++) {
				fscanf(fd, "%i %i %s %i %i %i\n", &ii,&ii, &ss, nv1+i, nv2+i, nv3+i);
				nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
		      	}
		        ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

		        i=0;
		        PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", nv1[i], nv2[i], nv3[i]);

		        fclose(fd);
 	      	}
      
	      	for (i=0; i<n_elmt; i++) {
      
			n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
		        dx12 = x_bp[n2e] - x_bp[n1e];
		        dy12 = y_bp[n2e] - y_bp[n1e];
		        dz12 = z_bp[n2e] - z_bp[n1e];
      
		        dx13 = x_bp[n3e] - x_bp[n1e];
		        dy13 = y_bp[n3e] - y_bp[n1e];
		        dz13 = z_bp[n3e] - z_bp[n1e];
      
		        nf_x[i] = dy12 * dz13 - dz12 * dy13;
		        nf_y[i] = -dx12 * dz13 + dz12 * dx13;
		        nf_z[i] = dx12 * dy13 - dy12 * dx13;
      
		        dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
		        nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
      
		      	if ((((1.-nf_z[i])<=1e-6 )&((-1.+nf_z[i])<1e-6))|
			  (((nf_z[i]+1.)<=1e-6 )&((-1.-nf_z[i])<1e-6))) {
				ns_x[i] = 1.;     
				ns_y[i] = 0.;     
				ns_z[i] = 0. ;
	
				nt_x[i] = 0.;
				nt_y[i] = 1.;
				nt_z[i] = 0.;
		        } else {
				ns_x[i] =  nf_y[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);      
				ns_y[i] = -nf_x[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);     
				ns_z[i] = 0. ;
	
				nt_x[i] = -nf_x[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
				nt_y[i] = -nf_y[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
				nt_z[i] = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
		        }
      
		        dA[i] = dr/2.; 
      
      			// Calc the center of the element
      			ibm->cent_x[i]= (x_bp[n1e]+x_bp[n2e]+x_bp[n3e])/3.;
		        ibm->cent_y[i]= (y_bp[n1e]+y_bp[n2e]+y_bp[n3e])/3.;
		        ibm->cent_z[i]= (z_bp[n1e]+z_bp[n2e]+z_bp[n3e])/3.;	
		}
    
    
    		ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;
    
    		ibm->dA = dA;
    		ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
	        ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    
    
	        MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
    	        MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
    		MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		int ti=0;
    		FILE *f;
    		sprintf(filen, "%s/%s_%2.2d_nf.dat",path,fname,ibi);
    		f = fopen(filen, "w");
    		PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z,nt_x,nt_y,nt_z,ns_x,ns_y,ns_z\n");
    		PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)\n", n_v, n_elmt);
    		for (i=0; i<n_v; i++) {
      			PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
    		}
    		for (i=0; i<n_v; i++) {
   	   		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
	    	}
    		for (i=0; i<n_v; i++) {	
	  	      	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
    		}
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
	 	        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_x[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
	   	        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_y[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
	 	        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_z[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
	 	        PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
	        }
    
    		fclose(f);

  	}
  	else if (rank) {
    		MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	        ibm->n_v = n_v;
    
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
	  
		x_bp = ibm->x_bp;	// added by seokkoo 03.04.2009
		y_bp = ibm->y_bp;
		z_bp = ibm->z_bp;
	  
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
    
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));
 
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
    
    
	        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
	        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
	        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

	        for (i=0; i<n_v; i++) {
	  		ibm->u[i].x = 0.;
  		        ibm->u[i].y = 0.;
	    	        ibm->u[i].z = 0.;

		        ibm->uold[i].x = 0.;
		        ibm->uold[i].y = 0.;
	   	        ibm->uold[i].z = 0.;
      
	  	        ibm->urm1[i].x = 0.;
		        ibm->urm1[i].y = 0.;
	 	        ibm->urm1[i].z = 0.;      
	        }

	        MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
        
	        MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);    
	        MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

	        MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

 	        MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	       	MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	       	MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    

	       	MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	       	n_elmt = ibm->n_elmt;

	       	PetscMalloc(n_elmt*sizeof(int), &nv1);
	       	PetscMalloc(n_elmt*sizeof(int), &nv2);
	       	PetscMalloc(n_elmt*sizeof(int), &nv3);

	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

	       	PetscMalloc(n_elmt*sizeof(PetscReal), &dA);

   	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

	       	PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);

	       	ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
	       	ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;
    
	       	ibm->dA = dA;
	       	ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
	       	ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    

	       	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));

	       	// rotor model
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

	        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
	        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
	        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));

	        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
	        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
	        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

	        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
	        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
	        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
          	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
              	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));

        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_attack));
        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_pitch));
        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->chord_blade));

                PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CD));
                PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CL));


	    	MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

	        MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	        MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

	        MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	        MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

	        MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	}
	PetscPrintf(PETSC_COMM_WORLD, "Read %s !\n", fname);

  	return(0);
}



// Call in main.c, after reading the grid for turbines
PetscErrorCode Pre_process(UserCtx *user, IBMNodes *ibm, int NumberOfObjects)
{
  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo  	info;
  	PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  	PetscInt        mx, my, mz; // Dimensions in three directions
  	PetscInt        i, j, k, l, ibi;
  	PetscInt	lxs, lxe, lys, lye, lzs, lze;

  	Cmpnts		***coor, ***csi, ***eta, ***zet;

  	PetscReal 	***aj;

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


	double *dhx_, *dhy_, *dhz_;
	int *iclose, *jclose, *kclose;

	double *sum_dhx_, *sum_dhy_, *sum_dhz_;
	int *sum_iclose, *sum_jclose, *sum_kclose;


  	DMDAGetGhostedCoordinates(da, &Coor);
  	DMDAVecGetArray(fda, Coor, &coor);

  	DMDAVecGetArray(fda, user->lCsi,  &csi);
  	DMDAVecGetArray(fda, user->lEta,  &eta);
  	DMDAVecGetArray(fda, user->lZet,  &zet);
  	DMDAVecGetArray(da,  user->lAj,  &aj);


  	int n1e, n2e, n3e;

  	for (ibi=0; ibi<NumberOfObjects; ibi++) {

		dhx_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		dhy_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		dhz_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));

		iclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		jclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		kclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));

		sum_dhx_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		sum_dhy_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		sum_dhz_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));

		sum_iclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		sum_jclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		sum_kclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));

	        for (l=0; l<ibm[ibi].n_elmt; l++) {

	               	dhx_[l] = 0.0;
	                dhy_[l] = 0.0;
	                dhz_[l] = 0.0;

	               	sum_dhx_[l] = 0.0;
        	        sum_dhy_[l] = 0.0;
                	sum_dhz_[l] = 0.0;

			iclose[l]=0;
			jclose[l]=0;
			kclose[l]=0;

			sum_iclose[l]=0;
			sum_jclose[l]=0;
			sum_kclose[l]=0;

		}


	        for (l=0; l<ibm[ibi].n_elmt; l++) {
			int imark=0;
			double dmin=1.e6;
			int ic,jc,kc;
	                for (i=lxs; i<lxe; i++)
        	        for (j=lys; j<lye; j++) 
                	for (k=lzs; k<lze; k++){ 
			
				double r1=ibm[ibi].cent_x[l]-coor[k][j][i].x, r2=ibm[ibi].cent_y[l]-coor[k][j][i].y, r3=ibm[ibi].cent_z[l]-coor[k][j][i].z; 
				double d1=sqrt(r1*r1+r2*r2+r3*r3);
				if (d1<dmin) {
					dmin=d1;
					ic=i; jc=j; kc=k;
				}
                	}
	
			double dmin_global;
			MPI_Allreduce (&dmin, &dmin_global, 1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);
			double diff=fabs(dmin-dmin_global);
			if (diff>1.e-6) {
				ic=0; jc=0; kc=0;
				iclose[l]=0; jclose[l]=0; kclose[l]=0;
				dhx_[l]=0.0;
				dhy_[l]=0.0;
				dhz_[l]=0.0;
			} else {

				iclose[l]=ic; jclose[l]=jc; kclose[l]=kc;
				i=ic; j=jc; k=kc;
		                double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
				dhx_[l]=1.0/aj[k][j][i]/area;

			       	area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
				dhy_[l]=1.0/aj[k][j][i]/area;

			       	area = sqrt( zet[k][j][i].x*zet[k][j][i].x + zet[k][j][i].y*zet[k][j][i].y + zet[k][j][i].z*zet[k][j][i].z );
				dhz_[l]=1.0/aj[k][j][i]/area;	
			}

	        }



	  	PetscBarrier(PETSC_NULL);

		MPI_Allreduce (&dhx_[0], &sum_dhx_[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&dhy_[0], &sum_dhy_[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&dhz_[0], &sum_dhz_[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

		MPI_Allreduce (&iclose[0], &sum_iclose[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&jclose[0], &sum_jclose[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&kclose[0], &sum_kclose[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);


	        for (l=0; l<ibm[ibi].n_elmt; l++) {

        	       	ibm[ibi].dhx[l]=sum_dhx_[l];
	               	ibm[ibi].dhy[l]=sum_dhy_[l];
        	       	ibm[ibi].dhz[l]=sum_dhz_[l];

	               	dhx_[l]=sum_dhx_[l];
	               	dhy_[l]=sum_dhy_[l];
	               	dhz_[l]=sum_dhz_[l];

			iclose[l]=sum_iclose[l];
			jclose[l]=sum_jclose[l];
			kclose[l]=sum_kclose[l];

		}


	        int iii1 = 0;
		int iii2 = 0;
	        int iii = 0;

	        for (l=0; l<ibm[ibi].n_elmt; l++) {

			int ic=iclose[l];
			int jc=jclose[l];
			int kc=kclose[l];

			int ic1=ic-Itpwidth, ic2=ic+Itpwidth; 
		
			if (ic1>lxe||ic2<lxs) {
				ibm[ibi].i_min[l]=lxs;
				ibm[ibi].i_max[l]=lxs;
			} else {
				ibm[ibi].i_min[l]=PetscMax(ic1, lxs);
				ibm[ibi].i_max[l]=PetscMin(ic2, lxe);
			}

			int jc1=jc-Itpwidth, jc2=jc+Itpwidth; 

	       		if (jc1>lye||jc2<lys) {
				ibm[ibi].j_min[l]=lys;
				ibm[ibi].j_max[l]=lys;
			} else {
				ibm[ibi].j_min[l]=PetscMax(jc1, lys);
				ibm[ibi].j_max[l]=PetscMin(jc2, lye);
			}

			int kc1=kc-Itpwidth, kc2=kc+Itpwidth; 

	       		if (kc1>lze||kc2<lzs) {
				ibm[ibi].k_min[l]=lzs;
				ibm[ibi].k_max[l]=lzs;
			} else {
				ibm[ibi].k_min[l]=PetscMax(kc1, lzs);
				ibm[ibi].k_max[l]=PetscMin(kc2, lze);
			}


		}

/*
	        for (l=0; l<ibm[ibi].n_elmt; l++) {

			ibm[ibi].i_min[l]=lxs;
			ibm[ibi].i_max[l]=lxe;

			ibm[ibi].j_min[l]=lys;
			ibm[ibi].j_max[l]=lye;


			ibm[ibi].k_min[l]=lzs;
			ibm[ibi].k_max[l]=lze;


		}	
*/

		if (rotor_model == 2) {

		int imin_g=1000000, imax_g=-1000000, jmin_g=1000000, jmax_g=-1000000, kmin_g=1000000, kmax_g=-1000000;

	        for (l=0; l<ibm[ibi].n_elmt; l++) {
			if (imin_g>ibm[ibi].i_min[l]) imin_g=ibm[ibi].i_min[l];	
			if (imax_g<ibm[ibi].i_max[l]) imax_g=ibm[ibi].i_max[l];	

			if (jmin_g>ibm[ibi].j_min[l]) jmin_g=ibm[ibi].j_min[l];	
			if (jmax_g<ibm[ibi].j_max[l]) jmax_g=ibm[ibi].j_max[l];	

			if (kmin_g>ibm[ibi].k_min[l]) kmin_g=ibm[ibi].k_min[l];	
			if (kmax_g<ibm[ibi].k_max[l]) kmax_g=ibm[ibi].k_max[l];	
		}

	        for (l=0; l<ibm[ibi].n_elmt; l++) {

			ibm[ibi].i_min[l]=imin_g;	
			ibm[ibi].j_min[l]=jmin_g;	
			ibm[ibi].k_min[l]=kmin_g;	

			ibm[ibi].i_max[l]=imax_g;	
			ibm[ibi].j_max[l]=jmax_g;	
			ibm[ibi].k_max[l]=kmax_g;	

		}	

		}

		free(dhx_);
		free(dhy_); 
		free(dhz_);
	        free(iclose);
		free(jclose);
		free(kclose);

		free(sum_dhx_);
		free(sum_dhy_); 
		free(sum_dhz_);
	        free(sum_iclose);
		free(sum_jclose);
		free(sum_kclose);

	}

  	DMDAVecRestoreArray(fda, Coor, &coor);

	DMDAVecRestoreArray(fda, user->lCsi,  &csi);
  	DMDAVecRestoreArray(fda, user->lEta,  &eta);
  	DMDAVecRestoreArray(fda, user->lZet,  &zet);
  	DMDAVecRestoreArray(da,  user->lAj,  &aj);

  	return(0);

}





// calculate the force on the actuator disk, called in the solvers.c beforce solving the momentum equation
PetscErrorCode Calc_F_lagr(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{

  	PetscInt	l, ibi;
  	double	pi = 3.141592653589793, a = 0.25;
  	double 	C_T;  // = 4.0 / 3.0;
  	double 	U_ref, A_sum;
  	PetscReal	sign;
  	double 	Uref_x, Uref_y, Uref_z;


  	Calc_U_lagr(user, ibm, fsi, NumberOfObjects);
  
  	for (ibi=0; ibi<NumberOfObjects; ibi++) {
		
		double indf_ax=ibm[ibi].indf_axis;
  
	  	C_T = 4.0 * indf_ax * (1-indf_ax);

	  	C_T = C_T / ( (1.0 - indf_ax)* (1.0 - indf_ax) );


    		U_ref = 0.0;
    		A_sum = 0.0;
    		Uref_x = 0.0;
    		Uref_y = 0.0;
    		Uref_z = 0.0;

    		for (l=0; l<ibm[ibi].n_elmt; l++) {
      			Uref_x += ibm[ibi].U_lagr_x[l] * ibm[ibi].dA[l];
      			Uref_y += ibm[ibi].U_lagr_y[l] * ibm[ibi].dA[l];
      			Uref_z += ibm[ibi].U_lagr_z[l] * ibm[ibi].dA[l];
      			A_sum += ibm[ibi].dA[l];
    		}
    		Uref_x /= A_sum;
    		Uref_y /= A_sum;
    		Uref_z /= A_sum;

    		U_ref = Uref_x*fsi[ibi].nx_tb+Uref_y*fsi[ibi].ny_tb+Uref_z*fsi[ibi].nz_tb;

    		PetscPrintf(PETSC_COMM_WORLD, "**** The U_ref at %i th body: %le\n", ibi, U_ref);

    		sign = U_ref / fabs(U_ref+1.e-9);
    		for (l=0; l<ibm[ibi].n_elmt; l++) {
      			ibm[ibi].F_lagr_x[l] = -0.5 * C_T * (U_ref * U_ref) * sign * fsi[ibi].nx_tb;
      			ibm[ibi].F_lagr_y[l] = -0.5 * C_T * (U_ref * U_ref) * sign * fsi[ibi].ny_tb;
      			ibm[ibi].F_lagr_z[l] = -0.5 * C_T * (U_ref * U_ref) * sign * fsi[ibi].nz_tb;  
    		}


 	}

 	return(0);

}


/* Interpolate the velocity at the Lagrangian points */
// subroutine for Calc_F_lagr_ACL and Calc_F_lagr
PetscErrorCode Calc_U_lagr(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{

  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo     info;
  	PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  	PetscInt        mx, my, mz; // Dimensions in three directions
  	PetscInt        i, j, k, l, ibi;
  	PetscInt	lxs, lxe, lys, lye, lzs, lze;

  	Cmpnts	***ucat, ***coor, ***csi, ***eta, ***zet;

  	PetscReal 	***aj;

  	Vec		Coor;

  	PetscReal	dfunc;

  	PetscReal	xc, yc, zc, hx, hy, hz, vol_eul, vol_lagr;

  	double r1, r2, r3;

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
  	DMDAVecGetArray(fda, user->lUcat, &ucat);
  	DMDAVecGetArray(fda, user->lCsi,  &csi);
  	DMDAVecGetArray(fda, user->lEta,  &eta);
  	DMDAVecGetArray(fda, user->lZet,  &zet);
  	DMDAVecGetArray(da,  user->lAj,  &aj);

  	for (ibi=0; ibi<NumberOfObjects; ibi++) {
    	for (l=0; l<ibm[ibi].n_elmt; l++) {
      		ibm[ibi].U_lagr_x[l] = 0.0;
      		ibm[ibi].U_lagr_y[l] = 0.0;
      		ibm[ibi].U_lagr_z[l] = 0.0;
    	}
  	}


        clock_t start, end;
        double elapsed;

        start = clock();
	double ni[3], nj[3], nk[3];

	for (ibi=0; ibi<NumberOfObjects; ibi++) {
  	for (l=0; l<ibm[ibi].n_elmt; l++) {

		double dhx_=ibm[ibi].dhx[l];
		double dhy_=ibm[ibi].dhy[l];
		double dhz_=ibm[ibi].dhz[l];

	    	for (k = ibm[ibi].k_min[l]; k<ibm[ibi].k_max[l]; k++) 
	      	for (j = ibm[ibi].j_min[l]; j<ibm[ibi].j_max[l]; j++) 
	        for (i = ibm[ibi].i_min[l]; i<ibm[ibi].i_max[l]; i++) {

			xc=coor[k][j][i].x;
			yc=coor[k][j][i].y;
			zc=coor[k][j][i].z;

	            	double rx= (xc - ibm[ibi].cent_x[l]), ry = (yc - ibm[ibi].cent_y[l]), rz = (zc - ibm[ibi].cent_z[l]);

			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);

			r1=(rx*ni[0]+ry*ni[1]+rz*ni[2])/dhx_; 
			r2=(rx*nj[0]+ry*nj[1]+rz*nj[2])/dhy_; 
			r3=(rx*nk[0]+ry*nk[1]+rz*nk[2])/dhz_; 

			if (deltafunc == 1) dfunc =  dfunc_2h(r1) * dfunc_2h(r2) * dfunc_2h(r3);
			if (deltafunc == 2) dfunc =  dfunc_4h(r1) * dfunc_4h(r2) * dfunc_4h(r3);
	                if (deltafunc == 3) dfunc =  dfunc_s3h(r1) * dfunc_s3h(r2) * dfunc_s3h(r3);
	                if (deltafunc == 4) dfunc =  dfunc_s4h(r1) * dfunc_s4h(r2) * dfunc_s4h(r3);

	            	ibm[ibi].U_lagr_x[l] += ucat[k][j][i].x * dfunc;
	            	ibm[ibi].U_lagr_y[l] += ucat[k][j][i].y * dfunc;
	            	ibm[ibi].U_lagr_z[l] += ucat[k][j][i].z * dfunc;          
   
      		}

  	}
  	}

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

        PetscPrintf(PETSC_COMM_WORLD, "Time for U_larg local %le \n", elapsed);


        start = clock();

  	PetscBarrier(PETSC_NULL);

	// the n_elmt for each turbine is assumed to be the same
  	int nelmt_Max=-1000;
  	for (ibi=0; ibi<NumberOfObjects; ibi++)  {
		if (ibm[ibi].n_elmt>nelmt_Max) nelmt_Max=ibm[ibi].n_elmt;	
  	}

  	double    u_local[NumberOfObjects][nelmt_Max] ,u_sum[NumberOfObjects][nelmt_Max];
  	int totNum = NumberOfObjects*nelmt_Max;

	// x
  	for (ibi=0; ibi<NumberOfObjects; ibi++)  
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
		u_local[ibi][i] = ibm[ibi].U_lagr_x[i];
  	}

  	MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  	for (ibi=0; ibi<NumberOfObjects; ibi++) 
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	ibm[ibi].U_lagr_x[i] = u_sum[ibi][i];
  	}

  	PetscBarrier(PETSC_NULL);
	// y
  	for (ibi=0; ibi<NumberOfObjects; ibi++) 
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
  	      u_local[ibi][i] = ibm[ibi].U_lagr_y[i];
  	}

  	MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  	for (ibi=0; ibi<NumberOfObjects; ibi++)
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	ibm[ibi].U_lagr_y[i] = u_sum[ibi][i];
  	}

  	PetscBarrier(PETSC_NULL);
// z
  	for (ibi=0; ibi<NumberOfObjects; ibi++) 
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	u_local[ibi][i] = ibm[ibi].U_lagr_z[i];
  	}

  	MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  	for (ibi=0; ibi<NumberOfObjects; ibi++)
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	ibm[ibi].U_lagr_z[i] = u_sum[ibi][i];
  	}


        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

        PetscPrintf(PETSC_COMM_WORLD, "Time for U_larg global sum %le \n", elapsed);


  	if (ii_periodicWT || jj_periodicWT || kk_periodicWT) {

    		double xc_min = 1.0e6;        
 	   	double yc_min = 1.0e6;
	    	double zc_min = 1.0e6;

  		for (ibi=0; ibi<NumberOfObjects; ibi++) {
      			xc_min = PetscMin(xc_min, fsi[ibi].x_c);
	      		yc_min = PetscMin(yc_min, fsi[ibi].y_c);
      			zc_min = PetscMin(zc_min, fsi[ibi].z_c);
	    	}

//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 1 \n");
		int IndWT[Nz_WT][Ny_WT][Nx_WT];

		double fac_x=1.0/Sx_WT, fac_y=1.0/Sy_WT, fac_z=1.0/Sz_WT;
        	for (ibi=0; ibi<NumberOfObjects; ibi++) {
			int ii=(int)((fsi[ibi].x_c-xc_min+1.e-9)*fac_x);
			int jj=(int)((fsi[ibi].y_c-yc_min+1.e-9)*fac_y);
			int kk=(int)((fsi[ibi].z_c-zc_min+1.e-9)*fac_z);

        		PetscPrintf(PETSC_COMM_WORLD, "ibi ii jj kk %d %d %d %d \n", ibi, ii, jj, kk);
			IndWT[kk][jj][ii]=ibi;
		}

		int i, j, k;

//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 2 \n");
		for (k=0;k<Nz_WT;k++)
		for (j=0;j<Ny_WT;j++)
		for (i=0;i<Nx_WT;i++){

			if (ii_periodicWT && i==0) {
				ibi=IndWT[k][j][i];
				int ibi1=IndWT[k][j][Nx_WT-1];
				for (l=0; l<ibm[ibi].n_elmt; l++) {
					ibm[ibi].U_lagr_x[l]=ibm[ibi].U_lagr_x[l]+ibm[ibi1].U_lagr_x[l];
					ibm[ibi].U_lagr_y[l]=ibm[ibi].U_lagr_y[l]+ibm[ibi1].U_lagr_y[l];
					ibm[ibi].U_lagr_z[l]=ibm[ibi].U_lagr_z[l]+ibm[ibi1].U_lagr_z[l];
				}
			}

                        if (jj_periodicWT && j==0) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[k][Ny_WT-1][i];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi].U_lagr_x[l]+ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi].U_lagr_y[l]+ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi].U_lagr_z[l]+ibm[ibi1].U_lagr_z[l];
                                }
                        }

                        if (kk_periodicWT && k==0) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[Nz_WT-1][j][i];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi].U_lagr_x[l]+ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi].U_lagr_y[l]+ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi].U_lagr_z[l]+ibm[ibi1].U_lagr_z[l];
                                }

//				PetscPrintf(PETSC_COMM_WORLD, "U_lagr_z %le \n",ibm[ibi].U_lagr_z[0] );
                        }
		}


//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 3 \n");
                for (k=0;k<Nz_WT;k++)
                for (j=0;j<Ny_WT;j++)
                for (i=0;i<Nx_WT;i++){

                        if (ii_periodicWT && i==Nx_WT-1) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[k][j][0];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi1].U_lagr_z[l];
                                }
                        }

                        if (jj_periodicWT && j==Ny_WT-1) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[k][0][i];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi1].U_lagr_z[l];
                                }
                        }

                        if (kk_periodicWT && k==Nz_WT-1) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[0][j][i];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi1].U_lagr_z[l];
                                }
                        }
                }

//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 4 \n");

	}


	if (MoveFrame) {
		for (ibi=0; ibi<NumberOfObjects; ibi++)
	  	for (l=0; l<ibm[ibi].n_elmt; l++) {
                	ibm[ibi].U_lagr_x[l] += u_frame;
                	ibm[ibi].U_lagr_y[l] += v_frame;
             		ibm[ibi].U_lagr_z[l] += w_frame;
  		}
  	}


  	DMDAVecRestoreArray(fda, Coor, &coor);
  	DMDAVecRestoreArray(fda, user->lUcat, &ucat);
  	DMDAVecRestoreArray(fda, user->lCsi,  &csi);
  	DMDAVecRestoreArray(fda, user->lEta,  &eta);
  	DMDAVecRestoreArray(fda, user->lZet,  &zet);
  	DMDAVecRestoreArray(da,  user->lAj,  &aj);

	return(0);
}


// distribute the force on the turbines to background grid.
//
// called after Calc_F_lagr_ACL or Calc_F_lagr before solving the momentum equatiion
PetscErrorCode Calc_F_eul(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{

  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo     info;
  	PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  	PetscInt        mx, my, mz; // Dimensions in three directions
  	PetscInt        i, j, k, l, ibi;
  	PetscInt	lxs, lxe, lys, lye, lzs, lze;

  	Cmpnts		***lf_eul, ***coor, ***csi, ***eta, ***zet;

  	PetscReal 	***aj;

  	Vec		Coor;

  	PetscReal	xc, yc, zc, hx, hy, hz, vol_eul, vol_lagr;
  
  	double		dfunc;

  	double r1, r2, r3;

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
  	DMDAVecGetArray(fda, user->lF_eul, &lf_eul);
  	DMDAVecGetArray(fda, user->lCsi,  &csi);
  	DMDAVecGetArray(fda, user->lEta,  &eta);
  	DMDAVecGetArray(fda, user->lZet,  &zet);
  	DMDAVecGetArray(da,  user->lAj,  &aj);

  	double ni[3], nj[3], nk[3];
  	for (ibi=0; ibi<NumberOfObjects; ibi++) {
  	for (l=0; l<ibm[ibi].n_elmt; l++) {

	    	for (k=ibm[ibi].k_min[l]; k<ibm[ibi].k_max[l]; k++) 
	      	for (j=ibm[ibi].j_min[l]; j<ibm[ibi].j_max[l]; j++) 
	        for (i=ibm[ibi].i_min[l]; i<ibm[ibi].i_max[l]; i++) {

//                xc = (coor[k  ][j  ][i].x + coor[k-1][j  ][i].x + coor[k  ][j-1][i].x + coor[k-1][j-1][i].x) * 0.25;
//                yc = (coor[k  ][j  ][i].y + coor[k-1][j  ][i].y + coor[k  ][j-1][i].y + coor[k-1][j-1][i].y) * 0.25;
//                zc = (coor[k  ][j  ][i].z + coor[k-1][j  ][i].z + coor[k  ][j-1][i].z + coor[k-1][j-1][i].z) * 0.25;

			xc=coor[k][j][i].x;
			yc=coor[k][j][i].y;
			zc=coor[k][j][i].z;

	            	double rx= (xc - ibm[ibi].cent_x[l]), ry = (yc - ibm[ibi].cent_y[l]), rz = (zc - ibm[ibi].cent_z[l]);

			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);

			double dhx_=ibm[ibi].dhx[l];
			double dhy_=ibm[ibi].dhy[l];
			double dhz_=ibm[ibi].dhz[l];

			r1=fabs(rx*ni[0]+ry*ni[1]+rz*ni[2])/dhx_; 
			r2=fabs(rx*nj[0]+ry*nj[1]+rz*nj[2])/dhy_; 
			r3=fabs(rx*nk[0]+ry*nk[1]+rz*nk[2])/dhz_; 

	          	vol_eul = aj[k][j][i];

			if (deltafunc == 1) dfunc = vol_eul * dfunc_2h(r1) * dfunc_2h(r2) * dfunc_2h(r3);
			if (deltafunc == 2) dfunc = vol_eul * dfunc_4h(r1) * dfunc_4h(r2) * dfunc_4h(r3);
			if (deltafunc == 3) dfunc = vol_eul * dfunc_s3h(r1) * dfunc_s3h(r2) * dfunc_s3h(r3);
			if (deltafunc == 4) dfunc = vol_eul * dfunc_s4h(r1) * dfunc_s4h(r2) * dfunc_s4h(r3);


	            	lf_eul[k][j][i].x += ibm[ibi].F_lagr_x[l] * dfunc * ibm[ibi].dA[l] * csi[k][j][i].x +
	                                 ibm[ibi].F_lagr_y[l] * dfunc * ibm[ibi].dA[l] * csi[k][j][i].y +
                                 	ibm[ibi].F_lagr_z[l] * dfunc * ibm[ibi].dA[l] * csi[k][j][i].z;

	            	lf_eul[k][j][i].y += ibm[ibi].F_lagr_x[l] * dfunc * ibm[ibi].dA[l] *  eta[k][j][i].x +
        	                         ibm[ibi].F_lagr_y[l] * dfunc * ibm[ibi].dA[l] *  eta[k][j][i].y +
                	                 ibm[ibi].F_lagr_z[l] * dfunc * ibm[ibi].dA[l] *  eta[k][j][i].z;

	              	lf_eul[k][j][i].z += ibm[ibi].F_lagr_x[l] * dfunc * ibm[ibi].dA[l] *  zet[k][j][i].x +
        	                         ibm[ibi].F_lagr_y[l] * dfunc * ibm[ibi].dA[l] *  zet[k][j][i].y +
                	                 ibm[ibi].F_lagr_z[l] * dfunc * ibm[ibi].dA[l] *  zet[k][j][i].z;
       
//			ibm[ibi].F_lagr_x[l]=1.00022323;
//			lf_eul[k][j][i].x+=ibm[ibi].F_lagr_x[l] ; 
//			lf_eul[k][j][i].y+=ibm[ibi].F_lagr_x[l] ; 
//			lf_eul[k][j][i].z+=ibm[ibi].F_lagr_x[l] ; 
	      	}
	}
	}

  
 	DMDAVecRestoreArray(fda, Coor, &coor);
  	DMDAVecRestoreArray(fda, user->lF_eul, &lf_eul);
  	DMDAVecRestoreArray(fda, user->lCsi,  &csi);
  	DMDAVecRestoreArray(fda, user->lEta,  &eta);
  	DMDAVecRestoreArray(fda, user->lZet,  &zet);
  	DMDAVecRestoreArray(da,  user->lAj,  &aj);

  	DMLocalToGlobalBegin(fda, user->lF_eul, INSERT_VALUES, user->F_eul);
  	DMLocalToGlobalEnd(fda, user->lF_eul, INSERT_VALUES, user->F_eul);


  return(0);
}


// exporting the forces on the turbines actuator disk model
PetscErrorCode Calc_forces_rotor(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int bi)
{
  	PetscInt      	l, ibi;
  	double        	pi = 3.141592653589793;
  	PetscReal	A_Sum, F_Sum, P_Sum, U_Sum;


  	for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    		A_Sum = 0.0; P_Sum = 0.0; U_Sum = 0.0, F_Sum=0.0;
		double nx=fsi[ibi].nx_tb, ny=fsi[ibi].ny_tb, nz=fsi[ibi].nz_tb;

    		for (l=0; l<ibm[ibi].n_elmt; l++) {

      			A_Sum += ibm[ibi].dA[l] ;

			double F_axis=ibm[ibi].F_lagr_x[l]*nx+ibm[ibi].F_lagr_y[l]*ny+ibm[ibi].F_lagr_z[l]*nz;
			double U_axis=ibm[ibi].U_lagr_x[l]*nx+ibm[ibi].U_lagr_y[l]*ny+ibm[ibi].U_lagr_z[l]*nz;

      			F_Sum += F_axis*ibm[ibi].dA[l] ;
      			P_Sum += F_axis*ibm[ibi].dA[l]*U_axis;

	      		U_Sum += U_axis*ibm[ibi].dA[l] ;

    		}
   
		U_Sum=U_Sum/A_Sum;

	    	int rank=0;
    		MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	    	if (!rank) {
      			FILE *f;
      			char filen[80];
	      		sprintf(filen, "Turbine_AD%2.2d_%2.2d",rotor_model,ibi);
      			if (ti==1) {
				f = fopen(filen, "w");
      				PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=\"time\", \"a\", \"F\", \"U\", \"P\" \n");
			} else f = fopen(filen, "a");

      			PetscFPrintf(PETSC_COMM_WORLD, f, "%d %le %le %le %le \n",ti, ibm[ibi].indf_axis,-F_Sum,U_Sum,-P_Sum);
	      		fclose(f);
	    	}


        	if ( ti % tiout == 0 /*ti == (ti/tiout) * tiout */) {
		        int rank=0;
        		MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
			int i;

			double dt = user->dt;
		        if (!rank) {
        		        FILE *f;
	                	char filen[80];
        		        sprintf(filen, "%s/ACDsurface%06d_%03d_nf.dat",path,ti,ibi);

		                f = fopen(filen, "w");
	
				int n_v=ibm[ibi].n_v; 	
				int n_elmt=ibm[ibi].n_elmt; 	

		                PetscFPrintf(PETSC_COMM_WORLD, f, "TITLE = \"Actuator disk mesh\" \n ");
			    	PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z\n");
		    		PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-6]=CELLCENTERED)\n", n_v, n_elmt);
                		PetscFPrintf(PETSC_COMM_WORLD, f, "STRANDID=0.1 SOLUTIONTIME=%le \n", ((double)ti)*dt);

		    		for (i=0; i<n_v; i++) {
	      				PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].x_bp[i]);
		    		}
    				for (i=0; i<n_v; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].y_bp[i]);
		    		}
		    		for (i=0; i<n_v; i++) {	
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].z_bp[i]);
		    		}
		    		for (i=0; i<n_elmt; i++) {
      					PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].nf_x[i]);
		    		}
    				for (i=0; i<n_elmt; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].nf_y[i]);
    				}
		    		for (i=0; i<n_elmt; i++) {
      					PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].nf_z[i]);
		    		}
    				for (i=0; i<n_elmt; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
		    		}
 

                		fclose(f);

		        }
        	}
	}

}




double dfunc_4h(double r)
{
  if (fabs(r) <= 1.0) {
    return (3.0-2.0*fabs(r)+sqrt(1.0+4.0*fabs(r)-4.0*r*r))/8.0;
  } else if (fabs(r) >= 1.0 && fabs(r) <= 2.0) {
    return (5.0-2.0*fabs(r)-sqrt(-7.0+12.0*fabs(r)-4.0*r*r))/8.0;
  } else {
    return 0.0;
  }
}

double dfunc_s3h(double r)
{
  if (fabs(r) <= 1.0) {
    return 17.0/48.0 + sqrt(3.0)*3.14159265/108.0 + fabs(r)/4.0 - r*r/4.0 + (1.0-2.0*fabs(r))*sqrt(-12.0*r*r+12.0*fabs(r)+1.0)/16.0 - sqrt(3.0)*asin(sqrt(3.0)*(2.0*fabs(r)-1.0)/2.0)/12.0;
  } else if (fabs(r) >= 1.0 && fabs(r) <= 2.0) {
    return 55.0/48.0 - sqrt(3.0)*3.14159265/108.0 - 13.0*fabs(r)/12.0 + r*r/4.0 + (2.0*fabs(r)-3.0)*sqrt(-12.0*r*r+36.0*fabs(r)-23.0)/48.0 + sqrt(3.0)*asin(sqrt(3.0)*(2.0*fabs(r)-3.0)/2.0)/36.0;
  } else { 
    return 0.0;
  }

}


double dfunc_2h(double r)
{
  if (fabs(r) < 1.0) {
    return 1.0-fabs(r);
  } else {
    return 0.0;
  }
}


double dfunc_s4h(double r)
{

  if (fabs(r) <= 0.5) {
    return 3.0/8.0+3.14159265/32.0-pow(r,2)/4.0;
  } else if (fabs(r) >= 0.5 && fabs(r) <= 1.5) {
    return 1.0/4.0+(1.0-fabs(r))*sqrt(-2.0+8.0*fabs(r)-4.0*pow(r,2))/8.0-asin(sqrt(2.0)*(fabs(r)-1.0))/8.0;
  } else if (fabs(r) >= 1.5 && fabs(r) <= 2.5) {
    return
17.0/16.0-3.14159265/64.0-3.0*fabs(r)/4.0+pow(r,2)/8.0+(fabs(r)-2.0)*sqrt(-14.0+16.0*fabs(r)-4.0*pow(r,2))/16.0+asin(sqrt(2.0)*(fabs(r)-2.0))/16.0;
  } else {
    return 0.0;
  }

}


Cmpnts ArbitraryRotate(Cmpnts p,double theta,Cmpnts r) 
{
   	Cmpnts q = {0.0,0.0,0.0};
   	double costheta,sintheta;
	
	double rr=sqrt(r.x*r.x+r.y*r.y+r.z*r.z)+1.0e-11;
	r.x=r.x/rr; r.y=r.y/rr; r.z=r.z/rr;

   	costheta = cos(theta);
   	sintheta = sin(theta);

   	q.x += (costheta + (1 - costheta) * r.x * r.x) * p.x;
   	q.x += ((1 - costheta) * r.x * r.y - r.z * sintheta) * p.y;
   	q.x += ((1 - costheta) * r.x * r.z + r.y * sintheta) * p.z;

   	q.y += ((1 - costheta) * r.x * r.y + r.z * sintheta) * p.x;
   	q.y += (costheta + (1 - costheta) * r.y * r.y) * p.y;
   	q.y += ((1 - costheta) * r.y * r.z - r.x * sintheta) * p.z;

   	q.z += ((1 - costheta) * r.x * r.z - r.y * sintheta) * p.x;
   	q.z += ((1 - costheta) * r.y * r.z + r.x * sintheta) * p.y;
   	q.z += (costheta + (1 - costheta) * r.z * r.z) * p.z;

   	return(q);
}




