
#include "variables.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector> 
#include <iostream>
using namespace std;

int num_innergrid = 81;
double pi = 3.14159265358979323846;

void innergrid( double *z_in, double h);

extern void Subtract_Scale_AddTo ( Cmpnts &A, Cmpnts &B, double a, Cmpnts *C);

extern void Subtract_Scale_Set ( Cmpnts &A, Cmpnts &B, double a, Cmpnts *C);

	
// double *u_in, *z_in, *nut_les, *nut_rans;

// =========================================================================================================================

// xyang 1-2-2011

// 

double utau_powerlaw(double nu, double ut_mag, double sc)
{
//	double nu = 1./user->ren;
//	double ybp = sb*(*ustar)/nu;
	double A=8.3;
	double B=1.0/7.0; //1.085/log(1.0/nu)+6.535/pow(log(1.0/nu),2);
	double ustar;
        ustar = pow( ut_mag * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));
	double ycp = sc*ustar/nu;
//	double ut_mag_modeled;
	if (ycp>12.0) {
        	ustar = pow( ut_mag * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));
//		ut_mag_modeled = A*(*ustar)*pow(ybp,B);
	} 
	else {
		ustar = sqrt(fabs(nu*ut_mag/sc));
//		ut_mag_modeled = ut_mag*sb/sc;	
	}



	return ustar; //pow( ut_mag * pow(nu, 1.0/7.0) / (8.3 * pow(sc, 1.0/7.0)),  7.0/8.0);	 

}

void wallmodel_1( double dpdx, double dpdy, double dpdz, double nu, double *nu_t_b, double nu_t_c, double sb, double sc, 
                  Cmpnts *Ub, Cmpnts Uc, Cmpnts Ua, PetscInt bctype, double ks,
                  double nx, double ny, double nz, double *f_i, double *f_j, double *f_k, double *f_i_c, double *f_j_c, double *f_k_c, PetscReal *ustar, Cmpnts *Ug)
{
	double kappa_rans = 0.41, A = 19;
	double kappa_les;
	double coeff_force, coeff_c1, f0_sum_c, f1_sum_c, f2_sum_c, u_flux, coeff_c0;

        double tau_w;

        double t1_x, t1_y, t1_z, t2_x, t2_y, t2_z;

        double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
        double un = u_c * nx + v_c * ny + w_c * nz;
        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );


	sb += 1.e-6;

        t1_x = ut / (ut_mag_c+1.e-6); t1_y = vt / (ut_mag_c+1.e-6); t1_z = wt / (ut_mag_c+1.e-6);

        double u_b = (*Ub).x - Ua.x, v_b = (*Ub).y - Ua.y, w_b = (*Ub).z - Ua.z;
        double ut_mag_b = u_b * t1_x + v_b * t1_y + w_b * t1_z;

        double u_a = Ua.x, v_a = Ua.y, w_a = Ua.z;
        double ut_mag_a = u_a * t1_x + v_a * t1_y + w_a * t1_z;

        double dpdt1 = dpdx * t1_x + dpdy * t1_y + dpdz * t1_z;

        int i, j, k;
        double yp, yp_c;
	
	double ustartp, ustarp;
	
	u_flux = ut_mag_b*sb*2.0 + ut_mag_c * ( sc - 2.0 * sb );

        num_innergrid = (int)(log(sc*(dhratio_wm-1)/dh1_wm+1.0)/log(dhratio_wm))+1;

        double *z_in, *f0, *f1, *f2, *nu_t_les, *nu_t_rans, *u_inner;

        z_in= (double *) malloc(num_innergrid*sizeof(double));
        f0= (double *) malloc(num_innergrid*sizeof(double));
        f1= (double *) malloc(num_innergrid*sizeof(double));
        f2= (double *) malloc(num_innergrid*sizeof(double));
        nu_t_les= (double *) malloc(num_innergrid*sizeof(double));
        nu_t_rans= (double *) malloc(num_innergrid*sizeof(double));
        u_inner= (double *) malloc(num_innergrid*sizeof(double));

	innergrid( z_in, sc );

        if ( ti == tistart) (*ustar) = utau_powerlaw(nu, ut_mag_c, sc); // Here, the ustar at last time step is used to calculate the nu_t, but for ti == tistart ustar is not available. The powerlaw is used to caculate the ustar.

	ustarp = pow( fabs(nu*dpdt1), 1.0/3.0 );

	ustartp = sqrt( (*ustar)*(*ustar) + ustarp * ustarp);  

        yp = sc * ustartp / nu;

        double damping = (1.0 - exp(-yp/A)) * (1.0 - exp(-yp/A));
        kappa_les = nu_t_c/(nu * yp * damping+1.e-9);

	if (kappa_les <= 0.0 || kappa_les > kappa_rans) kappa_les = kappa_rans;

	nu_t_rans[0] = 0.0;
	nu_t_les[0] = 0.0;
	for (k = 1; k < num_innergrid; k++) {
        	yp = z_in[k] * ustartp / nu;
        	damping = (1.0 - exp(-yp/A)) * (1.0 - exp(-yp/A));
        	nu_t_rans[k] = nu * kappa_rans * yp * damping;
        	nu_t_les[k] = nu * kappa_les * yp * damping;
	}


	double sum0, sum1, sum2, dz;
	sum0 = 0.0; sum1 = 0.0; //sum2 = 0.0;
	f0[0] = 0.0; f1[0] = 0.0; //f2[0] = 0.0;
	for (k = 1; k < num_innergrid; k++) {
		dz = z_in[k] - z_in[k-1];
		sum0 += dz / ( nu + 0.5*(nu_t_rans[k]+nu_t_rans[k-1]) );
		sum1 += dz * 0.5 * ( z_in[k] + z_in[k-1] ) / ( nu + 0.5*(nu_t_rans[k]+nu_t_rans[k-1]) );
		f0[k] = sum0; f1[k] = sum1; // f2[k] = sum2;	
	}

	f0_sum_c = 0.0; f1_sum_c = 0.0; // f2_sum_c = 0.0;
        for (k = 1; k < num_innergrid; k++) {
		dz = z_in[k] - z_in[k-1];
		f0_sum_c += dz * 0.5 * ( f0[k] + f0[k-1] );
		f1_sum_c += dz * 0.5 * ( f1[k] + f1[k-1] );
	}

	if (bctype == 1) {

        	coeff_c1 = ut_mag_c / f0[num_innergrid-1];
        	for (k = 0; k < num_innergrid; k++) {
                	u_inner[k] = ut_mag_a + coeff_c1 * f0[k];
        	}


	}	

	if (bctype == 2) {

        	double r1 = ut_mag_c - dpdt1 * f1[num_innergrid-1];
        	coeff_c1 = r1 / f0[num_innergrid-1];

        	for (k = 0; k < num_innergrid; k++) {
                	u_inner[k] = ut_mag_a + dpdt1 * f1[k] +  coeff_c1 * f0[k];
        	}

	}


	if (bctype ==3) {

        	sum2 = 0.0;
        	f2[0] = 0.0;
        	for (k = 1; k < num_innergrid; k++) {
                	dz = z_in[k] - z_in[k-1];
                	sum2 += dz * pow( 0.5 * ( z_in[k] + z_in[k-1] ), 1 ) / ( nu + 0.5*(nu_t_rans[k]+nu_t_rans[k-1]) );
                	f2[k] = sum2;
        	}

        	f2_sum_c = 0.0;
        	for (k = 1; k < num_innergrid; k++) {
                	dz = z_in[k] - z_in[k-1];
                	f2_sum_c += dz * 0.5 * ( f2[k] + f2[k-1] );
        	}


        	double r1 = ut_mag_c - dpdt1 * f1[num_innergrid-1], r2 = u_flux - dpdt1 * f1_sum_c;

        	double mat[2][2];

        	mat[0][0] = 0.5 * f2[num_innergrid-1]; mat[0][1] = f0[num_innergrid-1];
        	mat[1][0] = 0.5 * f2_sum_c; mat[1][1] = f0_sum_c;

        	double det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
        	double det1 = r1 * mat[1][1] - r2 * mat[0][1] ;
        	double det2 = mat[0][0] * r2 - mat[1][0] * r1;

      		coeff_force = det1 / det; coeff_c1 = det2 / det;

        	for (k = 0; k < num_innergrid; k++) {
                	u_inner[k] = ut_mag_a + dpdt1 * f1[k] + coeff_force * f2[k] +  coeff_c1 * f0[k];
        	}


	}

        if (bctype ==4) {

                sum2 = 0.0;
                f2[0] = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        sum2 += dz * pow( 0.5 * ( z_in[k] + z_in[k-1] ), 2 ) / ( nu + 0.5*(nu_t_rans[k]+nu_t_rans[k-1]) );
                        f2[k] = sum2;
                }

                f2_sum_c = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        f2_sum_c += dz * 0.5 * ( f2[k] + f2[k-1] );
                }


                double r1 = ut_mag_c - dpdt1 * f1[num_innergrid-1], r2 = u_flux - dpdt1 * f1_sum_c;

                double mat[2][2];

                mat[0][0] = 0.5 * f2[num_innergrid-1]; mat[0][1] = f0[num_innergrid-1];
                mat[1][0] = 0.5 * f2_sum_c; mat[1][1] = f0_sum_c;

                double det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
                double det1 = r1 * mat[1][1] - r2 * mat[0][1] ;
                double det2 = mat[0][0] * r2 - mat[1][0] * r1;

                coeff_force = det1 / det; coeff_c1 = det2 / det;

                for (k = 0; k < num_innergrid; k++) {
                        u_inner[k] = ut_mag_a + dpdt1 * f1[k] + 0.5 * coeff_force * f2[k] +  coeff_c1 * f0[k];
                }


        }

        if (bctype ==5) {

                sum2 = 0.0;
                f2[0] = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        sum2 += dz * pow( 0.5 * ( z_in[k] + z_in[k-1] ), 3 ) / ( nu + 0.5*(nu_t_rans[k]+nu_t_rans[k-1]) );
                        f2[k] = sum2;
                }

                f2_sum_c = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        f2_sum_c += dz * 0.5 * ( f2[k] + f2[k-1] );
                }


                double r1 = ut_mag_c - dpdt1 * f1[num_innergrid-1], r2 = u_flux - dpdt1 * f1_sum_c;
                double mat[2][2];

                mat[0][0] = 0.5 * f2[num_innergrid-1]; mat[0][1] = f0[num_innergrid-1];
                mat[1][0] = 0.5 * f2_sum_c; mat[1][1] = f0_sum_c;

                double det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
                double det1 = r1 * mat[1][1] - r2 * mat[0][1] ;
                double det2 = mat[0][0] * r2 - mat[1][0] * r1;

                coeff_force = det1 / det; coeff_c1 = det2 / det;

                for (k = 0; k < num_innergrid; k++) {
                        u_inner[k] = ut_mag_a + dpdt1 * f1[k] + coeff_force * f2[k]/3.0 +  coeff_c1 * f0[k];
                }


        }

        if (bctype == 6 ) {


		(*ustar) = ut_mag_c / ( log(sc/ks)/kappa_rans + 8.5);

                sum2 = 0.0;
                f2[0] = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        sum2 += dz * pow( 0.5 * ( z_in[k] + z_in[k-1] ), 2 ) / ( nu + 0.5*(nu_t_rans[k]+nu_t_rans[k-1]) );
                        f2[k] = sum2;
                }

                f2_sum_c = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        f2_sum_c += dz * 0.5 * ( f2[k] + f2[k-1] );
                }


                double r1 = ut_mag_c - (*ustar)*(*ustar)*f0[num_innergrid-1] - dpdt1 * f1[num_innergrid-1];
 		double r2 = u_flux - (*ustar)*(*ustar)*f0_sum_c - dpdt1 * f1_sum_c;

                double mat[2][2];

                mat[0][0] = 0.5 * f2[num_innergrid-1]; mat[0][1] = 1.0;
                mat[1][0] = 0.5 * f2_sum_c; mat[1][1] = sc;

                double det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
                double det1 = r1 * mat[1][1] - r2 * mat[0][1] ;
                double det2 = mat[0][0] * r2 - mat[1][0] * r1;

                coeff_force = det1 / det; coeff_c0 = det2 / det;

                for (k = 0; k < num_innergrid; k++) {
                        u_inner[k] = coeff_c0 + (*ustar)*(*ustar)*f0[k] + dpdt1 * f1[k] + 0.5 * coeff_force * f2[k];
                }


        }




	double dudz = (u_inner[1] - u_inner[0]) / (z_in[1] - z_in[0]);
	tau_w = nu * dudz; 

	double tau_w_all = sqrt( tau_w * tau_w );
	(*ustar) = sqrt(tau_w_all);


	int k_2sb;


	for (k = 1; k < num_innergrid; k++) {
		if (z_in[k-1] < 2.0*sb && z_in[k] >= 2.0*sb) k_2sb = k;
	}

	double force_sb=0.0;

        int k_sb;

        for (k = 1; k < num_innergrid; k++) {
                if (z_in[k-1] < sb && z_in[k] >= sb) k_sb = k;
        }

	(*nu_t_b)=nu_t_rans[k_sb];

	if (ApproxBC_wm==1) {
        	double tau_c = ( nu + nu_t_c ) * (ut_mag_c - ut_mag_b) / (sc - sb);
	        double dudz = (u_inner[k_sb+1] - u_inner[k_sb-1]) / (z_in[k_sb+1] - z_in[k_sb-1]);
		double tau_b = (nu+nu_t_rans[k_sb]) * dudz; 
	        force_sb = (tau_c-tau_b) /(sc-sb);
	}
	if (ApproxBC_wm==2) {
        	double tau_b = ( nu + (*nu_t_b) ) * (ut_mag_c - ut_mag_b) / (sc - sb);
		double dudz = (u_inner[1] - u_inner[0]) / (z_in[1] - z_in[0]);
		tau_w = nu * dudz; 
	        force_sb = (tau_b-tau_w) /sb;
	}


        double tau_c = ( nu + 0.5*(nu_t_c+(*nu_t_b)) ) * (ut_mag_c - ut_mag_b) / (sc - sb);

        tau_w = ( nu + 0.5*(*nu_t_b) ) * ut_mag_b / sb;
        double force_sb_noslip = (tau_c - tau_w) / (0.5*sc);
        force_sb = force_sb - force_sb_noslip;

       	(*f_i) = force_sb * t1_x;
        (*f_j) = force_sb * t1_y;
        (*f_k) = force_sb * t1_z;

       	(*f_i_c) = 0.0; //force_sc * t1_x;
     	(*f_j_c) = 0.0; //force_sc * t1_y;
     	(*f_k_c) = 0.0; //force_sc * t1_z;


        free(z_in);
        free(f0); 
        free(f1); 
        free(f2);
        free(nu_t_les);
        free(nu_t_rans);
        free(u_inner);


}


void wallmodel_tmprt( double sb, double nx, double ny, double nz, Cmpnts Ub, Cmpnts Ua, double Tb, double Ta, PetscInt bctype, double ks, double *qs)
//, double Dm, double Dt_b, double Dt_c)
{
	double kappa_rans = 0.4;

        double u_b = Ub.x - Ua.x, v_b = Ub.y - Ua.y, w_b = Ub.z - Ua.z;
        double un = u_b * nx + v_b * ny + w_b * nz;
        double ut = u_b - un * nx, vt = v_b - un * ny, wt = w_b - un * nz;
        double ut_mag_b = sqrt( ut*ut + vt*vt + wt*wt );
	
        if (bctype == 6 ) {
		double k0=ks*0.033;
		double k0_tmprt=k0*0.1;

		double fac1=log(sb/k0_tmprt)*log(sb/k0);
		double fac2=pow(kappa_rans,2)*(Ta-Tb)*ut_mag_b;
		(*qs)=fac2/fac1;
        }

}

/*
void wall_function_s (double nu, double ks, double sc, double sb, Cmpnts Ua, Cmpnts Uc, Cmpnts *Ub, PetscReal *ustar, double nx, double ny, double nz)
{
	double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
	double un = u_c * nx + v_c * ny + w_c * nz;
	double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
	double ut_mag = sqrt( ut*ut + vt*vt + wt*wt );

	double A=8.3;
	double B=1.0/7.0; 
        *ustar = pow( ut_mag * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));  /// 

	double kappa_rans = 0.4;
	double _ks=ks*0.033;
	if (_ks>1.e-9) {
		*ustar = ut_mag*kappa_rans/log(sc/_ks);
	}

	double ybp = sb*(*ustar)/nu;
	double ycp = sc*(*ustar)/nu;
	double ut_mag_modeled;
	if (ybp>12.0) {
		ut_mag_modeled = A*(*ustar)*pow(ybp,B);
		if (_ks>1.e-9) ut_mag_modeled = (*ustar)*log(sb/_ks)/kappa_rans;
	} 
	else {
		ut_mag_modeled = (*ustar)*ybp;	
	}


	if(ut_mag>1.e-10) {
		ut *= ut_mag_modeled/ut_mag;
		vt *= ut_mag_modeled/ut_mag;
		wt *= ut_mag_modeled/ut_mag;
	}
	else ut=vt=wt=0;
					
	// u = ut + (u.n)n
	(*Ub).x = ut + sb/sc * un * nx;
	(*Ub).y = vt + sb/sc * un * ny;
	(*Ub).z = wt + sb/sc * un * nz;
	
	(*Ub).x += Ua.x;
	(*Ub).y += Ua.y;
	(*Ub).z += Ua.z;
}
*/
void wallfunction_s( double nu, double sb, double sc, Cmpnts Uc, Cmpnts *Ub,  Cmpnts Ua,double ks,double nx, double ny, double nz, PetscReal *ustar, double dpdx, double dpdy, double dpdz) 
{

//	PetscPrintf(PETSC_COMM_WORLD, "***** Test 1 1  1 \n");
	double kappa_rans = 0.4;
        double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
        double un = u_c * nx + v_c * ny + w_c * nz;
        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );
	double eps=1.e-11;

	double A=8.3;
	double B=1.0/7.0; 
        *ustar = pow( ut_mag_c * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));  /// 


	double _ks=ks*0.033;
	if (_ks>1.e-9) {
		*ustar = ut_mag_c*kappa_rans/log(sc/_ks);
	}
	
	*ustar = fmax(sqrt(fabs(nu*ut_mag_c/sc)),*ustar);

	double ybp = sb*(*ustar)/nu;
	double ycp = sc*(*ustar)/nu;
	double ut_mag_modeled;
	if (ycp>12.0) { 
		ut_mag_modeled = A*(*ustar)*pow(ybp,B);
		if (_ks>1.e-9) ut_mag_modeled = (*ustar)*log(sb/_ks)/kappa_rans;
	} 
	else {
		ut_mag_modeled = ut_mag_c*sb/sc;	
	}


	//
	if(ut_mag_c>eps) {
		ut *= ut_mag_modeled/ut_mag_c;
		vt *= ut_mag_modeled/ut_mag_c;
		wt *= ut_mag_modeled/ut_mag_c;
	}
	else ut=vt=wt=0;
				
	double dpdn_sc=dpdx*nx+dpdy*ny+dpdy*nz;
	double dpdn_sb=dpdn_sc*sb/sc;
	double sign_sc=-dpdn_sc/(fabs(dpdn_sc)+eps);
	double coeff=sign_sc*un/sqrt(fabs(dpdn_sc)*sc);
	double sign_sb=-dpdn_sb/(fabs(dpdn_sb)+eps);
	double un_sb=fabs(coeff)*sqrt(fabs(dpdn_sb)*sb)*sign_sb;

	/*
	(*Ub).x = ut+un_sb*nx;
	(*Ub).y = vt+un_sb*ny;
	(*Ub).z = wt+un_sb*nz;
	*/
	
	(*Ub).x = ut + pow(sb/sc,1) * un * nx;
	(*Ub).y = vt + pow(sb/sc,1) * un * ny;
	(*Ub).z = wt + pow(sb/sc,1) * un * nz;
	

	
	(*Ub).x += Ua.x;
	(*Ub).y += Ua.y;
	(*Ub).z += Ua.z;

}




void wallmodel_s( double nu, double sb, double sc, Cmpnts Uc, Cmpnts *Ub,  Cmpnts Ua, PetscInt bctype, double ks,double nx, double ny, double nz, double *tau_w, PetscReal *ustar, double dpdx, double dpdy, double dpdz,double *nut_2sb, double nut_c)
{

	double kappa_rans = 0.4;
        double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
        double un = u_c * nx + v_c * ny + w_c * nz;
        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );
	double eps=1.e-11;

	double ustar_noslip=sqrt(nu*ut_mag_c/sc);

	double A=8.3;
	//double B=1.0/7.0; 
	
	double B=1.0/7.6; 
        *ustar = pow( ut_mag_c * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));  /// 

	//PetscPrintf(PETSC_COMM_WORLD, "ustar=%le\n", *ustar);


 	double _ks=ks*0.033;
	if (bctype==6)  *ustar = ut_mag_c*kappa_rans/log(sc/_ks);
	
	
//	*ustar = fmax(sqrt(fabs(nu*ut_mag_c/sc)),*ustar);

	double ybp = sb*max((*ustar),ustar_noslip)/nu;
	double ycp = sc*max((*ustar),ustar_noslip)/nu;
	double ut_mag_modeled;
	if (ycp>12.0) { 
		ut_mag_modeled = A*(*ustar)*pow(ybp,B);
		if (bctype==6) ut_mag_modeled = (*ustar)*log(sb/_ks)/kappa_rans;
	} 
	else {
		ut_mag_modeled = ut_mag_c*sb/sc;	
		*ustar = sqrt(fabs(nu*ut_mag_c/sc));
	}

	double sign=ut_mag_c/(fabs(ut_mag_c)+eps);
	*tau_w=pow(*ustar,2)*sign;

	//

        double t1_x, t1_y, t1_z, t2_x, t2_y, t2_z;

        t1_x = ut / (ut_mag_c+eps); t1_y = vt / (ut_mag_c+eps); t1_z = wt / (ut_mag_c+eps);

        double dpdt = dpdx * t1_x + dpdy * t1_y + dpdz * t1_z;

	double up = pow( fabs(nu*dpdt), 1.0/3.0 );
	double uall = sqrt(pow(*ustar,2)+up*up);  
        double yp=sc*uall/nu;

        double damping = (1.0-exp(-yp/A))*(1.0-exp(-yp/A));
        double kappa = nut_c/(nu*yp*damping+eps);

        yp=0.5*(sb+sc)*uall/nu;
        damping =(1.0-exp(-yp/A))*(1.0-exp(-yp/A));
	*nut_2sb=nu*kappa*yp*damping;

	//

	if(ut_mag_c>eps) {
		ut *= ut_mag_modeled/ut_mag_c;
		vt *= ut_mag_modeled/ut_mag_c;
		wt *= ut_mag_modeled/ut_mag_c;
	}
	else ut=vt=wt=0;
				
	double dpdn_sc=dpdx*nx+dpdy*ny+dpdy*nz;
	double dpdn_sb=dpdn_sc*sb/sc;
	double sign_sc=-dpdn_sc/(fabs(dpdn_sc)+eps);
	double coeff=sign_sc*un/sqrt(fabs(dpdn_sc)*sc);
	double sign_sb=-dpdn_sb/(fabs(dpdn_sb)+eps);
	double un_sb=fabs(coeff)*sqrt(fabs(dpdn_sb)*sb)*sign_sb;
	/*
	(*Ub).x = ut+un_sb*nx;
	(*Ub).y = vt+un_sb*ny;
	(*Ub).z = wt+un_sb*nz;
	*/

	(*Ub).x = ut + pow(sb/sc,1) * un * nx;
	(*Ub).y = vt + pow(sb/sc,1) * un * ny;
	(*Ub).z = wt + pow(sb/sc,1) * un * nz;	
	
	(*Ub).x += Ua.x;
	(*Ub).y += Ua.y;
	(*Ub).z += Ua.z;

}


void wallmodel_2( double dpdx_b, double dpdy_b, double dpdz_b, double nu, double *nu_t_b, double *nu_t_c, double sb, double sc, Cmpnts *Ub, Cmpnts Uc, Cmpnts Ua, PetscInt bctype, double ks,double nx, double ny, double nz, double *tau_w)
{
	
//        PetscPrintf(PETSC_COMM_WORLD, "HERE 1\n");
	double *u_inner, *z_in, *nut_les, *nut_rans, *f0, *f1, *f2;
	double t1_x, t1_y, t1_z, t2_x, t2_y, t2_z;

	num_innergrid = (int)(log(sc*(dhratio_wm-1)/dh1_wm+1.0)/log(dhratio_wm))+1;

	u_inner= (double *) malloc(num_innergrid*sizeof(double));
	z_in= (double *) malloc(num_innergrid*sizeof(double));
	nut_les= (double *) malloc(num_innergrid*sizeof(double));
	nut_rans= (double *) malloc(num_innergrid*sizeof(double));
	f0= (double *) malloc(num_innergrid*sizeof(double));
	f1= (double *) malloc(num_innergrid*sizeof(double));
	f2= (double *) malloc(num_innergrid*sizeof(double));

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2\n");

	int i,j,k;
        z_in[0] = 0.0;
        for (k = 1; k < num_innergrid; k++) {
                z_in[k] = z_in[k-1] + dh1_wm*pow(dhratio_wm,(double)k);
        }

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-1\n");

	double kappa_rans = 0.4, A = 19; 
	double kappa_les;
	double coeff_force, coeff_c1, f0_sum_c, f1_sum_c, f2_sum_c, u_flux, coeff_c0;

        double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
        double un = u_c * nx + v_c * ny + w_c * nz;
        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );
	double un_c=un;

	sb = sb + 1.e-11;

        t1_x = ut / (ut_mag_c+1.e-11); t1_y = vt / (ut_mag_c+1.e-11); t1_z = wt / (ut_mag_c+1.e-11);

        t2_x = ny*t1_z - nz*t1_y; t2_y = nz*t1_x - nx*t1_z; t2_z = nx*t1_y - ny*t1_x;

        double u_b = (*Ub).x - Ua.x, v_b = (*Ub).y - Ua.y, w_b = (*Ub).z - Ua.z;
        double un_b = u_b * nx + v_b * ny + w_b * nz;
        double ut_mag_b = u_b * t1_x + v_b * t1_y + w_b * t1_z;

        double u_a = Ua.x, v_a = Ua.y, w_a = Ua.z;
        double ut_mag_a = u_a * t1_x + v_a * t1_y + w_a * t1_z;

        double dpdt1_b = dpdx_b * t1_x + dpdy_b * t1_y + dpdz_b * t1_z;
        //dpdx_c * t1_x + dpdy_c * t1_y + dpdz_c * t1_z;

	double dpdt1;
	dpdt1=dpdt1_b;
	
        double yp, yp_c;
	
	double ustar, u_all, up;

	u_flux=2.0*ut_mag_b*sb+ut_mag_c*(sc-2.0*sb);

        if ( ti == tistart) ustar = utau_powerlaw(nu, ut_mag_c, sc); 

	up = pow( fabs(nu*dpdt1), 1.0/3.0 );

	u_all = sqrt( ustar*ustar + up * up);  

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-2\n");
	
	nut_rans[0] = 0.0;
	nut_les[0] = 0.0;
	for (k = 1; k < num_innergrid; k++) {
        	yp = z_in[k] * u_all / nu;
        	double damping = (1.0 - exp(-yp/A)) * (1.0 - exp(-yp/A));
		nut_rans[k] = kappa_rans*u_all*z_in[k]*damping;

		if (nut_rans[k]<1.e-9*nu) nut_rans[k]=0.0;
	}


//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-3\n");
	double sum0, sum1, sum2, dz;
	sum0 = 0.0; sum1 = 0.0; sum2 = 0.0;
	f0[0] = 0.0; f1[0] = 0.0; f2[0] = 0.0;
	for (k = 1; k < num_innergrid; k++) {
		dz = z_in[k] - z_in[k-1];
		sum0 += dz / ( nu + 0.5*(nut_rans[k]+nut_rans[k-1]) );
		sum1 += dz * 0.5 * ( z_in[k] + z_in[k-1] ) / ( nu + 0.5*(nut_rans[k]+nut_rans[k-1]) );
		sum2 += dz*0.5*pow(0.5*(z_in[k]+z_in[k-1]),2)/(nu+0.5*(nut_rans[k]+nut_rans[k-1]));
		f0[k] = sum0; f1[k] = sum1; f2[k] = sum2;	
	}

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-4\n");
	f0_sum_c = 0.0; f1_sum_c = 0.0; // f2_sum_c = 0.0;
        for (k = 1; k < num_innergrid; k++) {
		dz = z_in[k] - z_in[k-1];
		f0_sum_c += dz * 0.5 * ( f0[k] + f0[k-1] );
		f1_sum_c += dz * 0.5 * ( f1[k] + f1[k-1] );
	}

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-5\n");
	if (bctype == 1) {

        	coeff_c1 = ut_mag_c / f0[num_innergrid-1];
        	for (k = 0; k < num_innergrid; k++) {
                	u_inner[k] = ut_mag_a + coeff_c1 * f0[k];
        	}
	}	

	if (bctype == 2) {

        	double r1 = ut_mag_c - dpdt1 * f1[num_innergrid-1];
        	coeff_c1 = r1 / f0[num_innergrid-1];

        	for (k = 0; k < num_innergrid; k++) {
                	u_inner[k] = ut_mag_a + dpdt1 * f1[k] +  coeff_c1 * f0[k];
        	}

	}


//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-6\n");
	if (bctype ==3) {

        	sum2 = 0.0;
        	f2[0] = 0.0;
        	for (k = 1; k < num_innergrid; k++) {
                	dz = z_in[k] - z_in[k-1];
                	sum2 += dz * pow( 0.5 * ( z_in[k] + z_in[k-1] ), 1 ) / ( nu + 0.5*(nut_rans[k]+nut_rans[k-1]) );
                	f2[k] = sum2;
        	}

        	f2_sum_c = 0.0;
        	for (k = 1; k < num_innergrid; k++) {
                	dz = z_in[k] - z_in[k-1];
                	f2_sum_c += dz * 0.5 * ( f2[k] + f2[k-1] );
        	}


        	double r1 = ut_mag_c - dpdt1 * f1[num_innergrid-1], r2 = u_flux - dpdt1 * f1_sum_c;

        	double mat[2][2];

        	mat[0][0] = 0.5 * f2[num_innergrid-1]; mat[0][1] = f0[num_innergrid-1];
        	mat[1][0] = 0.5 * f2_sum_c; mat[1][1] = f0_sum_c;

        	double det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
        	double det1 = r1 * mat[1][1] - r2 * mat[0][1] ;
        	double det2 = mat[0][0] * r2 - mat[1][0] * r1;

      		coeff_force = det1 / det; coeff_c1 = det2 / det;

        	for (k = 0; k < num_innergrid; k++) {
                	u_inner[k] = ut_mag_a + dpdt1 * f1[k] + coeff_force * f2[k] +  coeff_c1 * f0[k];
        	}

	}

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-7\n");
        if (bctype ==4) {

                sum2 = 0.0;
                f2[0] = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        sum2 += dz * pow( 0.5 * ( z_in[k] + z_in[k-1] ), 2 ) / ( nu + 0.5*(nut_rans[k]+nut_rans[k-1]) );
                        f2[k] = sum2;
                }

                f2_sum_c = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        f2_sum_c += dz * 0.5 * ( f2[k] + f2[k-1] );
                }


                double r1 = ut_mag_c - dpdt1 * f1[num_innergrid-1], r2 = u_flux - dpdt1 * f1_sum_c;

                double mat[2][2];

                mat[0][0] = 0.5 * f2[num_innergrid-1]; mat[0][1] = f0[num_innergrid-1];
                mat[1][0] = 0.5 * f2_sum_c; mat[1][1] = f0_sum_c;

                double det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
                double det1 = r1 * mat[1][1] - r2 * mat[0][1] ;
                double det2 = mat[0][0] * r2 - mat[1][0] * r1;

                coeff_force = det1 / det; coeff_c1 = det2 / det;

                for (k = 0; k < num_innergrid; k++) {
                        u_inner[k] = ut_mag_a + dpdt1 * f1[k] + 0.5 * coeff_force * f2[k] +  coeff_c1 * f0[k];
                }


        }

        if (bctype ==5) {

                sum2 = 0.0;
                f2[0] = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        sum2 += dz * pow( 0.5 * ( z_in[k] + z_in[k-1] ), 3 ) / ( nu + 0.5*(nut_rans[k]+nut_rans[k-1]) );
                        f2[k] = sum2;
                }

                f2_sum_c = 0.0;
                for (k = 1; k < num_innergrid; k++) {
                        dz = z_in[k] - z_in[k-1];
                        f2_sum_c += dz * 0.5 * ( f2[k] + f2[k-1] );
                }


                double r1 = ut_mag_c - dpdt1 * f1[num_innergrid-1], r2 = u_flux - dpdt1 * f1_sum_c;
                double mat[2][2];

                mat[0][0] = 0.5 * f2[num_innergrid-1]; mat[0][1] = f0[num_innergrid-1];
                mat[1][0] = 0.5 * f2_sum_c; mat[1][1] = f0_sum_c;

                double det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
                double det1 = r1 * mat[1][1] - r2 * mat[0][1] ;
                double det2 = mat[0][0] * r2 - mat[1][0] * r1;

                coeff_force = det1 / det; coeff_c1 = det2 / det;

                for (k = 0; k < num_innergrid; k++) {
                        u_inner[k] = ut_mag_a + dpdt1 * f1[k] + coeff_force * f2[k]/3.0 +  coeff_c1 * f0[k];
                }


        }

        if (bctype == 6 ) {


		ustar = ut_mag_c / ( log(sc/ks)/kappa_rans + 8.5);


        }

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-8\n");
	if (bctype == 6) {
		double sign=ut_mag_c/(fabs(ut_mag_c)+1.e-11);
		*tau_w=pow(ustar,2)*sign;
	}
	else {
//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 2-8-1\n");
		double dudz = (u_inner[1] - u_inner[0]) / (z_in[1] - z_in[0]);
//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 2-8-2\n");
		*tau_w = nu * dudz; 
//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 2-8-3\n");
	}

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-9\n");
        int k_sb;

        for (k = 1; k < num_innergrid; k++) {
                if (z_in[k-1] < sb && z_in[k] >= sb) k_sb = k;
        }

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 2-10\n");
	*nu_t_b = nut_rans[k_sb];
	*nu_t_c = nut_rans[num_innergrid-1];

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 3\n");
        if (IB_wm) {
	

		sum0 = 0.0;
	        for (k = 1; k < num_innergrid; k++) {
        	        dz = z_in[k] - z_in[k-1];
                	sum0 += dz / ( nu + 0.5*(nut_rans[k]+nut_rans[k-1]) );
                	f0[k] = sum0;    
        	}
	
                coeff_c1 = ut_mag_c / f0[num_innergrid-1];

		double Ub_mag_model = 0.0;

//		Ub_mag_model = ut_mag_a + coeff_c1 * f0[k_sb];

		Ub_mag_model = ut_mag_a + sb * ut_mag_c/sc;
		double un_a = Ua.x*nx+Ua.y*ny+Ua.z*nz;

		ut = t1_x*Ub_mag_model;
		vt = t1_y*Ub_mag_model;
		wt = t1_z*Ub_mag_model;


        	double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
        	double un = u_c * nx + v_c * ny + w_c * nz;

		double sbplus = sb*ustar/nu;
//		if (sbplus>30) {
//	        	(*Ub).x = ut + (un_a+pow(sb/sc,2) * (un-un_a)) * nx;
//		        (*Ub).y = vt + (un_a+pow(sb/sc,2) * (un-un_a)) * ny;
//		        (*Ub).z = wt + (un_a+pow(sb/sc,2) * (un-un_a)) * nz;
//		}else if (sbplus<=30) {
                	(*Ub).x = ut + (un_a+pow(sb/sc,1) * (un-un_a)) * nx;
	                (*Ub).y = vt + (un_a+pow(sb/sc,1) * (un-un_a)) * ny;
        	        (*Ub).z = wt + (un_a+pow(sb/sc,1) * (un-un_a)) * nz;
//		}

//                double u_b = (*Ub).x - Ua.x, v_b = (*Ub).y - Ua.y, w_b = (*Ub).z - Ua.z;
  //              double un = u_b * nx + v_b * ny + w_b * nz;

    //            (*Ub).x = ut + (un_a+un) * nx;
      //          (*Ub).y = vt + (un_a+un) * ny; 
        //        (*Ub).z = wt + (un_a+un) * nz;


        }

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 4\n");


	free(u_inner);
	free(z_in);	
	free(nut_les);
	free(nut_rans);
	free(f0);
	free(f1);
	free(f2);

//        PetscPrintf(PETSC_COMM_WORLD, "HERE 5\n");


}

void tridag(int n, double *a, double *b, double *c, double *r, double *u)
{
        int j;
        double bet;
        double gam[n]; // One vector of workspace, gam, is needed.
        if (b[0] == 0.0) throw("Error 1 in tridag");
        //If this happens, then you should rewrite your equations as a set of order N  1, with u1
        //trivially eliminated.
        u[0]=r[0]/(bet=b[0]);
        for (j=1;j<n;j++) { //Decomposition and forward substitution.
                gam[j]=c[j-1]/bet;
                bet=b[j]-a[j]*gam[j];
                if (bet == 0.0) throw("Error 2 in tridag"); //Algorithm fails; see below.
                u[j]=(r[j]-a[j]*u[j-1])/bet;
        }
        for (j=(n-2);j>=0;j--)
                u[j] -= gam[j+1]*u[j+1]; //Backsubstitution.
}

void innergrid( double *z_in, double h)
{
	int k;
	double f;

//	z_in[0] = 0.0;
  //      for (k = 1; k < num_innergrid; k++) {
//		f = double(k) * 0.5 * pi / ( double(num_innergrid) - 1.0);
//		z_in[k] = h * (1.0 - cos(f));
//	}


        z_in[0] = 0.0;
        for (k = 1; k < num_innergrid; k++) {
                z_in[k] = z_in[k-1] + dh1_wm*pow(dhratio_wm,(double)k);
        }

	
//	printf("the minimum and maximum grid: %le %le \n", z_in[1]-z_in[0], z_in[num_innergrid-1]-z_in[num_innergrid-2]);
	
}

PetscErrorCode Calc_F_wmIB(UserCtx *user, IBMNodes *ibm )
{

  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo     info;
  	int        xs, xe, ys, ye, zs, ze; // Local grid information
  	int        mx, my, mz; // Dimensions in three directions
  	int        i, j, k, l, ibi;
  	int	lxs, lxe, lys, lye, lzs, lze;

  	Cmpnts	***lforce_wm, ***coor, ***csi, ***eta, ***zet;

  	Vec		Coor;
        Cmpnts  ***ucat, ***dp;

        PetscReal ***p;
        PetscReal       ***aj, ***nvert;


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

  	VecSet(user->lForce_wm,0.0);

	
  	//DMDAGetGhostedCoordinates(da, &Coor);

	
  	//DMDAVecGetArray(fda, Coor, &coor);
  	DMDAVecGetArray(fda, user->lForce_wm, &lforce_wm);
  	DMDAVecGetArray(fda, user->lCsi,  &csi);
  	DMDAVecGetArray(fda, user->lEta,  &eta);
  	DMDAVecGetArray(fda, user->lZet,  &zet);
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(fda, user->lUcat,  &ucat);
	DMDAVecGetArray(da, user->lP, &p);

        DMDAVecGetArray(da, user->lNvert, &nvert);



        PetscReal ***lnu_t;
        Cmpnts2 ***K_Omega;

        if(les) {
                DMDAVecGetArray(da, user->lNu_t, &lnu_t);
        }
        else if (rans) {
                DMDAVecGetArray(user->fda2, user->lK_Omega, &K_Omega);
                DMDAVecGetArray(da, user->lNu_t, &lnu_t);
        }


	PetscReal ***level, ***rho, ***mu;	
	if(levelset) {
		DMDAVecGetArray(da, user->lLevelset, &level);
		DMDAVecGetArray(da, user->lDensity, &rho);
		DMDAVecGetArray(da, user->lMu, &mu);
	}
	

        double area, area_1, area_2, area_3;
        double sb, sc, sd;
        double ni[3], nj[3], nk[3], ni_c[3], nj_c[3], nk_c[3];
        double nx, ny, nz, dh_c, dh_e, dh_z;
        PetscInt bctype;
        Cmpnts Ua, Ub, Uc, Ud;
	Cmpnts Ug;


        PetscReal ***ustar;
        DMDAVecGetArray(da, user->lUstar, &ustar);

		
	int iw, ie, js, jn, kb, kf;
        double f_i, f_j, f_k, ren, f_i_c, f_j_c, f_k_c, nu_t_b,  nu_t_c,  nu_t_d; // xyang


//	printf("I am here 1");


	if (IB_wm) {
	int tmp_max=1;
	IBMListNode *current;
	double ni_b[3], nj_b[3], nk_b[3];
	for(int tmp=0; tmp<tmp_max; tmp++) { //tmp_begin
		
		
		for(ibi=0; ibi<NumberOfBodies; ibi++) {
			current = user->ibmlist[ibi].head;
			
			while (current) {
				int i,j,k;
				
				const double ren = user->ren;
				
				IBMInfo *ibminfo = &current->ibm_intp;
				current = current->next;
					
				int ni = ibminfo->cell;
				int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
				int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
				int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
				i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;
					

				/*
				if (i<0 || i>mx-2 || j<0 || j>my-2 || k<0 || k>mz-2 || ip1<0 || ip1>mx-2 || jp1<0 || jp1>my-2 || kp1<0 || kp1>mz-2 || ip2<0 || ip2>mx-2 || jp2<0 || jp2>my-2 || kp2<0 || kp2>mz-2 || ip3<0 || ip3>mx-2 || jp3<0 || jp3>my-2 || kp3<0 || kp3>mz-2) { 
					printf("ni=%d ijk %d %d %d \n", ni, i, j, k);

					printf("ijk1 %d %d %d \n", ip1, jp1, kp1);
					printf("ijk2 %d %d %d \n", ip2, jp2, kp2);
					printf("ijk3 %d %d %d \n", ip3, jp3, kp3);
					printf("=============== \n");
				}
				*/

				double sb = ibminfo->d_s, sc = sb + ibminfo->d_i;
				double sk1  = ibminfo->cr1, sk2 = ibminfo->cr2, sk3 = ibminfo->cr3;
				double cs1 = ibminfo->cs1, cs2 = ibminfo->cs2, cs3 = ibminfo->cs3;
				double nfx = ibm->nf_x[ni], nfy = ibm->nf_y[ni], nfz = ibm->nf_z[ni];
				
							
				if (ni>=0) {
					Ua.x = ibm->u[ibm->nv1[ni]].x * cs1 + ibm->u[ibm->nv2[ni]].x * cs2 + ibm->u[ibm->nv3[ni]].x * cs3;
					Ua.y = ibm->u[ibm->nv1[ni]].y * cs1 + ibm->u[ibm->nv2[ni]].y * cs2 + ibm->u[ibm->nv3[ni]].y * cs3;
					Ua.z = ibm->u[ibm->nv1[ni]].z * cs1 + ibm->u[ibm->nv2[ni]].z * cs2 + ibm->u[ibm->nv3[ni]].z * cs3;
				}
				else {
					Ua.x = Ua.y = Ua.z = 0;
				}
				
				Uc.x = (ucat[kp1][jp1][ip1].x * sk1 + ucat[kp2][jp2][ip2].x * sk2 + ucat[kp3][jp3][ip3].x * sk3);
				Uc.y = (ucat[kp1][jp1][ip1].y * sk1 + ucat[kp2][jp2][ip2].y * sk2 + ucat[kp3][jp3][ip3].y * sk3);
				Uc.z = (ucat[kp1][jp1][ip1].z * sk1 + ucat[kp2][jp2][ip2].z * sk2 + ucat[kp3][jp3][ip3].z * sk3);


				int i1=ip1, j1=jp1, k1=kp1;

	                        double ajc = aj[k1][j1][i1];
        	                double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                	        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        	double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                        	double dpdc, dpde, dpdz;
				double dp_dx1, dp_dy1, dp_dz1;

	                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

        	                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx1, &dp_dy1, &dp_dz1);

                                i1=ip2; j1=jp2; k1=kp2;

                                ajc = aj[k1][j1][i1];
                                csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                                eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                                zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                                double dp_dx2, dp_dy2, dp_dz2;

                                Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx2, &dp_dy2, &dp_dz2);

                                i1=ip3; j1=jp3; k1=kp3;

                                ajc = aj[k1][j1][i1];
                                csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                                eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                                zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                                double dp_dx3, dp_dy3, dp_dz3;

                                Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx3, &dp_dy3, &dp_dz3);

                                double dp_dx = (dp_dx1*sk1 + dp_dx2*sk2 + dp_dx3*sk3 );
                                double dp_dy = (dp_dy1*sk1 + dp_dy2*sk2 + dp_dy3*sk3 );
                                double dp_dz = (dp_dz1*sk1 + dp_dz2*sk2 + dp_dz3*sk3 );

				(nu_t_b) = lnu_t[k][j][i];
				nu_t_c = (lnu_t[kp1][jp1][ip1]*sk1 + lnu_t[kp2][jp2][ip2]*sk2 + lnu_t[kp3][jp3][ip3]*sk3 );

				Ub.x = ucat[k][j][i].x;
				Ub.y = ucat[k][j][i].y;
				Ub.z = ucat[k][j][i].z;

				bctype = IB_wm; 

				double nu = 1./user->ren;
		
				if(levelset) {
					double rho_c = (rho[kp1][jp1][ip1] * sk1 + rho[kp2][jp2][ip2] * sk2 + rho[kp3][jp3][ip3] * sk3);
					double mu_c = (mu[kp1][jp1][ip1] * sk1 + mu[kp2][jp2][ip2] * sk2 + mu[kp3][jp3][ip3] * sk3);
					nu = mu_c/rho_c;
						//nu = mu_water/rho_water;
				}
	

                        	wallmodel_1( dp_dx, dp_dy, dp_dz, nu, &nu_t_b, nu_t_c, sb, sc, 
                                             &ucat[k][j][i], Uc, Ua, bctype, roughness_size,
                                             ibm->nf_x[ni], ibm->nf_y[ni], ibm->nf_z[ni], &f_i, &f_j, &f_k, &f_i_c, &f_j_c, &f_k_c, &ustar[k][j][i], &Ug);

				double fac1=sk1;

                                i1=ip1; j1=jp1; k1=kp1;
                                lforce_wm[k1][j1][i1].x += fac1*(f_i * csi[k1][j1][i1].x + f_j * csi[k1][j1][i1].y  + f_k * csi[k1][j1][i1].z);
                                lforce_wm[k1][j1][i1].y += fac1*(f_i * eta[k1][j1][i1].x + f_j * eta[k1][j1][i1].y  + f_k * eta[k1][j1][i1].z);
                                lforce_wm[k1][j1][i1].z += fac1*(f_i * zet[k1][j1][i1].x + f_j * zet[k1][j1][i1].y  + f_k * zet[k1][j1][i1].z);


                                double fac2=sk2;

                                i1=ip2; j1=jp2; k1=kp2;
                                lforce_wm[k1][j1][i1].x += fac2*(f_i * csi[k1][j1][i1].x + f_j * csi[k1][j1][i1].y  + f_k * csi[k1][j1][i1].z);
                                lforce_wm[k1][j1][i1].y += fac2*(f_i * eta[k1][j1][i1].x + f_j * eta[k1][j1][i1].y  + f_k * eta[k1][j1][i1].z);
                                lforce_wm[k1][j1][i1].z += fac2*(f_i * zet[k1][j1][i1].x + f_j * zet[k1][j1][i1].y  + f_k * zet[k1][j1][i1].z);

                                double fac3=sk3;

                                i1=ip3; j1=jp3; k1=kp3;
                                lforce_wm[k1][j1][i1].x += fac3*(f_i * csi[k1][j1][i1].x + f_j * csi[k1][j1][i1].y  + f_k * csi[k1][j1][i1].z);
                                lforce_wm[k1][j1][i1].y += fac3*(f_i * eta[k1][j1][i1].x + f_j * eta[k1][j1][i1].y  + f_k * eta[k1][j1][i1].z);
                                lforce_wm[k1][j1][i1].z += fac3*(f_i * zet[k1][j1][i1].x + f_j * zet[k1][j1][i1].y  + f_k * zet[k1][j1][i1].z);


				lnu_t[k][j][i] = (nu_t_b);
//				printf("ijk %d %d %d \n", i, j, k);
//				printf("ijk1 %d %d %d \n", ip1, jp1, kp1);
//				printf("ijk2 %d %d %d \n", ip2, jp2, kp2);
//				printf("ijk3 %d %d %d \n", ip3, jp3, kp3);
//				printf("=============== \n");
				
			}
		}

	}	
	}



  	//DMDAVecRestoreArray(fda, Coor, &coor);
  	DMDAVecRestoreArray(fda, user->lForce_wm, &lforce_wm);
  	DMDAVecRestoreArray(fda, user->lCsi,  &csi);
  	DMDAVecRestoreArray(fda, user->lEta,  &eta);
  	DMDAVecRestoreArray(fda, user->lZet,  &zet);
        DMDAVecRestoreArray(da, user->lAj, &aj);
        DMDAVecRestoreArray(fda, user->lUcat,  &ucat);
        DMDAVecRestoreArray(da, user->lP, &p);

        DMDAVecRestoreArray(da, user->lNvert, &nvert);


        if(les) {
                DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);
        }
        else if (rans) {
                DMDAVecRestoreArray(user->fda2, user->lK_Omega, &K_Omega);
                DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);
        }

	if(levelset) {
		DMDAVecRestoreArray(da, user->lLevelset, &level);
		DMDAVecRestoreArray(da, user->lDensity, &rho);
		DMDAVecRestoreArray(da, user->lMu, &mu);
	}
	
        DMDAVecRestoreArray(da, user->lUstar, &ustar);

  	DMLocalToGlobalBegin(fda, user->lForce_wm, INSERT_VALUES, user->Force_wm);
  	DMLocalToGlobalEnd(fda, user->lForce_wm, INSERT_VALUES, user->Force_wm);

	

  	return(0);
}



PetscErrorCode Calc_F_wm(UserCtx *user)
{

  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo     info;
  	int        xs, xe, ys, ye, zs, ze; // Local grid information
  	int        mx, my, mz; // Dimensions in three directions
  	int        i, j, k, l, ibi;
  	int	lxs, lxe, lys, lye, lzs, lze;

  	Cmpnts		***lforce_wm, ***coor, ***csi, ***eta, ***zet;

  	Vec		Coor;
        Cmpnts  	***ucat, ***dp;

        PetscReal 	***p;
        PetscReal       ***aj, ***nvert;

	PetscReal	***tmprt, ***lforce_wmtmprt;

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

//  	VecSet(user->lForce_wm,0.0);

//	if (temperature) {
//		DMDAVecGetArray(da, user->lTmprt, &tmprt);
//  		DMDAVecGetArray(da, user->lForce_wmtmprt, &lforce_wmtmprt);
//	}

  	DMDAGetGhostedCoordinates(da, &Coor);
  	DMDAVecGetArray(fda, Coor, &coor);
  	DMDAVecGetArray(fda, user->lForce_wm, &lforce_wm);
  	DMDAVecGetArray(fda, user->lCsi,  &csi);
  	DMDAVecGetArray(fda, user->lEta,  &eta);
  	DMDAVecGetArray(fda, user->lZet,  &zet);
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(fda, user->lUcat,  &ucat);
	DMDAVecGetArray(da, user->lP, &p);

        DMDAVecGetArray(da, user->lNvert, &nvert);


        PetscReal ***lnu_t;

        if(les) DMDAVecGetArray(da, user->lNu_t, &lnu_t);
        

	PetscReal ***level, ***rho, ***mu;	
	if(levelset) {
		DMDAVecGetArray(da, user->lLevelset, &level);
		DMDAVecGetArray(da, user->lDensity, &rho);
		DMDAVecGetArray(da, user->lMu, &mu);
	}
	

        double area, area_1, area_2, area_3;
        double sb, sc, sd;
        double ni[3], nj[3], nk[3], ni_c[3], nj_c[3], nk_c[3];
        double nx, ny, nz, dh_c, dh_e, dh_z;
        PetscInt bctype;
        Cmpnts Ua, Ub, Uc, Ud;
	Cmpnts Ug;


        PetscReal ***ustar;
        DMDAVecGetArray(da, user->lUstar, &ustar);

	
	int iw, ie, js, jn, kb, kf;
        double f_i, f_j, f_k, ren, f_i_c, f_j_c, f_k_c, nu_t_b,  nu_t_c,  nu_t_d; // xyang

//  	for (k=lzs; k<lze; k++) {
//  	for (j=lys; j<lye; j++) {
//  	for (i=lxs; i<lxe; i++) {
//		lforce_wm[k][j][i].x = 0.0;
//        	lforce_wm[k][j][i].y = 0.0;
//        	lforce_wm[k][j][i].z = 0.0;
//  	}
//  	}
//  	}


  	for (k=lzs; k<lze; k++) {
  	for (j=lys; j<lye; j++) {
  	for (i=lxs; i<lxe; i++) {

                if ( imin_wm != 0 && i == 1 ) {
	                area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );

                        Ua.x = Ua.y = Ua.z = 0;

			

                        sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j][i+1]/area;
                        Uc = ucat[k][j][i+1];

	                sb = 0.5/aj[k][j][i]/area;
                        Ub = ucat[k][j][i];

                        sd = 1.0/aj[k][j][i]/area + 1.0/aj[k][j][i+1]/area + 0.5/aj[k][j][i+2]/area;

                        Ud = ucat[k][j][i+2];

		       	bctype = imin_wm; //user->bctype[0];
			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
			nx =  ni[0], ny =  ni[1], nz =  ni[2];

                        Calculate_normal(csi[k][j][i+1], eta[k][j][i+1], zet[k][j][i+1], ni_c, nj_c, nk_c);


                        ren = user->ren;


			if ( les || rans ) { 
				nu_t_c = lnu_t[k][j][i+1];
				(nu_t_b) = lnu_t[k][j][i];
				nu_t_d = lnu_t[k][j][i+2];
			}

                        int i1 = i+1, j1 = j, k1 = k;
                        double ajc = aj[k1][j1][i1];
                        double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

			double dpdc, dpde, dpdz, dp_dx, dp_dy, dp_dz;

			Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

			Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);

			double nu = 1./user->ren;

	
			if(levelset) {
				nu = mu[k][j][i]/rho[k][j][i];
			}

		
			wallmodel_1( dp_dx, dp_dy, dp_dz, nu, &nu_t_b, nu_t_c, sb, sc, 
                  		     &Ub, Uc, Ua, bctype, roughness_size,
                  		     nx, ny, nz, &f_i, &f_j, &f_k, &f_i_c, &f_j_c, &f_k_c, &ustar[k][j][i], &Ug);

                      	lforce_wm[k][j][i].x += f_i * csi[k][j][i].x + f_j * csi[k][j][i].y  + f_k * csi[k][j][i].z;
                      	lforce_wm[k][j][i].y += f_i * eta[k][j][i].x + f_j * eta[k][j][i].y  + f_k * eta[k][j][i].z;
	                lforce_wm[k][j][i].z += f_i * zet[k][j][i].x + f_j * zet[k][j][i].y  + f_k * zet[k][j][i].z;

			//lnu_t[k][j][i] = (nu_t_b);
                }



                if ( imax_wm != 0 && i == mx - 2 ) {
                        area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );

                        Ua.x = Ua.y = Ua.z = 0;

                        sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j][i-1]/area;
                        Uc = ucat[k][j][i-1];

                        sb = 0.5/aj[k][j][i]/area;
                        Ub = ucat[k][j][i];

                        bctype = imax_wm; //user->bctype[1];
                        Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
                        nx =  -ni[0], ny =  -ni[1], nz =  -ni[2];

                        Calculate_normal(csi[k][j][i-1], eta[k][j][i-1], zet[k][j][i-1], ni_c, nj_c, nk_c);

                        ren = user->ren;

                        sd = 1.0/aj[k][j][i]/area + 1.0/aj[k][j][i-1]/area + 0.5/aj[k][j][i-2]/area;

                        Ud = ucat[k][j][i-2];

                        if ( les || rans ) {
                                nu_t_c = lnu_t[k][j][i-1];
                                (nu_t_b) = lnu_t[k][j][i];
                                nu_t_d = lnu_t[k][j][i-2];
                        }


                        int i1 = i-1, j1 = j, k1 = k;
                        double ajc = aj[k1][j1][i1];
                        double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;


                        double dpdc, dpde, dpdz, dp_dx, dp_dy, dp_dz;

                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                        Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);

			double nu = 1./user->ren;
		
			if(levelset) {
				nu = mu[k][j][i]/rho[k][j][i];
			}
	
	
                        wallmodel_1( dp_dx, dp_dy, dp_dz, nu, &nu_t_b, nu_t_c, sb, sc, 
                                     &Ub, Uc, Ua, bctype, roughness_size,
                                     nx, ny, nz, &f_i, &f_j, &f_k, &f_i_c, &f_j_c, &f_k_c, &ustar[k][j][i], &Ug);

                        lforce_wm[k][j][i].x += f_i * csi[k][j][i].x + f_j * csi[k][j][i].y  + f_k * csi[k][j][i].z;
                       	lforce_wm[k][j][i].y += f_i * eta[k][j][i].x + f_j * eta[k][j][i].y  + f_k * eta[k][j][i].z;
                       	lforce_wm[k][j][i].z += f_i * zet[k][j][i].x + f_j * zet[k][j][i].y  + f_k * zet[k][j][i].z;

			//lnu_t[k][j][i] = (nu_t_b);

                }



                if ( jmin_wm != 0 && j == 1 ) {
                        area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );

			// noslip
                        if (user->bctype[2]==1) Ua.x = Ua.y = Ua.z = 0;

			// idealized water wave && moving frame
                	if (user->bctype[2]==1001 && MoveFrame) {
                        	double k_iww=2.0*M_PI/lamda_iww;
	                        Ua.x=0.0;
        	                Ua.y=C_iww*a_iww*k_iww*sin(k_iww*coor[k][j][i].z);
                	        Ua.z=C_iww*(-1.0+a_iww*k_iww*cos(k_iww*coor[k][j][i].z));
                	}


                        sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j+1][i]/area;
                        Uc = ucat[k][j+1][i];

                        sb = 0.5/aj[k][j][i]/area;
                        Ub = ucat[k][j][i];

                        bctype = jmin_wm; //user->bctype[2];
                        Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
                        nx =  nj[0], ny =  nj[1], nz =  nj[2];

                        Calculate_normal(csi[k][j+1][i], eta[k][j+1][i], zet[k][j+1][i], ni_c, nj_c, nk_c);

                        ren = user->ren;

                        sd = 1.0/aj[k][j][i]/area + 1.0/aj[k][j+1][i]/area + 0.5/aj[k][j+2][i]/area;

                        Ud = ucat[k][j+2][i];

                        if ( les || rans ) {
                                nu_t_c = lnu_t[k][j+1][i];
                                (nu_t_b) = lnu_t[k][j][i];
                                nu_t_d = lnu_t[k][j+2][i];
                        }


                        int i1 = i, j1 = j+1, k1 = k;
                        double ajc = aj[k1][j1][i1];
                        double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;


                        double dpdc, dpde, dpdz, dp_dx=0.0, dp_dy=0.0, dp_dz=0.0;

                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                        Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);

			double nu = 1./user->ren;
		
			if(levelset) {
				nu = mu[k][j][i]/rho[k][j][i];
			}
	
                       	wallmodel_1( dp_dx, dp_dy, dp_dz, nu, &nu_t_b, nu_t_c, sb, sc,
                                     &Ub, Uc, Ua, bctype, roughness_size,
                                     nx, ny, nz, &f_i, &f_j, &f_k, &f_i_c, &f_j_c, &f_k_c, &ustar[k][j][i], &Ug);

                        lforce_wm[k][j][i].x += f_i * csi[k][j][i].x + f_j * csi[k][j][i].y  + f_k * csi[k][j][i].z;
                       	lforce_wm[k][j][i].y += f_i * eta[k][j][i].x + f_j * eta[k][j][i].y  + f_k * eta[k][j][i].z;
                       	lforce_wm[k][j][i].z += f_i * zet[k][j][i].x + f_j * zet[k][j][i].y  + f_k * zet[k][j][i].z;

			//lnu_t[k][j][i] = (nu_t_b);

                }

                if ( jmax_wm != 0 && j == my - 2 ) {

                        area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );

                        Ua.x = Ua.y = Ua.z = 0;

                        sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j-1][i]/area;
                        Uc = ucat[k][j-1][i];

                        sb = 0.5/aj[k][j][i]/area;
                        Ub = ucat[k][j][i];

                        bctype = jmax_wm; //user->bctype[3];
                        Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
                        nx =  -nj[0], ny =  -nj[1], nz =  -nj[2];

                        Calculate_normal(csi[k][j-1][i], eta[k][j-1][i], zet[k][j-1][i], ni_c, nj_c, nk_c);

                        ren = user->ren;

                        sd = 1.0/aj[k][j][i]/area + 1.0/aj[k][j-1][i]/area + 0.5/aj[k][j-2][i]/area;

                        Ud = ucat[k][j-2][i];

                        if ( les || rans ) {
                                nu_t_c = lnu_t[k][j-1][i];
                                (nu_t_b) = lnu_t[k][j][i];
                                nu_t_d = lnu_t[k][j-2][i];
                        }

                        int i1 = i, j1 = j-1, k1 = k;
                        double ajc = aj[k1][j1][i1];
                        double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;


                        double dpdc, dpde, dpdz, dp_dx=0.0, dp_dy=0.0, dp_dz=0.0;

                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                        Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);

			double nu = 1./user->ren;
		
			if(levelset) {
				nu = mu[k][j][i]/rho[k][j][i];
			}
	
	            	wallmodel_1( dp_dx, dp_dy, dp_dz, nu, &nu_t_b, nu_t_c, sb, sc,
                                     &Ub, Uc, Ua, bctype, roughness_size,
                                     nx, ny, nz, &f_i, &f_j, &f_k, &f_i_c, &f_j_c, &f_k_c, &ustar[k][j][i], &Ug);

                        lforce_wm[k][j][i].x += f_i * csi[k][j][i].x + f_j * csi[k][j][i].y  + f_k * csi[k][j][i].z;
                       	lforce_wm[k][j][i].y += f_i * eta[k][j][i].x + f_j * eta[k][j][i].y  + f_k * eta[k][j][i].z;
                       	lforce_wm[k][j][i].z += f_i * zet[k][j][i].x + f_j * zet[k][j][i].y  + f_k * zet[k][j][i].z;

			//lnu_t[k][j][i] = (nu_t_b);

                }


	}
	}
	}


  	DMDAVecRestoreArray(fda, Coor, &coor);
  	DMDAVecRestoreArray(fda, user->lForce_wm, &lforce_wm);
  	DMDAVecRestoreArray(fda, user->lCsi,  &csi);
  	DMDAVecRestoreArray(fda, user->lEta,  &eta);
  	DMDAVecRestoreArray(fda, user->lZet,  &zet);
        DMDAVecRestoreArray(da, user->lAj, &aj);
        DMDAVecRestoreArray(fda, user->lUcat,  &ucat);
        DMDAVecRestoreArray(da, user->lP, &p);

        DMDAVecRestoreArray(da, user->lNvert, &nvert);

//	if (temperature) {
//		DMDAVecRestoreArray(da, user->lTmprt, &tmprt);
//  		DMDAVecRestoreArray(da, user->lForce_wmtmprt, &lforce_wmtmprt);
//	}

        if(les) DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);
        

	if(levelset) {
		DMDAVecRestoreArray(da, user->lLevelset, &level);
		DMDAVecRestoreArray(da, user->lDensity, &rho);
		DMDAVecRestoreArray(da, user->lMu, &mu);
	}
	
        DMDAVecRestoreArray(da, user->lUstar, &ustar);

/*
	if (les) {
        	DMDALocalToLocalBegin(da, user->lNu_t, INSERT_VALUES, user->lNu_t);
	        DMDALocalToLocalEnd(da, user->lNu_t, INSERT_VALUES, user->lNu_t);
	}

	if (les) {

        	DMDAVecGetArray(da, user->lNu_t, &lnu_t);
        	for(k=zs; k<ze; k++)
	        for(j=ys; j<ye; j++)
        	for(i=xs; i<xe; i++) {
                	int flag=0, a=i, b=j, c=k;

	                if(i_periodic && i==0) a=mx-2, flag=1;
        	        else if(i_periodic && i==mx-1) a=1, flag=1;

                	if(j_periodic && j==0) b=my-2, flag=1;
	                else if(j_periodic && j==my-1) b=1, flag=1;

        	        if(k_periodic && k==0) c=mz-2, flag=1;
                	else if(k_periodic && k==mz-1) c=1, flag=1;

	
        	        if(ii_periodic && i==0) a=-2, flag=1;
                	else if(ii_periodic && i==mx-1) a=mx+1, flag=1;

	                if(jj_periodic && j==0) b=-2, flag=1;
        	        else if(jj_periodic && j==my-1) b=my+1, flag=1;

                	if(kk_periodic && k==0) c=-2, flag=1;
	                else if(kk_periodic && k==mz-1) c=mz+1, flag=1;


        	        if(flag) {
                	        lnu_t[k][j][i] = lnu_t[c][b][a];
	                }
        	}
        	DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);

	}

*/
	
        DMDALocalToLocalBegin(fda, user->lForce_wm, INSERT_VALUES, user->lForce_wm);
	DMDALocalToLocalEnd(fda, user->lForce_wm, INSERT_VALUES, user->lForce_wm);
	

        DMDAVecGetArray(fda, user->lForce_wm, &lforce_wm);
        for(k=zs; k<ze; k++)
        for(j=ys; j<ye; j++)
        for(i=xs; i<xe; i++) {
                int flag=0, a=i, b=j, c=k;

	        if(i_periodic && i==0) a=mx-2, flag=1;
                else if(i_periodic && i==mx-1) a=1, flag=1;

               	if(j_periodic && j==0) b=my-2, flag=1;
 	        else if(j_periodic && j==my-1) b=1, flag=1;

                if(k_periodic && k==0) c=mz-2, flag=1;
                else if(k_periodic && k==mz-1) c=1, flag=1;

	
        	if(ii_periodic && i==0) a=-2, flag=1;
                else if(ii_periodic && i==mx-1) a=mx+1, flag=1;

	        if(jj_periodic && j==0) b=-2, flag=1;
        	else if(jj_periodic && j==my-1) b=my+1, flag=1;

                if(kk_periodic && k==0) c=-2, flag=1;
	        else if(kk_periodic && k==mz-1) c=mz+1, flag=1;

        	if(flag) {
                        lforce_wm[k][j][i].x = lforce_wm[c][b][a].x;
                        lforce_wm[k][j][i].y = lforce_wm[c][b][a].y;
                        lforce_wm[k][j][i].z = lforce_wm[c][b][a].z;
	        }
        }
        DMDAVecRestoreArray(fda, user->lForce_wm, &lforce_wm);


  	DMLocalToGlobalBegin(fda, user->lForce_wm, INSERT_VALUES, user->Force_wm);
  	DMLocalToGlobalEnd(fda, user->lForce_wm, INSERT_VALUES, user->Force_wm);


  	return(0);
}




PetscErrorCode TECIOOut_F_wm(UserCtx *user)	
{


	int IMax, JMax, KMax;
        PetscReal x, y, z, F_wm_x, F_wm_y, F_wm_z;
	char filen[80];
	sprintf(filen, "F_wm.plt");


        FILE * pFile;
        pFile = fopen(filen, "w");


        fprintf(pFile, " TITLE = \"wall model force\" \n");
        fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"F_wm_x\", \"F_wm_y\", \"F_wm_z\" \n" );

        DM              da = user->da, fda = user->fda;
        DMDALocalInfo     info;

        DMDAGetLocalInfo(da, &info);

	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
    
	int	lxs, lys, lzs, lxe, lye, lze;
	int	i, j, k;
	Cmpnts		***force_wm,***coor;
	Vec		Coor;
		
	IMax = mx-2;
	JMax = my-2;
	KMax = mz-2;

	fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax, JMax, KMax );
		
	DMDAGetCoordinates(da, &Coor);
	DMDAVecGetArray(fda, Coor, &coor);
        DMDAVecGetArray(fda, user->Force_wm, &force_wm);

  	for (k=zs+1; k<ze-1; k++){
        for (j=ys+1; j<ye-1; j++){
        for (i=xs+1; i<xe-1; i++){
		x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z; 
		F_wm_x = force_wm[k][j][i].x; F_wm_y = force_wm[k][j][i].y; F_wm_z = force_wm[k][j][i].z; 
		fprintf(pFile, "%le %le %le %le %le %le \n", x, y, z, F_wm_x, F_wm_y, F_wm_z );

	}
	}
	}

	fclose(pFile);

        DMDAVecRestoreArray(fda, Coor, &coor);		
        DMDAVecRestoreArray(fda, user->Force_wm, &force_wm);



	return 0;
}


PetscErrorCode TECIOOut_rhs(UserCtx *user, Vec Rhs, char fname[80])	
{


	int IMax, JMax, KMax;
        PetscReal x, y, z, F_wm_x, F_wm_y, F_wm_z;

        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	char filen[160];
	// sprintf(filen, "rhs%03d.plt", rank);

	sprintf(filen, "%s%03d.plt", fname, rank);

        FILE * pFile;
        pFile = fopen(filen, "w");


        fprintf(pFile, " TITLE = \"rhs\" \n");
        fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"rhs_x\", \"rhs_y\", \"rhs_z\" \n" );

        DM              da = user->da, fda = user->fda;
        DMDALocalInfo     info;

        DMDAGetLocalInfo(da, &info);

	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
    
	int	lxs, lys, lzs, lxe, lye, lze;
	int	i, j, k;
	Cmpnts		***coor;
	Cmpnts		***rhs;
	Vec		Coor;
		
	IMax = mx-2;
	JMax = my-2;
	KMax = mz-2;

	fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", xe-xs-1, ye-ys-1, ze-zs-1 );
		
	DMDAGetCoordinates(da, &Coor);
	DMDAVecGetArray(fda, Coor, &coor);

	DMDAVecGetArray(fda, Rhs,  &rhs);


  	for (k=zs; k<ze-1; k++){
        for (j=ys; j<ye-1; j++){
        for (i=xs; i<xe-1; i++){
		x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
 		fprintf(pFile, "%le %le %le %le %le %le \n", x, y, z, rhs[k][j][i].x, rhs[k][j][i].y, rhs[k][j][i].z );

	}
	}
	}

	fclose(pFile);

        DMDAVecRestoreArray(fda, Coor, &coor);		

	DMDAVecRestoreArray(fda, Rhs,  &rhs);

	return 0;
}



PetscErrorCode TECIOOut_rhs_da(UserCtx *user, Vec Rhs, char fname[80])	
{


	int IMax, JMax, KMax;
        PetscReal x, y, z, F_wm_x, F_wm_y, F_wm_z;

        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	char filen[160];
	sprintf(filen, "%s%03d.plt", fname, rank);


        FILE * pFile;
        pFile = fopen(filen, "w");


        fprintf(pFile, " TITLE = \"rhs\" \n");
        fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"rhs\" \n" );

        DM              da = user->da, fda = user->fda;
        DMDALocalInfo     info;

        DMDAGetLocalInfo(da, &info);

	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
    
	int	lxs, lys, lzs, lxe, lye, lze;
	int	i, j, k;
	Cmpnts		***coor;
	PetscReal	***rhs;
	Vec		Coor;
		
	IMax = mx-2;
	JMax = my-2;
	KMax = mz-2;

	fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", xe-xs-1, ye-ys-1, ze-zs-1 );
		
	DMDAGetCoordinates(da, &Coor);
	DMDAVecGetArray(fda, Coor, &coor);

	DMDAVecGetArray(da, Rhs,  &rhs);


  	for (k=zs; k<ze-1; k++){
        for (j=ys; j<ye-1; j++){
        for (i=xs; i<xe-1; i++){
		x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
 		fprintf(pFile, "%le %le %le %le \n", x, y, z, rhs[k][j][i]);

	}
	}
	}

	fclose(pFile);

        DMDAVecRestoreArray(fda, Coor, &coor);		

	DMDAVecRestoreArray(da, Rhs,  &rhs);

	return 0;
}


// assume the k-face direction is the same as the main stream wall tangent direction
PetscErrorCode force_jwall(UserCtx *user)	
{


	int IMax, JMax, KMax;
        PetscReal *p_isum1, *friction_isum1, *p_isum2, *friction_isum2;

        DM              da = user->da, fda = user->fda;
        DMDALocalInfo     info;

        DMDAGetLocalInfo(da, &info);

	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
    
	int	lxs, lys, lzs, lxe, lye, lze;
	int	i, j, k;
	Cmpnts		***force_wm,***coor;
	Vec		Coor;
		
	IMax = mx-2;
	JMax = my-2;
	KMax = mz-2;





	if (ti == tistart+1) {

  		PetscMalloc(mz*sizeof(PetscReal), &(user->p_jmin));
  		PetscMalloc(mz*sizeof(PetscReal), &(user->p_jmax));
  		PetscMalloc(mz*sizeof(PetscReal), &(user->friction_jmin));
  		PetscMalloc(mz*sizeof(PetscReal), &(user->friction_jmax));

  		for(k=0; k<mz; k++) {
	
        		user->p_jmin[k] = 0.0;
        		user->p_jmax[k] = 0.0;
        		user->friction_jmin[k] = 0.0;
        		user->friction_jmax[k] = 0.0;
  		}
	}

        PetscMalloc(mz*sizeof(PetscReal), &(p_isum1));
        PetscMalloc(mz*sizeof(PetscReal), &(friction_isum1));
        PetscMalloc(mz*sizeof(PetscReal), &(p_isum2));
        PetscMalloc(mz*sizeof(PetscReal), &(friction_isum2));

		
	DMDAGetCoordinates(da, &Coor);
	DMDAVecGetArray(fda, Coor, &coor);

//	PetscScalar	***p;
	PetscReal	***p;
        DMDAVecGetArray(da, user->lP, &p);

        PetscReal ***ustar;
        DMDAVecGetArray(da, user->lUstar, &ustar);

        PetscReal ***tau;
        DMDAVecGetArray(da, user->lTau, &tau);


        Cmpnts  	***csi, ***eta, ***zet;
        Cmpnts  ***ucat;
        PetscReal       ***aj;

        DMDAVecGetArray(fda, user->lCsi,  &csi);
        DMDAVecGetArray(fda, user->lEta,  &eta);
        DMDAVecGetArray(fda, user->lZet,  &zet);
        DMDAVecGetArray(da, user->lAj, &aj);
        DMDAVecGetArray(fda, user->lUcat,  &ucat);


        for (k=0; k<mz; k++){
		p_isum1[k] = 0.0;
		friction_isum1[k] = 0.0;
	        p_isum2[k] = 0.0;
                friction_isum2[k] = 0.0;
	
	}


        double ren, area, area_1, area_2, area_3;
        double sb, sc;
        double ni[3], nj[3], nk[3];
        double nx, ny, nz;
        PetscInt bctype;
        Cmpnts Ua, Ub, Uc;

        double t1_x, t1_y, t1_z, t2_x, t2_y, t2_z;

        double u_c, v_c, w_c;
        double un;
        double ut, vt, wt;
        double ut_mag_c;

        double u_b, v_b, w_b;
        double ut_mag_b;

        double u_a, v_a, w_a;
        double ut_mag_a;

	double f_i, f_j, f_k;

	PetscReal	fac = 1.0 / (double)mx; 

  	for (k=zs; k<ze; k++){
        for (j=ys; j<ye; j++){
        for (i=xs; i<xe; i++){

                if (j==my-2 || j==1) {

                        area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );

                        Ua.x = Ua.y = Ua.z = 0;

	                if (j==1 && user->bctype[2]==1001 && MoveFrame) {
        	                double k_iww=2.0*M_PI/lamda_iww;
                	        Ua.x=0.0;
                        	Ua.y=C_iww*a_iww*k_iww*sin(k_iww*coor[k][j][i].z);
	                        Ua.z=C_iww*(-1.0+a_iww*k_iww*cos(k_iww*coor[k][j][i].z));
                	}

                	if (j==my-2 && user->bctype[3]==1002 && MoveFrame) {
                        	Ua.x=0.0;
	                        Ua.y=0.0;
        	                Ua.z=(1.0-C_iww);
	                }
        	        //

                        if (j==my-2) sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j-1][i]/area;
                        if (j==my-2) Uc = ucat[k][j-1][i];

                        if (j==1) sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j+1][i]/area;
                        if (j==1) Uc = ucat[k][j+1][i];

                        sb = 0.5/aj[k][j][i]/area;
                        Ub = ucat[k][j][i];

                        bctype = jmin_wm; //user->bctype[2];
                        Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
                        if (j==1) nx =  nj[0], ny =  nj[1], nz =  nj[2];
                        if (j==my-2) nx =  -nj[0], ny =  -nj[1], nz =  -nj[2];

                        ren = user->ren;

//        		t2_x = ny*t1_z - nz*t1_y; t2_y = nz*t1_x - nx*t1_z; t2_z = nx*t1_y - ny*t1_x;
	                u_c = Uc.x - Ua.x; v_c = Uc.y - Ua.y; w_c = Uc.z - Ua.z;
        	        un = u_c * nx + v_c * ny + w_c * nz;
              	        ut = u_c - un * nx; vt = v_c - un * ny; wt = w_c - un * nz;
                       	ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

	                t1_x = ut / ut_mag_c; t1_y = vt / ut_mag_c; t1_z = wt / ut_mag_c;

//        		u_b = Ub.x - Ua.x; v_b = Ub.y - Ua.y; w_b = Ub.z - Ua.z;
//        		ut_mag_b = u_b * t1_x + v_b * t1_y + w_b * t1_z;

        		u_b = Ub.x - Ua.x; v_b = Ub.y - Ua.y; w_b = Ub.z - Ua.z;
                        un = u_b * nx + v_b * ny + w_b * nz;
                        ut = u_b - un * nx; vt = v_b - un * ny; wt = w_b - un * nz;
                        ut_mag_b = sqrt( ut*ut + vt*vt + wt*wt );

	        	double t1_x_b = ut / ut_mag_b, t1_y_b = vt / ut_mag_b, t1_z_b = wt / ut_mag_b;

			if (j==1) {
				if (jmin_wm != 0) {
					f_i = tau[k][j][i]*t1_x;
					f_j = tau[k][j][i]*t1_y;
					f_k = tau[k][j][i]*t1_z;
				}

				if (jmin_wm == 0) {
					f_i = ut_mag_b*t1_x_b/sb/ren;
                                	f_j = ut_mag_b*t1_y_b/sb/ren;
                                	f_k = ut_mag_b*t1_z_b/sb/ren;
				}
                        	p_isum1[k] += p[k][j][i] * fac;
                        	friction_isum1[k] += (f_i * nk[0] + f_j * nk[1] + f_k * nk[2]) * fac;
			}
	
                        if (j==my-2) {
                                if (jmax_wm != 0) {
                                        f_i = tau[k][j][i]*t1_x;
                                        f_j = tau[k][j][i]*t1_y;
                                        f_k = tau[k][j][i]*t1_z;
                                }

                                if (jmax_wm == 0) {
                                        f_i = ut_mag_b*t1_x_b/sb/ren;
                                        f_j = ut_mag_b*t1_y_b/sb/ren;
                                        f_k = ut_mag_b*t1_z_b/sb/ren;
                                }
                                p_isum2[k] += p[k][j][i] * fac;
                                friction_isum2[k] += (f_i * nk[0] + f_j * nk[1] + f_k * nk[2]) * fac;

                        }
                }

	}
	}
	}


	PetscReal psum, fsum;
        for (k=0; k<mz; k++){
//      		PetscGlobalSum(&(p_isum1[k]), &psum, PETSC_COMM_WORLD);
//      		PetscGlobalSum(&(friction_isum1[k]), &fsum, PETSC_COMM_WORLD);

                MPI_Allreduce (&(p_isum1[k]), &psum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
                MPI_Allreduce (&(friction_isum1[k]), &fsum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

                user->p_jmin[k] += psum;
                user->friction_jmin[k] += fsum;

//                PetscGlobalSum(&(p_isum2[k]), &psum, PETSC_COMM_WORLD);
//                PetscGlobalSum(&(friction_isum2[k]), &fsum, PETSC_COMM_WORLD);

               	MPI_Allreduce (&(p_isum2[k]), &psum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
                MPI_Allreduce (&(friction_isum2[k]), &fsum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);


                user->p_jmax[k] += psum;
                user->friction_jmax[k] += fsum;

        }

        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


	char filen[80];
        FILE * pFile;

	i = xs;
	j = 0;

	if (ys==0) {
        	sprintf(filen, "force_lowall%03d.plt", zs);
        	pFile = fopen(filen, "w");

		fprintf(pFile, " Variables = \"z\", \"p\", \"f\", \"nx_p\", \"ny_p\", \"nz_p\", \"nx_f\", \"ny_f\", \"nz_f\", \"dh_k\", \"dh_j\"  \n");

        	for (k=zs+1; k<ze-1; k++){
                        double area_k = sqrt( zet[k][j+1][i+1].x*zet[k][j+1][i+1].x + zet[k][j+1][i+1].y*zet[k][j+1][i+1].y + zet[k][j+1][i+1].z*zet[k][j+1][i+1].z );
			double dh_k = 1.0/aj[k][j+1][i+1]/area_k;
                        double area_j = sqrt( eta[k][j+1][i+1].x*eta[k][j+1][i+1].x + eta[k][j+1][i+1].y*eta[k][j+1][i+1].y + eta[k][j+1][i+1].z*eta[k][j+1][i+1].z );
			double dh_j = 1.0/aj[k][j+1][i+1]/area_j;
                        Calculate_normal(csi[k][j+1][i+1], eta[k][j+1][i+1], zet[k][j+1][i+1], ni, nj, nk);
			fprintf(pFile, "%le %le %le %le %le %le %le %le %le %le %le \n", coor[k][j][i].z, user->p_jmin[k]/( (double)(ti-tistart) ), user->friction_jmin[k]/( (double)(ti-tistart) ), nj[0], nj[1], nj[2], nk[0], nk[1], nk[2], dh_k, dh_j );
		}
        	fclose(pFile);

	}

        i = xs;
        j = my-2;

        if (ye==my) {
                sprintf(filen, "force_upwall%03d.plt", zs);
                pFile = fopen(filen, "w");

                fprintf(pFile, " Variables = \"z\", \"p\", \"f\", \"nx_p\", \"ny_p\", \"nz_p\", \"nx_f\", \"ny_f\", \"nz_f\", \"dh_k\", \"dh_j\"  \n");

                for (k=zs+1; k<ze-1; k++){
                        double area_k = sqrt( zet[k][j][i+1].x*zet[k][j][i+1].x + zet[k][j][i+1].y*zet[k][j][i+1].y + zet[k][j][i+1].z*zet[k][j][i+1].z );
                        double dh_k = 1.0/aj[k][j][i+1]/area_k;
                        double area_j = sqrt( eta[k][j][i+1].x*eta[k][j][i+1].x + eta[k][j][i+1].y*eta[k][j][i+1].y + eta[k][j][i+1].z*eta[k][j][i+1].z );
                        double dh_j = 1.0/aj[k][j+1][i+1]/area_j;
                        Calculate_normal(csi[k][j][i+1], eta[k][j][i+1], zet[k][j][i+1], ni, nj, nk);
                        fprintf(pFile, "%le %le %le %le %le %le %le %le %le %le %le \n", coor[k][j][i].z, user->p_jmin[k]/( (double)(ti-tistart) ), user->friction_jmax[k]/( (double)(ti-tistart) ), nj[0], nj[1], nj[2], nk[0], nk[1], nk[2], dh_k, dh_j );
                }

               	fclose(pFile);
        }

	
        PetscFree((p_isum1));
        PetscFree((friction_isum1));
        PetscFree((p_isum2));
        PetscFree((friction_isum2));

	if (ti == (tistart + tisteps -1)) {
  		PetscFree((user->p_jmin));
  		PetscFree((user->p_jmax));
  		PetscFree((user->friction_jmin));
  		PetscFree((user->friction_jmax));

	}

        DMDAVecRestoreArray(fda, Coor, &coor);		
        DMDAVecRestoreArray(da, user->lP, &p);
        DMDAVecRestoreArray(da, user->lUstar, &ustar);
        DMDAVecRestoreArray(da, user->lTau, &tau);

        DMDAVecRestoreArray(fda, user->lCsi,  &csi);
        DMDAVecRestoreArray(fda, user->lEta,  &eta);
        DMDAVecRestoreArray(fda, user->lZet,  &zet);
        DMDAVecRestoreArray(da, user->lAj, &aj);
        DMDAVecRestoreArray(fda, user->lUcat,  &ucat);

	return 0;
}




PetscErrorCode Visc_wm(UserCtx *user)
{
	Vec	Csi = user->lCsi, Eta = user->lEta, Zet = user->lZet;
	Vec 	Ucat=user->lUcat;
	
	Cmpnts	***ucat, ***lucat_o;

	Cmpnts	***csi, ***eta, ***zet;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;

	PetscReal	***nvert;

	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	
	PetscReal	***aj, ***iaj, ***jaj, ***kaj;//, ***vol;

	int	lxs, lxe, lys, lye, lzs, lze;

	PetscReal	dudc, dude, dudz, dvdc, dvde, dvdz, dwdc, dwde, dwdz;
	PetscReal	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2;
	PetscReal	g11, g21, g31;
	PetscReal	r11, r21, r31, r12, r22, r32, r13, r23, r33;

        PetscReal dudc_wm, dvdc_wm, dwdc_wm, dude_wm, dvde_wm, dwde_wm, dudz_wm, dvdz_wm, dwdz_wm;
        PetscReal r11_wm, r21_wm, r31_wm, r12_wm, r22_wm, r32_wm, r13_wm, r23_wm, r33_wm;

	Vec Coor;
	Cmpnts ***coor;

	PetscReal ***p;
	PetscScalar	solid;

	solid = 0.5;

  	int ibi;

        double area, nu_t;
        double sb, sc, st;
        double nx, ny, nz;
        PetscInt bctype;
        Cmpnts Ua, Ub, Uc;
	Cmpnts Ug;

        double ren, nu_t_b,  nu_t_c;

        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscReal ***level, ***rho, ***mu;
        PetscReal ***ustar;
        PetscReal ***tau;
	PetscReal ***lnu_t;
	Cmpnts ***visc1_wm, ***visc2_wm, ***visc3_wm;

	double tau_w;

	DMDAGetLocalInfo(da, &info);
	mx = info.mx; my = info.my; mz = info.mz;
	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;

  /* First we calculate the flux on cell surfaces. Stored on the upper integer
     node. For example, along i direction, the flux are stored at node 0:mx-2*/
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;

	VecSet(user->lVisc1_wm, 0.0);
	VecSet(user->lVisc2_wm, 0.0);
	VecSet(user->lVisc3_wm, 0.0);

	DMDAVecGetArray(fda, user->lVisc1_wm, &visc1_wm);
	DMDAVecGetArray(fda, user->lVisc2_wm, &visc2_wm);
	DMDAVecGetArray(fda, user->lVisc3_wm, &visc3_wm);
	
	DMDAVecGetArray(da, user->lNu_t, &lnu_t);
	

 	DMDAGetGhostedCoordinates(da, &Coor);
	DMDAVecGetArray(fda, Coor, &coor);

	DMDAVecGetArray(fda, user->lUcat,  &ucat);
	DMDAVecGetArray(da, user->lP, &p);

	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);

	DMDAVecGetArray(fda, user->lICsi, &icsi);
	DMDAVecGetArray(fda, user->lIEta, &ieta);
	DMDAVecGetArray(fda, user->lIZet, &izet);

	DMDAVecGetArray(fda, user->lJCsi, &jcsi);
	DMDAVecGetArray(fda, user->lJEta, &jeta);
	DMDAVecGetArray(fda, user->lJZet, &jzet);

	DMDAVecGetArray(fda, user->lKCsi, &kcsi);
	DMDAVecGetArray(fda, user->lKEta, &keta);
	DMDAVecGetArray(fda, user->lKZet, &kzet);

	DMDAVecGetArray(da, user->lNvert, &nvert);
	
	DMDAVecGetArray(da, user->lAj, &aj);
   
        DMDAVecGetArray(da, user->lUstar, &ustar);
    
        DMDAVecGetArray(da, user->lTau, &tau);
    
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);
	
	if(levelset) {
		DMDAVecGetArray(da, user->lLevelset, &level);
		DMDAVecGetArray(da, user->lDensity, &rho);
		DMDAVecGetArray(da, user->lMu, &mu);
	}


	
	if (immersed && IB_wm) { 
		
		IBMListNode *current;
		for(int ibi=0; ibi<NumberOfBodies; ibi++) {
			current = user->ibmlist[ibi].head;
			
			IBMNodes *ibm = &ibm_ptr[ibi];
			std::vector<double> count;

			while (current) {

				int i,j,k;
				
				const double ren = user->ren;
				
				IBMInfo *ibminfo = &current->ibm_intp;
				current = current->next;

                                int ni = ibminfo->cell;
                                int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
                                int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
                                int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
                                i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;

                                sb = ibminfo->d_s; sc = sb + ibminfo->d_i;
                                double sk1  = ibminfo->cr1, sk2 = ibminfo->cr2, sk3 = ibminfo->cr3;
                                double cs1 = ibminfo->cs1, cs2 = ibminfo->cs2, cs3 = ibminfo->cs3;
				double nx = ibm->nf_x[ni], ny = ibm->nf_y[ni], nz = ibm->nf_z[ni];
							
				if (ni>=0) {
					Ua.x = ibm->u[ibm->nv1[ni]].x * cs1 + ibm->u[ibm->nv2[ni]].x * cs2 + ibm->u[ibm->nv3[ni]].x * cs3;
					Ua.y = ibm->u[ibm->nv1[ni]].y * cs1 + ibm->u[ibm->nv2[ni]].y * cs2 + ibm->u[ibm->nv3[ni]].y * cs3;
					Ua.z = ibm->u[ibm->nv1[ni]].z * cs1 + ibm->u[ibm->nv2[ni]].z * cs2 + ibm->u[ibm->nv3[ni]].z * cs3;
				}
				else {
					Ua.x = Ua.y = Ua.z = 0;
				}
				
				Uc.x = (ucat[kp1][jp1][ip1].x * sk1 + ucat[kp2][jp2][ip2].x * sk2 + ucat[kp3][jp3][ip3].x * sk3);
				Uc.y = (ucat[kp1][jp1][ip1].y * sk1 + ucat[kp2][jp2][ip2].y * sk2 + ucat[kp3][jp3][ip3].y * sk3);
				Uc.z = (ucat[kp1][jp1][ip1].z * sk1 + ucat[kp2][jp2][ip2].z * sk2 + ucat[kp3][jp3][ip3].z * sk3);

				int i1, j1, k1;

				double ajc;
				double csi0, csi1, csi2;
				double eta0, eta1, eta2;
				double zet0, zet1, zet2;
	
				i1=ip1; j1=jp1; k1=kp1;

	                        ajc = aj[k1][j1][i1];
        	                csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                	        eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        	zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;


                        	double dpdc, dpde, dpdz;
				double dp_dx1, dp_dy1, dp_dz1;

	                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

        	                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx1, &dp_dy1, &dp_dz1);

                                i1=ip2; j1=jp2; k1=kp2;

                                ajc = aj[k1][j1][i1];
                                csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                                eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                                zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                                double dp_dx2, dp_dy2, dp_dz2;

                                Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx2, &dp_dy2, &dp_dz2);

                                i1=ip3; j1=jp3; k1=kp3;

                                ajc = aj[k1][j1][i1];
                                csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                                eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                                zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                                double dp_dx3, dp_dy3, dp_dz3;

                                Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx3, &dp_dy3, &dp_dz3);

                                double dp_dx = (dp_dx1*sk1 + dp_dx2*sk2 + dp_dx3*sk3 );
                                double dp_dy = (dp_dy1*sk1 + dp_dy2*sk2 + dp_dy3*sk3 );
                                double dp_dz = (dp_dz1*sk1 + dp_dz2*sk2 + dp_dz3*sk3 );


				nu_t_b = lnu_t[k][j][i];
				nu_t_c = lnu_t[kp1][jp1][ip1]*sk1 + lnu_t[kp2][jp2][ip2]*sk2 + lnu_t[kp3][jp3][ip3]*sk3;

				Ub.x = ucat[k][j][i].x;
				Ub.y = ucat[k][j][i].y;
				Ub.z = ucat[k][j][i].z;

        			double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
			        double un = u_c * nx + v_c * ny + w_c * nz;
			        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
			        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

			        double t1x = ut / (ut_mag_c+1.e-11), t1y = vt / (ut_mag_c+1.e-11), t1z = wt / (ut_mag_c+1.e-11);

			        double t2x = ny*t1z - nz*t1y, t2y = nz*t1x - nx*t1z, t2z = nx*t1y - ny*t1x;

				bctype = IB_wm; 
				
				double nu = 1./user->ren;
		
				if(levelset) {
					double rho_c = (rho[kp1][jp1][ip1] * sk1 + rho[kp2][jp2][ip2] * sk2 + rho[kp3][jp3][ip3] * sk3);
					double mu_c = (mu[kp1][jp1][ip1] * sk1 + mu[kp2][jp2][ip2] * sk2 + mu[kp3][jp3][ip3] * sk3);
					nu = mu_c/rho_c;
				}
	
				double nut_b, nut_c, tau_w, nut_2sb;

				wallmodel_s(nu,sb,sc,Uc,&ucat[k][j][i],Ua,bctype,roughness_size,nx,ny,nz,&tau_w,&ustar[k][j][i],dp_dx,dp_dy,dp_dz,&nut_2sb,nu_t_c);

				//PetscPrintf(PETSC_COMM_WORLD, "ustar=%le\n", ustar[k][j][i]);
				// wallmodel_2( dp_dx,dp_dy,dp_dz,nu,&nut_b,&nut_c,sb,sc,&ucat[k][j][i],Uc,Ua,bctype,roughness_size,nx,ny,nz,&tau_w);

				// lnu_t[k][j][i]= nut_b;  //xiaolei

				tau[k][j][i]=(tau_w);

                                //ucat[k][j][i].x = Ub.x;
                                //ucat[k][j][i].y = Ub.y;
                                //ucat[k][j][i].z = Ub.z;

                                double nx_1=nx, ny_1=ny, nz_1=nz;
                                double t1x_1=t1x, t1y_1=t1y, t1z_1=t1z;
                                double t2x_1=t2x, t2y_1=t2y, t2z_1=t2z;


               			double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;

				double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;		

                                double dut1dn_wm;
				double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;


				i1=i; j1=j; k1=k;		

				int is=i-1;
				int ie=i+1;

        			//printf("Here TEST wallmodel III i=%d j=%d, k=%d\n", i1, j1, k1);
//				for (k1=k-1;k1<k+1;k1++)
//				for (j1=j-1;j1<j+1;j1++)
				for (i1=is;i1<ie;i1++){
				if ( (nvert[k1][j1][i1]<0.1 && nvert[k1][j1][i1+1]>0.1) || (nvert[k1][j1][i1]>0.1 && nvert[k1][j1][i1+1]<0.1)) {
					ajc = iaj[k1][j1][i1];
					csi0 = icsi[k1][j1][i1].x, csi1 = icsi[k1][j1][i1].y, csi2 = icsi[k1][j1][i1].z;
					eta0 = ieta[k1][j1][i1].x, eta1 = ieta[k1][j1][i1].y, eta2 = ieta[k1][j1][i1].z;
					zet0 = izet[k1][j1][i1].x, zet1 = izet[k1][j1][i1].y, zet2 = izet[k1][j1][i1].z;

//                        		area = sqrt( icsi[k1][j1][i1].x*icsi[k1][j1][i1].x + icsi[k1][j1][i1].y*icsi[k1][j1][i1].y + icsi[k1][j1][i1].z*icsi[k1][j1][i1].z );

					Compute1_du_i (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

			                Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

					Comput_du_wmlocal(nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2); 

                                        if ( i1==0 || nvert[k1][j1][i1]>0.1 ) {
						//lnu_t[k1][j1][i1+1] = nu_t_b;
						nu_t = lnu_t[k1][j1][i1+1];
					}
                                        else if( i1==mx-2 || nvert[k1][j1][i1+1]>0.1 ) {
						//lnu_t[k1][j1][i1] = nu_t_b;
						nu_t = lnu_t[k1][j1][i1];

					}
                                        else nu_t = 0.5 * (lnu_t[k1][j1][i1] + lnu_t[k1][j1][i1+1]);


                                        dut1dn_wm = (tau_w);

                                        if (!infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
                                        if (infRe) dut1dn_wm = dut1dn_wm/(les_eps+nu_t);

					dut1dn = dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;


					Comput_JacobTensor_i(i1, j1, k1, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

					Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);


                                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

					g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
					g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
					g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;
			
					r11 = dudc * csi0 + dude * eta0 + dudz * zet0;	//du_dx * J
					r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;	//dv_dx * J
					r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;	//dw_dx * J

					r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
					r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
					r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

					r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
					r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
					r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;


                                        if (infRe) nu=0.0;
					visc1_wm[k1][j1][i1].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * csi0 + r21 * csi1 + r31 * csi2) * ajc * (nu_t+nu);
					visc1_wm[k1][j1][i1].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * csi0 + r22 * csi1 + r32 * csi2) * ajc * (nu_t+nu);
					visc1_wm[k1][j1][i1].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * csi0 + r23 * csi1 + r33 * csi2) * ajc * (nu_t+nu);
		
				}
				}
				

				i1=i, k1=k;
				j1=j;
				int js=j-1;
				int je=j+1;

        			// printf("Here TEST wallmodel JJJ i=%d j=%d, k=%d\n", i1, j1, k1);
//                                for (k1=k-1;k1<k+1;k1++)
                                for (j1=js;j1<je;j1++) {
//				for (j1=j-1;j1<j+1;j1++){
				if ( (nvert[k1][j1][i1]<0.1 && nvert[k1][j1+1][i1]>0.1) || (nvert[k1][j1][i1]>0.1 && nvert[k1][j1+1][i1]<0.1)) {
					ajc = jaj[k1][j1][i1];
					csi0 = jcsi[k1][j1][i1].x, csi1 = jcsi[k1][j1][i1].y, csi2 = jcsi[k1][j1][i1].z;
					eta0 = jeta[k1][j1][i1].x, eta1 = jeta[k1][j1][i1].y, eta2 = jeta[k1][j1][i1].z;
					zet0 = jzet[k1][j1][i1].x, zet1 = jzet[k1][j1][i1].y, zet2 = jzet[k1][j1][i1].z;


					Compute1_du_j (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                                        Comput_du_wmlocal(nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                                        if ( j1==0 || nvert[k1][j1][i1]>0.1 ) {
						//lnu_t[k1][j1+1][i1] = nu_t_b;
						nu_t = lnu_t[k1][j1+1][i1];
					}
                                        else if( j1==my-2 || nvert[k1][j1+1][i1]>0.1 ) {
						//lnu_t[k1][j1][i1] = nu_t_b;
						nu_t = lnu_t[k1][j1][i1];
					}
                                        else nu_t = 0.5 * (lnu_t[k1][j1][i1] + lnu_t[k1][j1+1][i1]);



                                        dut1dn_wm = (tau_w);

                                        if (!infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
                                        if (infRe) dut1dn_wm = dut1dn_wm/(les_eps+nu_t);

                                        dut1dn = dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;

                                        Comput_JacobTensor_j(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

                                        Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);

//                                        Computewm_du_j (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz, dudc_wm, dvdc_wm, dwdc_wm, dude_wm,dvde_wm, dwde_wm, dudz_wm, dvdz_wm, dwdz_wm);

					dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
					dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
					dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

					g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
					g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
					g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;
		
					r11 = dudc * csi0 + dude * eta0 + dudz * zet0;
					r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;
					r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;

					r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
					r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
					r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;
	
					r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
					r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
					r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;

                                        if (infRe) nu=0.0;
					visc2_wm[k1][j1][i1].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * eta0 + r21 * eta1 + r31 * eta2) * ajc * (nu_t+nu);
					visc2_wm[k1][j1][i1].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * eta0 + r22 * eta1 + r32 * eta2) * ajc * (nu_t+nu);
					visc2_wm[k1][j1][i1].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * eta0 + r23 * eta1 + r33 * eta2) * ajc * (nu_t+nu);
				}
				}

				i1=i, j1=j;
				k1=k;
				int ks=k-1;
				int ke=k+1;
				
        			//printf("Here TEST wallmodel KKK i=%d j=%d, k=%d\n", i1, j1, k1);
                                for (k1=ks;k1<ke;k1++ ) {
//                                for (j1=j-1;j1<j+1;j1++)
//				for (i1=i-1;i1<i+1;i1++){
				if ( (nvert[k1][j1][i1]<0.1 && nvert[k1+1][j1][i1]>0.1) || (nvert[k1][j1][i1]>0.1 && nvert[k1+1][j1][i1]<0.1)) {
					ajc = kaj[k1][j1][i1];
					csi0 = kcsi[k1][j1][i1].x, csi1 = kcsi[k1][j1][i1].y, csi2 = kcsi[k1][j1][i1].z;
					eta0 = keta[k1][j1][i1].x, eta1 = keta[k1][j1][i1].y, eta2 = keta[k1][j1][i1].z;
					zet0 = kzet[k1][j1][i1].x, zet1 = kzet[k1][j1][i1].y, zet2 = kzet[k1][j1][i1].z;


					Compute1_du_k (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                                        Comput_du_wmlocal(nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                                        if ( k1==0 || nvert[k1][j1][i1]>0.1 ) {
						//lnu_t[k1+1][j1][i1] = nu_t_b;
						nu_t = lnu_t[k1+1][j1][i1];
					}
                                        else if( k1==mz-2 || nvert[k1+1][j1][i1]>0.1 ) 	{
						//lnu_t[k1][j1][i1] = nu_t_b;
						nu_t = lnu_t[k1][j1][i1];
					}
                                        else nu_t = 0.5 * (lnu_t[k1][j1][i1] + lnu_t[k1+1][j1][i1]);


                                        dut1dn_wm = (tau_w);


                                        if (!infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
					if (infRe) dut1dn_wm = dut1dn_wm/(les_eps+nu_t);
                                        dut1dn = dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;

//                                        dut1dn = (u_in[jj_sb]-u_in[jj_sb-1])/(z_in[jj_sb]-z_in[jj_sb-1]);
//                                        dut1dn = sign * fabs(dut1dn);

//                                        dut1dn = dut1dn/nu;

                                        Comput_JacobTensor_k(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

                                        Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);

                                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

					g11 = csi0 * zet0 + csi1 * zet1 + csi2 * zet2;
					g21 = eta0 * zet0 + eta1 * zet1 + eta2 * zet2;
					g31 = zet0 * zet0 + zet1 * zet1 + zet2 * zet2;

					r11 = dudc * csi0 + dude * eta0 + dudz * zet0;
					r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;
					r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;

					r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
					r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
					r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

					r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
					r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
					r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;

                                        if (infRe) nu=0.0;
			
					visc3_wm[k1][j1][i1].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * zet0 + r21 * zet1 + r31 * zet2) * ajc * (nu+nu_t);
					visc3_wm[k1][j1][i1].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * zet0 + r22 * zet1 + r32 * zet2) * ajc * (nu+nu_t);
					visc3_wm[k1][j1][i1].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * zet0 + r23 * zet1 + r33 * zet2) * ajc * (nu+nu_t);
				}
				}

			}
		}
	}













	
/*	
        // PetscPrintf(PETSC_COMM_WORLD, "Here TEST 1 \n");
	if (immersed && IB_wm) {
		IBMListNode *current;
		double ni_b[3], nj_b[3], nk_b[3];	
		
		for(ibi=0; ibi<NumberOfBodies; ibi++) {
			current = user->ibmlist[ibi].head;
			
        		// PetscPrintf(PETSC_COMM_WORLD, "Here TEST ibi=%d \n", ibi);
			while (current) {
				int i,j,k;
				
				const double ren = user->ren;
				
				IBMInfo *ibminfo = &current->ibm_intp;
				current = current->next;

                                int ni = ibminfo->cell;
                                int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
                                int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
                                int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
                                i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;

                                sb = ibminfo->d_s; sc = sb + ibminfo->d_i;
                                double sk1  = ibminfo->cr1, sk2 = ibminfo->cr2, sk3 = ibminfo->cr3;
                                double cs1 = ibminfo->cs1, cs2 = ibminfo->cs2, cs3 = ibminfo->cs3;
				double nx = ibm[ibi].nf_x[ni], ny = ibm[ibi].nf_y[ni], nz = ibm[ibi].nf_z[ni];
							
				if (ni>=0) {
					Ua.x = ibm[ibi].u[ibm[ibi].nv1[ni]].x * cs1 + ibm[ibi].u[ibm[ibi].nv2[ni]].x * cs2 + ibm[ibi].u[ibm[ibi].nv3[ni]].x * cs3;
					Ua.y = ibm[ibi].u[ibm[ibi].nv1[ni]].y * cs1 + ibm[ibi].u[ibm[ibi].nv2[ni]].y * cs2 + ibm[ibi].u[ibm[ibi].nv3[ni]].y * cs3;
					Ua.z = ibm[ibi].u[ibm[ibi].nv1[ni]].z * cs1 + ibm[ibi].u[ibm[ibi].nv2[ni]].z * cs2 + ibm[ibi].u[ibm[ibi].nv3[ni]].z * cs3;
				}
				else {
					Ua.x = Ua.y = Ua.z = 0;
				}
				
				Uc.x = (ucat[kp1][jp1][ip1].x * sk1 + ucat[kp2][jp2][ip2].x * sk2 + ucat[kp3][jp3][ip3].x * sk3);
				Uc.y = (ucat[kp1][jp1][ip1].y * sk1 + ucat[kp2][jp2][ip2].y * sk2 + ucat[kp3][jp3][ip3].y * sk3);
				Uc.z = (ucat[kp1][jp1][ip1].z * sk1 + ucat[kp2][jp2][ip2].z * sk2 + ucat[kp3][jp3][ip3].z * sk3);

				int i1=ip1, j1=jp1, k1=kp1;

	                        double ajc = aj[k1][j1][i1];
        	                double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                	        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        	double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

        			// PetscPrintf(PETSC_COMM_WORLD, "Here TEST dp ibi=%d \n", ibi);

                        	double dpdc, dpde, dpdz;
				double dp_dx1, dp_dy1, dp_dz1;

	                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

        	                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx1, &dp_dy1, &dp_dz1);

                                i1=ip2; j1=jp2; k1=kp2;

                                ajc = aj[k1][j1][i1];
                                csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                                eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                                zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                                double dp_dx2, dp_dy2, dp_dz2;

                                Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx2, &dp_dy2, &dp_dz2);

                                i1=ip3; j1=jp3; k1=kp3;

                                ajc = aj[k1][j1][i1];
                                csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                                eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                                zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                                double dp_dx3, dp_dy3, dp_dz3;

                                Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                                Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx3, &dp_dy3, &dp_dz3);

                                double dp_dx = (dp_dx1*sk1 + dp_dx2*sk2 + dp_dx3*sk3 );
                                double dp_dy = (dp_dy1*sk1 + dp_dy2*sk2 + dp_dy3*sk3 );
                                double dp_dz = (dp_dz1*sk1 + dp_dz2*sk2 + dp_dz3*sk3 );

				(nu_t_b) = lnu_t[k][j][i];
				nu_t_c = (lnu_t[kp1][jp1][ip1]*sk1 + lnu_t[kp2][jp2][ip2]*sk2 + lnu_t[kp3][jp3][ip3]*sk3 );

				Ub.x = ucat[k][j][i].x;
				Ub.y = ucat[k][j][i].y;
				Ub.z = ucat[k][j][i].z;


        			double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
			        double un = u_c * nx + v_c * ny + w_c * nz;
			        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
			        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

			        double t1x = ut / (ut_mag_c+1.e-11), t1y = vt / (ut_mag_c+1.e-11), t1z = wt / (ut_mag_c+1.e-11);

			        double t2x = ny*t1z - nz*t1y, t2y = nz*t1x - nx*t1z, t2z = nx*t1y - ny*t1x;

				i1 = ip1, j1 = jp1, k1 = kp1; 				

                                Compute_du_center (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                                double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                                Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                                double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                                Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

				double dundn_1 = dundn, dut1dn_1 = dut1dn;


                                i1 = ip2, j1 = jp2, k1 = kp2;

                                Compute_du_center (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                                Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                                Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                                double dundn_2 = dundn, dut1dn_2 = dut1dn;


                                i1 = ip3, j1 = jp3, k1 = kp3;

                                Compute_du_center (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                                Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                                Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                                double dundn_3 = dundn, dut1dn_3 = dut1dn;

                                double dundn_out = (dundn_1*sk1 + dundn_2*sk2 + dundn_3*sk3 );
                                double dut1dn_out = (dut1dn_1*sk1 + dut1dn_2*sk2 + dut1dn_3*sk3 );
				
				bctype = IB_wm; 

				num_innergrid = (int)(log(sc*(dhratio_wm-1)/dh1_wm+1.0)/log(dhratio_wm))+1;
				
	
				double *u_in, *z_in, *nut_les, *nut_rans;

				u_in= (double *) malloc(num_innergrid*sizeof(double));
				nut_les= (double *) malloc(num_innergrid*sizeof(double));
				z_in= (double *) malloc(num_innergrid*sizeof(double));
				nut_rans= (double *) malloc(num_innergrid*sizeof(double));

				innergrid( z_in, sc );
				double nu = 1./user->ren;
		
				if(levelset) {
					double rho_c = (rho[kp1][jp1][ip1] * sk1 + rho[kp2][jp2][ip2] * sk2 + rho[kp3][jp3][ip3] * sk3);
					double mu_c = (mu[kp1][jp1][ip1] * sk1 + mu[kp2][jp2][ip2] * sk2 + mu[kp3][jp3][ip3] * sk3);
					nu = mu_c/rho_c;
						//nu = mu_water/rho_water;
				}
	
	
        			// PetscPrintf(PETSC_COMM_WORLD, "Here TEST wallmodel ibi=%d \n", ibi);

				wallmodel_2( dp_dx, dp_dy, dp_dz, dp_dx, dp_dy, dp_dz, nu, &nu_t_b, &nu_t_c, sb, sc, &Ub, Uc, Ua, bctype, roughness_size, nx, ny, nz, &t1x, &t1y, &t1z, &t2x, &t2y, &t2z, u_in, z_in, &ustar[k][j][i], nut_les, nut_rans);

				lnu_t[k][j][i]=nu_t_b;
				//lnu_t[kp1][jp1][ip1]=nu_t_c;
				//lnu_t[kp2][jp2][ip2]=nu_t_c;
				//lnu_t[kp3][jp3][ip3]=nu_t_c;

				tau[k][j][i]=(nu+nut_rans[1])*(u_in[1]-u_in[0])/(z_in[1]-z_in[0]);

                                ucat[k][j][i].x = Ub.x;
                                ucat[k][j][i].y = Ub.y;
                                ucat[k][j][i].z = Ub.z;

				int jj, jj_sb;

                                double sb_1 = 0.5*(sc+sb);

                                for (jj=1;jj<num_innergrid;jj++) {
                                        if (sb_1>= z_in[jj-1] && sb_1 < z_in[jj]) jj_sb = jj;

                                }

				j1=j; k1=k;		
				i1=i;
				int is=i-1;
				int ie=i+1;

        			//printf("Here TEST wallmodel III i=%d j=%d, k=%d\n", i1, j1, k1);
//				for (k1=k-1;k1<k+1;k1++)
//				for (j1=j-1;j1<j+1;j1++)
				for (i1=is;i1<ie;i1++){
				if ( (nvert[k1][j1][i1]<0.1 && nvert[k1][j1][i1+1]>0.1) || (nvert[k1][j1][i1]>0.1 && nvert[k1][j1][i1+1]<0.1)) {
					PetscReal ajc = iaj[k1][j1][i1];
					csi0 = icsi[k1][j1][i1].x, csi1 = icsi[k1][j1][i1].y, csi2 = icsi[k1][j1][i1].z;
					eta0 = ieta[k1][j1][i1].x, eta1 = ieta[k1][j1][i1].y, eta2 = ieta[k1][j1][i1].z;
					zet0 = izet[k1][j1][i1].x, zet1 = izet[k1][j1][i1].y, zet2 = izet[k1][j1][i1].z;

//                        		area = sqrt( icsi[k1][j1][i1].x*icsi[k1][j1][i1].x + icsi[k1][j1][i1].y*icsi[k1][j1][i1].y + icsi[k1][j1][i1].z*icsi[k1][j1][i1].z );
                                        double nx_1=nx, ny_1=ny, nz_1=nz;
                                        double t1x_1=t1x, t1y_1=t1y, t1z_1=t1z;
                                        double t2x_1=t2x, t2y_1=t2y, t2z_1=t2z;


					Compute1_du_i (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                			double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
			                Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

					double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;		
					Comput_du_wmlocal(nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2); 

                                        if ( i1==0 || nvert[k1][j1][i1]>0.1 ) {
						//lnu_t[k1][j1][i1+1] = nu_t_b;
						nu_t = lnu_t[k1][j1][i1+1];
					}
                                        else if( i1==mx-2 || nvert[k1][j1][i1+1]>0.1 ) {
						//lnu_t[k1][j1][i1] = nu_t_b;
						nu_t = lnu_t[k1][j1][i1];

					}
                                        else nu_t = 0.5 * (lnu_t[k1][j1][i1] + lnu_t[k1][j1][i1+1]);

					


                			double centx = 0.125 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x + coor[k-1][j  ][i  ].x + coor[k-1][j-1][i  ].x + coor[k  ][j  ][i-1].x + coor[k  ][j-1][i-1].x + coor[k-1][j  ][i-1].x + coor[k-1][j-1][i-1].x);
                			double centy = 0.125 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y + coor[k-1][j  ][i  ].y + coor[k-1][j-1][i  ].y + coor[k  ][j  ][i-1].y + coor[k  ][j-1][i-1].y + coor[k-1][j  ][i-1].y + coor[k-1][j-1][i-1].y);
                			double centz = 0.125 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z + coor[k-1][j  ][i  ].z + coor[k-1][j-1][i  ].z + coor[k  ][j  ][i-1].z + coor[k  ][j-1][i-1].z + coor[k-1][j  ][i-1].z + coor[k-1][j-1][i-1].z);


//					double centx = (coor[k  ][j  ][i].x + coor[k-1][j  ][i].x + coor[k  ][j-1][i].x + coor[k-1][j-1][i].x) * 0.25;
//				        double centy = (coor[k  ][j  ][i].y + coor[k-1][j  ][i].y + coor[k  ][j-1][i].y + coor[k-1][j-1][i].y) * 0.25;
//				        double centz = (coor[k  ][j  ][i].z + coor[k-1][j  ][i].z + coor[k  ][j-1][i].z + coor[k-1][j-1][i].z) * 0.25;

				        double centx_1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x + coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
				        double centy_1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y + coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
				        double centz_1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z + coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
					
					double sb_1 = sb + (centx_1-centx)*nx + (centy_1-centy)*ny + (centz_1-centz)*nz;

                                        if (sb_1 < 0.0) {
                                                PetscPrintf(PETSC_COMM_WORLD,"less than 0: sb, dist %le %le \n", sb, sb_1-sb);
                                                sb_1 = 0.0;
                                        }


                                        if (sb_1 > sc) {
                                                PetscPrintf(PETSC_COMM_WORLD,"larger than sc: sb, dist %le %le \n", sb, sb_1-sb);
                                                sb_1 = sc;
                                        }


//					sb_1 = 0.5*(sc+sb);

					for (jj=1;jj<num_innergrid;jj++) {
						if (sb_1>= z_in[jj-1] && sb_1 < z_in[jj]) jj_sb = jj; 
					}

					jj_sb = 1;
					double ratio_wm = 1.0; //fabs(nx*nx_1 + ny*ny_1 + nz*nz_1);
//					nu_t = 0.0;
//					nu_t = nut_in[jj_sb];
//                                        double dut1dn_wm = (nu+nut_in[jj_sb])*(u_in[jj_sb]-u_in[jj_sb-1])/(z_in[jj_sb]-z_in[jj_sb-1]);
                                        double dut1dn_wm;

                                        dut1dn_wm = nu*(u_in[1]-u_in[0])/(z_in[1]-z_in[0]);

                                        if (!infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
                                        if (infRe) dut1dn_wm = dut1dn_wm/(les_eps+nu_t);

					dut1dn = ratio_wm*dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;

//                                        dut1dn = (u_in[jj_sb]-u_in[jj_sb-1])/(z_in[jj_sb]-z_in[jj_sb-1]);

					double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
					Comput_JacobTensor_i(i1, j1, k1, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

					Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);


                                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

					g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
					g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
					g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;
			
					r11 = dudc * csi0 + dude * eta0 + dudz * zet0;	//du_dx * J
					r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;	//dv_dx * J
					r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;	//dw_dx * J

					r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
					r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
					r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

					r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
					r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
					r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;


                                        if (infRe) nu=0.0;
					visc1_wm[k1][j1][i1].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * csi0 + r21 * csi1 + r31 * csi2) * ajc * (nu_t+nu);
					visc1_wm[k1][j1][i1].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * csi0 + r22 * csi1 + r32 * csi2) * ajc * (nu_t+nu);
					visc1_wm[k1][j1][i1].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * csi0 + r23 * csi1 + r33 * csi2) * ajc * (nu_t+nu);
		
				}
				}
				

				i1=i, k1=k;
				j1=j;
				int js=j-1;
				int je=j+1;

        			// printf("Here TEST wallmodel JJJ i=%d j=%d, k=%d\n", i1, j1, k1);
//                                for (k1=k-1;k1<k+1;k1++)
                                for (j1=js;j1<je;j1++) {
//				for (j1=j-1;j1<j+1;j1++){
				if ( (nvert[k1][j1][i1]<0.1 && nvert[k1][j1+1][i1]>0.1) || (nvert[k1][j1][i1]>0.1 && nvert[k1][j1+1][i1]<0.1)) {
					PetscReal ajc = jaj[k1][j1][i1];
					csi0 = jcsi[k1][j1][i1].x, csi1 = jcsi[k1][j1][i1].y, csi2 = jcsi[k1][j1][i1].z;
					eta0 = jeta[k1][j1][i1].x, eta1 = jeta[k1][j1][i1].y, eta2 = jeta[k1][j1][i1].z;
					zet0 = jzet[k1][j1][i1].x, zet1 = jzet[k1][j1][i1].y, zet2 = jzet[k1][j1][i1].z;

					double nx_1=nx, ny_1=ny, nz_1=nz;
					double t1x_1=t1x, t1y_1=t1y, t1z_1=t1z;
					double t2x_1=t2x, t2y_1=t2y, t2z_1=t2z;

					Compute1_du_j (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                                        Comput_du_wmlocal(nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                                        if ( j1==0 || nvert[k1][j1][i1]>0.1 ) {
						//lnu_t[k1][j1+1][i1] = nu_t_b;
						nu_t = lnu_t[k1][j1+1][i1];
					}
                                        else if( j1==my-2 || nvert[k1][j1+1][i1]>0.1 ) {
						//lnu_t[k1][j1][i1] = nu_t_b;
						nu_t = lnu_t[k1][j1][i1];
					}
                                        else nu_t = 0.5 * (lnu_t[k1][j1][i1] + lnu_t[k1][j1+1][i1]);

                			double centx = 0.125 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x + coor[k-1][j  ][i  ].x + coor[k-1][j-1][i  ].x + coor[k  ][j  ][i-1].x + coor[k  ][j-1][i-1].x + coor[k-1][j  ][i-1].x + coor[k-1][j-1][i-1].x);
                			double centy = 0.125 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y + coor[k-1][j  ][i  ].y + coor[k-1][j-1][i  ].y + coor[k  ][j  ][i-1].y + coor[k  ][j-1][i-1].y + coor[k-1][j  ][i-1].y + coor[k-1][j-1][i-1].y);
                			double centz = 0.125 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z + coor[k-1][j  ][i  ].z + coor[k-1][j-1][i  ].z + coor[k  ][j  ][i-1].z + coor[k  ][j-1][i-1].z + coor[k-1][j  ][i-1].z + coor[k-1][j-1][i-1].z);


//                                        double centx = (coor[k  ][j  ][i].x + coor[k-1][j  ][i].x + coor[k  ][j][i-1].x + coor[k-1][j][i-1].x) * 0.25;
//                                        double centy = (coor[k  ][j  ][i].y + coor[k-1][j  ][i].y + coor[k  ][j][i-1].y + coor[k-1][j][i-1].y) * 0.25;
//                                        double centz = (coor[k  ][j  ][i].z + coor[k-1][j  ][i].z + coor[k  ][j][i-1].z + coor[k-1][j][i-1].z) * 0.25;

                                        double centx_1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x + coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
                                        double centy_1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y + coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
                                        double centz_1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z + coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

                                        double sb_1 = sb + (centx_1-centx)*nx + (centy_1-centy)*ny + (centz_1-centz)*nz;

                                        if (sb_1 < 0.0) {
                                                PetscPrintf(PETSC_COMM_WORLD,"less than 0: sb, dist %le %le \n", sb, sb_1-sb);
                                                sb_1 = 0.0;
                                        }


                                        if (sb_1 > sc) {
                                                PetscPrintf(PETSC_COMM_WORLD,"larger than sc: sb, dist %le %le \n", sb, sb_1-sb);
                                                sb_1 = sc;
                                        }



//                                        if (sb_1 < 0.0) sb_1 = 0.0;

//                                        if (sb_1 > sc) sb_1 = sc;

                                        for (jj=1;jj<num_innergrid;jj++) {
                                              if (sb_1>= z_in[jj-1] && sb_1 < z_in[jj]) jj_sb = jj;

                                        }
					
//					double sign = fabs(dut1dn+1.e-9)/(dut1dn+1.e-9);					


                                        jj_sb = 1;
                                        double ratio_wm = 1.0; //fabs(nx*nx_1 + ny*ny_1 + nz*nz_1);
//                                      nu_t = 0.0;
//                                      nu_t = nut_in[jj_sb];
//                                        double dut1dn_wm = (nu+nut_in[jj_sb])*(u_in[jj_sb]-u_in[jj_sb-1])/(z_in[jj_sb]-z_in[jj_sb-1]);
                                        double dut1dn_wm;

                                        dut1dn_wm = nu*(u_in[1]-u_in[0])/(z_in[1]-z_in[0]);

                                        if (!infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
                                        if (infRe) dut1dn_wm = dut1dn_wm/(les_eps+nu_t);

                                        dut1dn = ratio_wm*dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;

                                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                                        Comput_JacobTensor_j(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

                                        Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);

//                                        Computewm_du_j (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz, dudc_wm, dvdc_wm, dwdc_wm, dude_wm,dvde_wm, dwde_wm, dudz_wm, dvdz_wm, dwdz_wm);

					dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
					dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
					dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

					g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
					g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
					g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;
		
					r11 = dudc * csi0 + dude * eta0 + dudz * zet0;
					r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;
					r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;

					r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
					r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
					r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;
	
					r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
					r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
					r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;

                                        if (infRe) nu=0.0;
					visc2_wm[k1][j1][i1].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * eta0 + r21 * eta1 + r31 * eta2) * ajc * (nu_t+nu);
					visc2_wm[k1][j1][i1].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * eta0 + r22 * eta1 + r32 * eta2) * ajc * (nu_t+nu);
					visc2_wm[k1][j1][i1].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * eta0 + r23 * eta1 + r33 * eta2) * ajc * (nu_t+nu);
				}
				}

				i1=i, j1=j;
				k1=k;
				int ks=k-1;
				int ke=k+1;
				
        			//printf("Here TEST wallmodel KKK i=%d j=%d, k=%d\n", i1, j1, k1);
                                for (k1=ks;k1<ke;k1++ ) {
//                                for (j1=j-1;j1<j+1;j1++)
//				for (i1=i-1;i1<i+1;i1++){
				if ( (nvert[k1][j1][i1]<0.1 && nvert[k1+1][j1][i1]>0.1) || (nvert[k1][j1][i1]>0.1 && nvert[k1+1][j1][i1]<0.1)) {
					PetscReal ajc = kaj[k1][j1][i1];
					csi0 = kcsi[k1][j1][i1].x, csi1 = kcsi[k1][j1][i1].y, csi2 = kcsi[k1][j1][i1].z;
					eta0 = keta[k1][j1][i1].x, eta1 = keta[k1][j1][i1].y, eta2 = keta[k1][j1][i1].z;
					zet0 = kzet[k1][j1][i1].x, zet1 = kzet[k1][j1][i1].y, zet2 = kzet[k1][j1][i1].z;


                                        double nx_1=nx, ny_1=ny, nz_1=nz;
                                        double t1x_1=t1x, t1y_1=t1y, t1z_1=t1z;
                                        double t2x_1=t2x, t2y_1=t2y, t2z_1=t2z;

	
					Compute1_du_k (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                                        Comput_du_wmlocal(nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                                        if ( k1==0 || nvert[k1][j1][i1]>0.1 ) {
						//lnu_t[k1+1][j1][i1] = nu_t_b;
						nu_t = lnu_t[k1+1][j1][i1];
					}
                                        else if( k1==mz-2 || nvert[k1+1][j1][i1]>0.1 ) 	{
						//lnu_t[k1][j1][i1] = nu_t_b;
						nu_t = lnu_t[k1][j1][i1];
					}
                                        else nu_t = 0.5 * (lnu_t[k1][j1][i1] + lnu_t[k1+1][j1][i1]);

                			double centx = 0.125 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x + coor[k-1][j  ][i  ].x + coor[k-1][j-1][i  ].x + coor[k  ][j  ][i-1].x + coor[k  ][j-1][i-1].x + coor[k-1][j  ][i-1].x + coor[k-1][j-1][i-1].x);
                			double centy = 0.125 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y + coor[k-1][j  ][i  ].y + coor[k-1][j-1][i  ].y + coor[k  ][j  ][i-1].y + coor[k  ][j-1][i-1].y + coor[k-1][j  ][i-1].y + coor[k-1][j-1][i-1].y);
                			double centz = 0.125 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z + coor[k-1][j  ][i  ].z + coor[k-1][j-1][i  ].z + coor[k  ][j  ][i-1].z + coor[k  ][j-1][i-1].z + coor[k-1][j  ][i-1].z + coor[k-1][j-1][i-1].z);


//                                        double centx = (coor[k  ][j  ][i].x + coor[k][j-1][i].x + coor[k  ][j][i-1].x + coor[k][j-1][i-1].x) * 0.25;
//                                       double centy = (coor[k  ][j  ][i].y + coor[k][j-1][i].y + coor[k  ][j][i-1].y + coor[k][j-1][i-1].y) * 0.25;
//                                        double centz = (coor[k  ][j  ][i].z + coor[k][j-1][i].z + coor[k  ][j][i-1].z + coor[k][j-1][i-1].z) * 0.25;

                                        double centx_1 = (coor[k1  ][j1  ][i1].x + coor[k1][j1-1][i1].x + coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
                                        double centy_1 = (coor[k1  ][j1  ][i1].y + coor[k1][j1-1][i1].y + coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
                                        double centz_1 = (coor[k1  ][j1  ][i1].z + coor[k1][j1-1][i1].z + coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

                                        double sb_1 = sb + (centx_1-centx)*nx + (centy_1-centy)*ny + (centz_1-centz)*nz;

                                        if (sb_1 < 0.0) {
                                                PetscPrintf(PETSC_COMM_WORLD,"less than 0: sb, dist %le %le \n", sb, sb_1-sb);
                                                sb_1 = 0.0;
                                        }


                                        if (sb_1 > sc) {
                                                PetscPrintf(PETSC_COMM_WORLD, "larger than sc: sb, dist %le %le \n", sb, sb_1-sb);
                                                sb_1 = sc;
                                        }


//                                        if (sb_1 < 0.0) sb_1 = 0.0;

 //                                      if (sb_1 > sc) sb_1 = sc;

                                        for (jj=1;jj<num_innergrid;jj++) {
                                                if (sb_1>= z_in[jj-1] && sb_1 < z_in[jj]) jj_sb = jj;

                                        }


//                                        double sign = fabs(dut1dn+1.e-9)/(dut1dn+1.e-9);

                                        jj_sb = 1;
                                        double ratio_wm = 1.0; //fabs(nx*nx_1 + ny*ny_1 + nz*nz_1);
//                                      nu_t = 0.0;
//                                      nu_t = nut_in[jj_sb];
//                                        double dut1dn_wm = (nu+nut_in[jj_sb])*(u_in[jj_sb]-u_in[jj_sb-1])/(z_in[jj_sb]-z_in[jj_sb-1]);

                                        double dut1dn_wm;

                                        dut1dn_wm = nu*(u_in[1]-u_in[0])/(z_in[1]-z_in[0]);


                                        if (!infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
					if (infRe) dut1dn_wm = dut1dn_wm/(les_eps+nu_t);
                                        dut1dn = ratio_wm*dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;

//                                        dut1dn = (u_in[jj_sb]-u_in[jj_sb-1])/(z_in[jj_sb]-z_in[jj_sb-1]);
//                                        dut1dn = sign * fabs(dut1dn);

//                                        dut1dn = dut1dn/nu;

                                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                                        Comput_JacobTensor_k(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

                                        Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx_1, ny_1, nz_1, t1x_1, t1y_1, t1z_1, t2x_1, t2y_1, t2z_1, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);

                                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

					g11 = csi0 * zet0 + csi1 * zet1 + csi2 * zet2;
					g21 = eta0 * zet0 + eta1 * zet1 + eta2 * zet2;
					g31 = zet0 * zet0 + zet1 * zet1 + zet2 * zet2;

					r11 = dudc * csi0 + dude * eta0 + dudz * zet0;
					r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;
					r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;

					r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
					r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
					r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

					r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
					r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
					r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;

                                        if (infRe) nu=0.0;
			
					visc3_wm[k1][j1][i1].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * zet0 + r21 * zet1 + r31 * zet2) * ajc * (nu+nu_t);
					visc3_wm[k1][j1][i1].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * zet0 + r22 * zet1 + r32 * zet2) * ajc * (nu+nu_t);
					visc3_wm[k1][j1][i1].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * zet0 + r23 * zet1 + r33 * zet2) * ajc * (nu+nu_t);
				}
				}

				free(u_in);
				free(z_in);	
				free(nut_les);
				free(nut_rans);
			}
		}

	}

*/
        double ni[3], nj[3], nk[3];
	
	// i direction
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {

		if(i==mx-1 || j==my-1 || k==mz-1) continue;
		if(j==0 || k==0) continue;

                if ( imin_wm != 0 && i == 0 ) {
	                area = sqrt( csi[k][j][i+1].x*csi[k][j][i+1].x + csi[k][j][i+1].y*csi[k][j][i+1].y + csi[k][j][i+1].z*csi[k][j][i+1].z );

			st = sqrt(area);

                        Ua.x = Ua.y = Ua.z = 0;

                        sc = 2* 0.5/aj[k][j][i+1]/area + 0.5/aj[k][j][i+2]/area;
                        Uc = ucat[k][j][i+2];

			/*
                        sc = 1.0/aj[k][j][i+1]/area + 1.0/aj[k][j][i+2]/area + 0.5/aj[k][j][i+3]/area;
                        Uc = ucat[k][j][i+3];
			*/

	                sb = 0.5/aj[k][j][i+1]/area;
                        Ub = ucat[k][j][i+1];

		       	bctype = imin_wm; //user->bctype[0];
			Calculate_normal(csi[k][j][i+1], eta[k][j][i+1], zet[k][j][i+1], ni, nj, nk);
			nx =  ni[0], ny =  ni[1], nz =  ni[2];

			nu_t_c = lnu_t[k][j][i+2];
			nu_t_b = lnu_t[k][j][i+1];

                        int i1 = i+2, j1 = j, k1 = k;
                        double ajc = aj[k1][j1][i1];
                        double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

			double dpdc, dpde, dpdz, dp_dx, dp_dy, dp_dz;

			Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

			Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);


        		double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
		        double un = u_c * nx + v_c * ny + w_c * nz;
		        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
		        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

			double t1x = ut / (ut_mag_c+1.e-11), t1y = vt / (ut_mag_c+1.e-11), t1z = wt / (ut_mag_c+1.e-11);

			double t2x = ny*t1z - nz*t1y, t2y = nz*t1x - nx*t1z, t2z = nx*t1y - ny*t1x;

			i1 = i+2, j1 = j, k1 = k; 				

                        Compute_du_center (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                        Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

			double dundn_out = dundn, dut1dn_out = dut1dn;

			
			double nu = 1./user->ren, nu_t=0;
		
			if(levelset) {
				nu = mu[k][j][i+2]/rho[k][j][i+2];
			}


			double nut_b, nut_c, tau_w, nut_2sb;

			wallmodel_s(nu,sb,sc,Uc,&Ub,Ua,bctype,roughness_size,nx,ny,nz,&tau_w,&ustar[k][j][i+1],dp_dx,dp_dy,dp_dz,&nut_2sb,nu_t_c);

			//wallmodel_2( dp_dx,dp_dy,dp_dz,nu,&nut_b,&nut_c,sb,sc,&Ub,Uc,Ua,bctype,roughness_size,nx,ny,nz,&tau_w);

			//lnu_t[k][j][i+1]= (nut_b);

			tau[k][j][i+1]=(tau_w);

			ajc = iaj[k][j][i];
			csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
			eta0 = ieta[k][j][i].x, eta1 = ieta[k][j][i].y, eta2 = ieta[k][j][i].z;
			zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;

			Compute1_du_i (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

//                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

//                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                        Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                        if ( i==0 || nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j][i+1];
                        else if( i==mx-2 || nvert[k][j][i+1]>0.1 ) nu_t = lnu_t[k][j][i];
                        else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j][i+1]);
	
                        //nu_t = 0.5 * (lnu_t[k][j][i+1] + lnu_t[k][j][i+2]);
                        dut1dn = tau_w;

                        if (!infRe) dut1dn = dut1dn/(nu /*+nu_t*/);

			if (infRe) dut1dn = dut1dn/(les_eps+nu_t);
//                        dut1dn = dut1dn/nu;

                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                        Comput_JacobTensor_i(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

                        Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);

                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;
		
			g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
			g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
			g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

			r11 = dudc * csi0 + dude * eta0 + dudz * zet0;	//du_dx * J
			r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;	//dv_dx * J
			r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;	//dw_dx * J

			r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
			r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
			r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

			r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
			r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
			r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;
		
		  	
			if (infRe) nu=0.0;
			visc1_wm[k][j][i].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * csi0 + r21 * csi1 + r31 * csi2) * ajc * (nu_t+nu);
			visc1_wm[k][j][i].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * csi0 + r22 * csi1 + r32 * csi2) * ajc * (nu_t+nu);
			visc1_wm[k][j][i].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * csi0 + r23 * csi1 + r33 * csi2) * ajc * (nu_t+nu);

		}

                if ( imax_wm != 0 && i == mx-2 ) {
                        area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );

			st = sqrt(area);
                        Ua.x = Ua.y = Ua.z = 0;
			
                        sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j][i-1]/area;
                        Uc = ucat[k][j][i-1];

			/*
                        sc = 1.0/aj[k][j][i]/area + 1.0/aj[k][j][i-1]/area+ 0.5/aj[k][j][i-2]/area;
                        Uc = ucat[k][j][i-2];
			*/

                        sb = 0.5/aj[k][j][i]/area;
                        Ub = ucat[k][j][i];

                        bctype = imax_wm; //user->bctype[1];
                        Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
                        nx =  -ni[0], ny =  -ni[1], nz =  -ni[2];

                        nu_t_c = lnu_t[k][j][i-1];
                        nu_t_b = lnu_t[k][j][i];

                        int i1 = i-1, j1 = j, k1 = k;
                        double ajc = aj[k1][j1][i1];
                        double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                        double dpdc, dpde, dpdz, dp_dx, dp_dy, dp_dz;

                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                        Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);


                        double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
                        double un = u_c * nx + v_c * ny + w_c * nz;
                        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
                        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

                        double t1x = ut / (ut_mag_c+1.e-11), t1y = vt / (ut_mag_c+1.e-11), t1z = wt / (ut_mag_c+1.e-11);

                        double t2x = ny*t1z - nz*t1y, t2y = nz*t1x - nx*t1z, t2z = nx*t1y - ny*t1x;

                        i1 = i-1, j1 = j, k1 = k;

                        Compute_du_center (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                        Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                        double dundn_out = dundn, dut1dn_out = dut1dn;


			double nu = 1./user->ren, nu_t=0;		

			if(levelset) {
				nu = mu[k][j][i-1]/rho[k][j][i-1];
			}


			double nut_b, nut_c, tau_w, nut_2sb;

			wallmodel_s(nu,sb,sc,Uc,&Ub,Ua,bctype,roughness_size,nx,ny,nz,&tau_w,&ustar[k][j][i],dp_dx,dp_dy,dp_dz,&nut_2sb,nu_t_c);
			//wallmodel_2( dp_dx,dp_dy,dp_dz,nu,&nut_b,&nut_c,sb,sc,&Ub,Uc,Ua,bctype,roughness_size,nx,ny,nz,&tau_w);

			//lnu_t[k][j][i]= (nut_b);

			tau[k][j][i]=(tau_w);


			ajc = iaj[k][j][i];
			csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
			eta0 = ieta[k][j][i].x, eta1 = ieta[k][j][i].y, eta2 = ieta[k][j][i].z;
			zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;

			Compute1_du_i (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

//                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );


//                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                        Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                        if ( i==0 || nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j][i+1];
                        else if( i==mx-2 || nvert[k][j][i+1]>0.1 ) nu_t = lnu_t[k][j][i];
                        else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j][i+1]);

                        //nu_t = 0.5 * (lnu_t[k][j][i-1] + lnu_t[k][j][i]);
                        dut1dn = tau_w;

			if (!infRe) dut1dn = dut1dn/(nu /*+nu_t*/);

			if (infRe) dut1dn = dut1dn/(les_eps+nu_t);
//                        dut1dn = dut1dn/nu;

                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                        Comput_JacobTensor_i(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

                        Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);

                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;
		
			g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
			g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
			g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

			r11 = dudc * csi0 + dude * eta0 + dudz * zet0;	//du_dx * J
			r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;	//dv_dx * J
			r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;	//dw_dx * J

			r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
			r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
			r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

			r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
			r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
			r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;
		

			if (infRe) nu=0.0;

			visc1_wm[k][j][i].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * csi0 + r21 * csi1 + r31 * csi2) * ajc * (nu_t+nu);
			visc1_wm[k][j][i].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * csi0 + r22 * csi1 + r32 * csi2) * ajc * (nu_t+nu);
			visc1_wm[k][j][i].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * csi0 + r23 * csi1 + r33 * csi2) * ajc * (nu_t+nu);


		}
	
		
	}
  
	// j direction
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		if(i==mx-1 || j==my-1 || k==mz-1) continue;
//		if(i==mx-2 || i==2) continue;
		if(i==0 || k==0) continue;

                if ( jmin_wm != 0 && j == 0 ) {
                        area = sqrt( eta[k][j+1][i].x*eta[k][j+1][i].x + eta[k][j+1][i].y*eta[k][j+1][i].y + eta[k][j+1][i].z*eta[k][j+1][i].z );

			st = sqrt(area);
                        Ua.x = Ua.y = Ua.z = 0;

                        if (user->bctype[2]==1001 && MoveFrame) {
                                double k_iww=2.0*M_PI/lamda_iww;
                                Ua.x=0.0;
                                Ua.y=C_iww*a_iww*k_iww*sin(k_iww*coor[k][j][i].z);
                                Ua.z=C_iww*(-1.0+a_iww*k_iww*cos(k_iww*coor[k][j][i].z));
                        }

                        sc = 2* 0.5/aj[k][j+1][i]/area + 0.5/aj[k][j+2][i]/area;
                        Uc = ucat[k][j+2][i];

			/*			
                        sc = 1.0/aj[k][j+1][i]/area + 1.0/aj[k][j+2][i]/area + 0.5/aj[k][j+3][i]/area;
                        Uc = ucat[k][j+3][i];
			*/

//			if (!rank) printf("k=%d j=%d i=%d Uc=%le \n", k,j,i, ucat[k][j+2][i].z);

                        sb = 0.5/aj[k][j+1][i]/area;
                        Ub = ucat[k][j+1][i];

                        bctype = jmin_wm; //user->bctype[2];
                        Calculate_normal(csi[k][j+1][i], eta[k][j+1][i], zet[k][j+1][i], ni, nj, nk);
                        nx =  nj[0], ny =  nj[1], nz =  nj[2];

                        nu_t_c = lnu_t[k][j+2][i];
                        nu_t_b = lnu_t[k][j+1][i];


                        int i1 = i, j1 = j+1, k1 = k;
                        double ajc = aj[k1][j1][i1];
                        double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                        double dpdc, dpde, dpdz, dp_dx=0.0, dp_dy=0.0, dp_dz=0.0;

                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                        Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);
			double dpdx_b=dp_dx, dpdy_b=dp_dy, dpdz_b=dp_dz;

                        i1 = i; j1 = j+2; k1 = k;
                        ajc = aj[k1][j1][i1];
                        csi0 = csi[k1][j1][i1].x; csi1 = csi[k1][j1][i1].y; csi2 = csi[k1][j1][i1].z;
                        eta0 = eta[k1][j1][i1].x; eta1 = eta[k1][j1][i1].y; eta2 = eta[k1][j1][i1].z;
                        zet0 = zet[k1][j1][i1].x; zet1 = zet[k1][j1][i1].y; zet2 = zet[k1][j1][i1].z;

                        dpdc=0.0; dpde=0.0; dpdz=0.0; dp_dx=0.0; dp_dy=0.0; dp_dz=0.0;

                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                        Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);
			double dpdx_c=dp_dx, dpdy_c=dp_dy, dpdz_c=dp_dz;

                        double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
                        double un = u_c * nx + v_c * ny + w_c * nz;
                        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
                        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

                        double t1x = ut / (ut_mag_c+1.e-11), t1y = vt / (ut_mag_c+1.e-11), t1z = wt / (ut_mag_c+1.e-11);

                        double t2x = ny*t1z - nz*t1y, t2y = nz*t1x - nx*t1z, t2z = nx*t1y - ny*t1x;

                        i1 = i, j1 = j+2, k1 = k;

                        Compute_du_center (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                        Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                        double dundn_out = dundn, dut1dn_out = dut1dn;

			
        	        double nu = 1./user->ren, nu_t = 0;
				
			if(levelset) {
				nu = mu[k][j+2][i]/rho[k][j+2][i];
			}


			double nut_b, nut_c, tau_w, nut_2sb;

			wallmodel_s(nu,sb,sc,Uc,&Ub,Ua,bctype,roughness_size,nx,ny,nz,&tau_w,&ustar[k][j+1][i],dp_dx,dp_dy,dp_dz,&nut_2sb,nu_t_c);
			//wallmodel_2( dp_dx,dp_dy,dp_dz,nu,&nut_b,&nut_c,sb,sc,&Ub,Uc,Ua,bctype,roughness_size,nx,ny,nz,&tau_w);

			//lnu_t[k][j+1][i]= (nut_b);

			tau[k][j+1][i]=(tau_w);

                	ajc = jaj[k][j][i];
	                csi0 = jcsi[k][j][i].x, csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
        	        eta0 = jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
                	zet0 = jzet[k][j][i].x, zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;

	                Compute1_du_j (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

//                        printf("dudc, dvdc, dwdc %le %le %le \n", dudc, dvdc, dwdc);
//                        printf("dude, dvde, dwde %le %le %le \n", dude, dvde, dwde);
//                        printf("dudz, dvdz, dwdz %le %le %le \n", dudz, dvdz, dwdz);
		

//                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

//                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                        Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                        if ( j==0 || nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j+1][i];
                        else if( j==my-2 || nvert[k][j+1][i]>0.1 ) nu_t = lnu_t[k][j][i];
                        else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j+1][i]);

                        //nu_t = 0.5 * (lnu_t[k][j+1][i] + lnu_t[k][j+2][i]);
                        dut1dn = tau_w; 

//                        printf("dut1dn before %le \n", dut1dn);
//                        dut1dn = nu*(u_in[1]-u_in[0])/(z_in[1]-z_in[0]);
			if (!infRe) dut1dn = dut1dn/(nu /*+nu_t*/);

			if (infRe) dut1dn = dut1dn/(les_eps+nu_t);
//                        dut1dn = dut1dn/nu;
//                        printf("dut1dn after %le \n", dut1dn);
                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                        Comput_JacobTensor_j(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

                        Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);



//                        printf("dudc_wm, dvdc_wm, dwdc_wm %le %le %le \n", dudc_wm, dvdc_wm, dwdc_wm);
//                        printf("dude_wm, dvde_wm, dwde_wm %le %le %le \n", dude_wm, dvde_wm, dwde_wm);
//                        printf("dudz_wm, dvdz_wm, dwdz_wm %le %le %le \n", dudz_wm, dvdz_wm, dwdz_wm);
                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

	
        	        g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
                	g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
	                g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;

        	        r11 = dudc * csi0 + dude * eta0 + dudz * zet0;
                	r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;
	                r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;

        	        r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
                	r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
	                r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

        	        r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
                	r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
	                r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;

			if (infRe) nu = 0.0;
                        visc2_wm[k][j][i].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * eta0 + r21 * eta1 + r31 * eta2) * ajc * (nu_t+nu);
                        visc2_wm[k][j][i].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * eta0 + r22 * eta1 + r32 * eta2) * ajc * (nu_t+nu);
                        visc2_wm[k][j][i].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * eta0 + r23 * eta1 + r33 * eta2) * ajc * (nu_t+nu);

	
                }


                if ( jmax_wm != 0 && j == my-2 ) {

                        area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );


			st = sqrt(area);
                        Ua.x = Ua.y = Ua.z = 0;

                        sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j-1][i]/area;
                        Uc = ucat[k][j-1][i];
		
			/*
                        sc = 1.0/aj[k][j][i]/area + 1.0/aj[k][j-1][i]/area + 0.5/aj[k][j-2][i]/area;
                        Uc = ucat[k][j-2][i];
			*/

                        sb = 0.5/aj[k][j][i]/area;
                        Ub = ucat[k][j][i];

                        bctype = jmax_wm; //user->bctype[3];
                        Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
                        nx =  -nj[0], ny =  -nj[1], nz =  -nj[2];

                        nu_t_c = lnu_t[k][j-1][i];
                        nu_t_b = lnu_t[k][j][i];

                        int i1 = i, j1 = j-1, k1 = k;
                        double ajc = aj[k1][j1][i1];
                        double csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        double eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        double zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                        double dpdc, dpde, dpdz, dp_dx=0.0, dp_dy=0.0, dp_dz=0.0;

                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                        Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);

			double dpdx_b=dp_dx, dpdy_b=dp_dy, dpdz_b=dp_dz;


                        i1 = i, j1 = j-2, k1 = k;
                        ajc = aj[k1][j1][i1];
                        csi0 = csi[k1][j1][i1].x, csi1 = csi[k1][j1][i1].y, csi2 = csi[k1][j1][i1].z;
                        eta0 = eta[k1][j1][i1].x, eta1 = eta[k1][j1][i1].y, eta2 = eta[k1][j1][i1].z;
                        zet0 = zet[k1][j1][i1].x, zet1 = zet[k1][j1][i1].y, zet2 = zet[k1][j1][i1].z;

                        Compute_dscalar_center (i1, j1, k1, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );

                        Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);

			double dpdx_c=dp_dx, dpdy_c=dp_dy, dpdz_c=dp_dz;

                        double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
                        double un = u_c * nx + v_c * ny + w_c * nz;
                        double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
                        double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

                        double t1x = ut / (ut_mag_c+1.e-11), t1y = vt / (ut_mag_c+1.e-11), t1z = wt / (ut_mag_c+1.e-11);

                        double t2x = ny*t1z - nz*t1y, t2y = nz*t1x - nx*t1z, t2z = nx*t1y - ny*t1x;

                        i1 = i, j1 = j-1, k1 = k;

                        Compute_du_center (i1, j1, k1, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                        Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

                        double dundn_out = dundn, dut1dn_out = dut1dn;


        	        double nu = 1./user->ren, nu_t = 0;		

			if(levelset) {
				nu = mu[k][j-1][i]/rho[k][j-1][i];
			}
	

			double nut_b, nut_c, tau_w, nut_2sb;

			wallmodel_s(nu,sb,sc,Uc,&Ub,Ua,bctype,roughness_size,nx,ny,nz,&tau_w,&ustar[k][j][i],dp_dx,dp_dy,dp_dz,&nut_2sb,nu_t_c);
			//wallmodel_2( dp_dx,dp_dy,dp_dz,nu,&nut_b,&nut_c,sb,sc,&Ub,Uc,Ua,bctype,roughness_size,nx,ny,nz,&tau_w);

			//lnu_t[k][j][i]= (nut_b);

			tau[k][j][i]=(tau_w);

                	ajc = jaj[k][j][i];
	                csi0 = jcsi[k][j][i].x, csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
        	        eta0= jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
                	zet0 = jzet[k][j][i].x, zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;

	                Compute1_du_j (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

//                        double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                        Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

//                        double dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;
                        Comput_du_wmlocal(nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz, &dut1dn, &dut2dn, &dundn, &dut1dt1, &dut2dt1, &dundt1, &dut1dt2, &dut2dt2, &dundt2);

//                        printf("dut1dn before %le \n", dut1dn);
                        if ( j==0 || nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j+1][i];
                        else if( j==my-2 || nvert[k][j+1][i]>0.1 ) nu_t = lnu_t[k][j][i];
                        else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j+1][i]);
//                        printf("jUb Uc %le %le  %le %le \n", sb, sc, Ub.z, Uc.z);

                        //nu_t = 0.5 * (lnu_t[k][j-1][i] + lnu_t[k][j][i]);
                        dut1dn = tau_w;

//                        dut1dn = nu*(u_in[1]-u_in[0])/(z_in[1]-z_in[0]);
                        if (!infRe) dut1dn = dut1dn/(nu /*+nu_t*/);

			if (infRe) dut1dn = dut1dn/(les_eps+nu_t);
//                        dut1dn = dut1dn/nu;
//
//                        printf("dut1dn after %le \n", dut1dn);

                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                        Comput_JacobTensor_j(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

                        Comput_du_Compgrid(dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z, dut1dn, dut2dn, dundn, dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2, &dudc_wm, &dvdc_wm, &dwdc_wm, &dude_wm, &dvde_wm, &dwde_wm, &dudz_wm, &dvdz_wm, &dwdz_wm);

//                        printf("dudc, dvdc, dwdc %le %le %le \n", dudc, dvdc, dwdc);
//                        printf("dudc_wm, dvdc_wm, dwdc_wm %le %le %le \n", dudc_wm, dvdc_wm, dwdc_wm);
//                        printf("dude, dvde, dwde %le %le %le \n", dude, dvde, dwde);
//                        printf("dude_wm, dvde_wm, dwde_wm %le %le %le \n", dude_wm, dvde_wm, dwde_wm);
//                        printf("dudz, dvdz, dwdz %le %le %le \n", dudz, dvdz, dwdz);
//                        printf("dudz_wm, dvdz_wm, dwdz_wm %le %le %le \n", dudz_wm, dvdz_wm, dwdz_wm);

                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;
	
        	        g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
                	g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
	                g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;

        	        r11 = dudc * csi0 + dude * eta0 + dudz * zet0;
                	r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;
	                r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;

        	        r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
                	r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
	                r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

        	        r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
                	r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
	                r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;

			if (infRe) nu=0.0;
                        visc2_wm[k][j][i].x = (g11 * dudc + g21 * dude + g31 * dudz + r11 * eta0 + r21 * eta1 + r31 * eta2) * ajc * (nu_t+nu);
                        visc2_wm[k][j][i].y = (g11 * dvdc + g21 * dvde + g31 * dvdz + r12 * eta0 + r22 * eta1 + r32 * eta2) * ajc * (nu_t+nu);
                        visc2_wm[k][j][i].z = (g11 * dwdc + g21 * dwde + g31 * dwdz + r13 * eta0 + r23 * eta1 + r33 * eta2) * ajc * (nu_t+nu);


                }

	
	}


        for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++) {

                if (jmin_wm && user->bctype[0]==1 && j==0 && i==1) {
                                Set(&visc2_wm[k][j][i],0);
                }

                if (jmin_wm && user->bctype[1]==1 && j==0 && i==mx-2) {
                                Set(&visc2_wm[k][j][i],0);
                }

                if (jmax_wm && user->bctype[0]==1 && j==my-2 && i==1) {
                                Set(&visc2_wm[k][j][i],0);
                }

                if (jmax_wm && user->bctype[1]==1 && j==my-2 && i==mx-2) {
                                Set(&visc2_wm[k][j][i],0);
                }

		// 
                if (imin_wm && user->bctype[2]==1 && i==0 && j==1) {
                                Set(&visc1_wm[k][j][i],0);
                }

                if (imin_wm && user->bctype[3]==1 && i==0 && j==my-2) {
                                Set(&visc1_wm[k][j][i],0);
                }

                if (imax_wm && user->bctype[2]==1 && i==mx-2 && j==1) {
                                Set(&visc1_wm[k][j][i],0);
                }

                if (imax_wm && user->bctype[3]==1 && i==mx-2 && j==my-2) {
                                Set(&visc1_wm[k][j][i],0);
                }

        }

		
	DMDAVecRestoreArray(fda, user->lVisc1_wm, &visc1_wm);
	DMDAVecRestoreArray(fda, user->lVisc2_wm, &visc2_wm);
	DMDAVecRestoreArray(fda, user->lVisc3_wm, &visc3_wm);
	DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);

	DMDALocalToLocalBegin(fda, user->lVisc1_wm, INSERT_VALUES, user->lVisc1_wm);
	DMDALocalToLocalEnd(fda, user->lVisc1_wm, INSERT_VALUES, user->lVisc1_wm);
	DMDALocalToLocalBegin(fda, user->lVisc2_wm, INSERT_VALUES, user->lVisc2_wm);
	DMDALocalToLocalEnd(fda, user->lVisc2_wm, INSERT_VALUES, user->lVisc2_wm);
	DMDALocalToLocalBegin(fda, user->lVisc3_wm, INSERT_VALUES, user->lVisc3_wm);
	DMDALocalToLocalEnd(fda, user->lVisc3_wm, INSERT_VALUES, user->lVisc3_wm);
        DMDALocalToLocalBegin(da, user->lNu_t, INSERT_VALUES, user->lNu_t);
	DMDALocalToLocalEnd(da, user->lNu_t, INSERT_VALUES, user->lNu_t);

	DMDAVecGetArray(fda, user->lVisc1_wm, &visc1_wm);
	DMDAVecGetArray(fda, user->lVisc2_wm, &visc2_wm);
	DMDAVecGetArray(fda, user->lVisc3_wm, &visc3_wm);
        DMDAVecGetArray(da, user->lNu_t, &lnu_t);

	if(periodic)
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		int a=i, b=j, c=k;

		int flag=0;
		
		if(i_periodic && i==0) a=mx-2, flag=1;
		else if(i_periodic && i==mx-1) a=1, flag=1;
		
		if(j_periodic && j==0) b=my-2, flag=1;
		else if(j_periodic && j==my-1) b=1, flag=1;
		
		if(k_periodic && k==0) c=mz-2, flag=1;
		else if(k_periodic && k==mz-1) c=1, flag=1;
		
		if(ii_periodic && i==0) a=-2, flag=1;
		else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
		
		if(jj_periodic && j==0) b=-2, flag=1;
		else if(jj_periodic && j==my-1) b=my+1, flag=1;
		
		if(kk_periodic && k==0) c=-2, flag=1;
		else if(kk_periodic && k==mz-1) c=mz+1, flag=1;
				
		if(flag) {
			visc1_wm[k][j][i] = visc1_wm[c][b][a];
			visc2_wm[k][j][i] = visc2_wm[c][b][a];
			visc3_wm[k][j][i] = visc3_wm[c][b][a];
               	        lnu_t[k][j][i] = lnu_t[c][b][a];
		}
	}

	DMDAVecRestoreArray(fda, user->lVisc1_wm, &visc1_wm);
	DMDAVecRestoreArray(fda, user->lVisc2_wm, &visc2_wm);
	DMDAVecRestoreArray(fda, user->lVisc3_wm, &visc3_wm);
	DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);



	DMDAVecRestoreArray(fda, Coor, &coor);

	DMDAVecRestoreArray(fda, user->lUcat,  &ucat);
	DMDAVecRestoreArray(da, user->lP, &p);

	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);

	DMDAVecRestoreArray(fda, user->lICsi, &icsi);
	DMDAVecRestoreArray(fda, user->lIEta, &ieta);
	DMDAVecRestoreArray(fda, user->lIZet, &izet);

	DMDAVecRestoreArray(fda, user->lJCsi, &jcsi);
	DMDAVecRestoreArray(fda, user->lJEta, &jeta);
	DMDAVecRestoreArray(fda, user->lJZet, &jzet);

	DMDAVecRestoreArray(fda, user->lKCsi, &kcsi);
	DMDAVecRestoreArray(fda, user->lKEta, &keta);
	DMDAVecRestoreArray(fda, user->lKZet, &kzet);

	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	
	DMDAVecRestoreArray(da, user->lAj, &aj);
   
        DMDAVecRestoreArray(da, user->lUstar, &ustar);
    
        DMDAVecRestoreArray(da, user->lTau, &tau);
    
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);
	
	if(levelset) {
		DMDAVecRestoreArray(da, user->lLevelset, &level);
		DMDAVecRestoreArray(da, user->lDensity, &rho);
		DMDAVecRestoreArray(da, user->lMu, &mu);
	}


	return(0);
};

PetscErrorCode Visc_wm_tmprt(UserCtx *user, IBMNodes *ibm)
{
	Vec	Csi = user->lCsi, Eta = user->lEta, Zet = user->lZet;
	Vec 	Ucat=user->lUcat;
	
	Cmpnts	***ucat, ***lucat_o;

	Cmpnts	***csi, ***eta, ***zet;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;

	PetscReal	***nvert;

	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	
	PetscReal	***aj, ***iaj, ***jaj, ***kaj;//, ***vol;

	int	lxs, lxe, lys, lye, lzs, lze;

	PetscReal	dtdc, dtde, dtdz;
	PetscReal	dt_dx, dt_dy, dt_dz;
	PetscReal	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2;
	PetscReal	g11, g21, g31;

        PetscReal 	dtdc_wm, dtde_wm, dtdz_wm;

	Vec Coor;
	Cmpnts ***coor;

	PetscReal ***tmprt, ***pr_t, ***q_scalar;
	PetscReal ***visc1_wmtmprt, ***visc2_wmtmprt, ***visc3_wmtmprt;
	PetscScalar	solid;

	solid = 0.5;
  
	DMDAGetLocalInfo(da, &info);
	mx = info.mx; my = info.my; mz = info.mz;
	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;

  // First we calculate the flux on cell surfaces. Stored on the upper integer
  //   node. For example, along i direction, the flux are stored at node 0:mx-2
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
	DMDAVecGetArray(fda, Csi, &csi);
	DMDAVecGetArray(fda, Eta, &eta);
	DMDAVecGetArray(fda, Zet, &zet);
	DMDAVecGetArray(fda, user->lICsi, &icsi);
	DMDAVecGetArray(fda, user->lIEta, &ieta);
	DMDAVecGetArray(fda, user->lIZet, &izet);
	DMDAVecGetArray(fda, user->lJCsi, &jcsi);
	DMDAVecGetArray(fda, user->lJEta, &jeta);
	DMDAVecGetArray(fda, user->lJZet, &jzet);
	DMDAVecGetArray(fda, user->lKCsi, &kcsi);
	DMDAVecGetArray(fda, user->lKEta, &keta);
	DMDAVecGetArray(fda, user->lKZet, &kzet);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);

	DMDAVecGetArray(fda, Ucat,  &ucat);
	DMDAVecGetArray(da, user->lTmprt, &tmprt);
	if(les_prt) DMDAVecGetArray(da, user->lPr_t, &pr_t);

        DMDAVecGetArray(da, user->lQ_scalar, &q_scalar);
 
	VecSet(user->lVisc1_wmtmprt, 0.0);
	VecSet(user->lVisc2_wmtmprt, 0.0);
	VecSet(user->lVisc3_wmtmprt, 0.0);

	DMDAVecGetArray(da, user->lVisc1_wmtmprt, &visc1_wmtmprt);
	DMDAVecGetArray(da, user->lVisc2_wmtmprt, &visc2_wmtmprt);
	DMDAVecGetArray(da, user->lVisc3_wmtmprt, &visc3_wmtmprt);


	int ibi;

        double area, nu_t;
        double sb, sc, st;
        double nx, ny, nz;
        PetscInt bctype;
        double Ta, Tb, Tc;
        Cmpnts Ua, Ub, Uc;
	PetscReal qs;	

        double ren, nu_t_b,  nu_t_c;

	double Dm, Dt, Dt_b, Dt_c;

	if (IB_wmtmprt) {
		// need to be developed
	}

	Dm = 1./ (user->ren * user->Pr);

        double ni[3], nj[3], nk[3];
	
	// i direction
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {

		if(i==mx-1 || j==my-1 || k==mz-1) continue;
		if(j==0 || k==0) continue;

                if ( imin_wmtmprt != 0 && i == 0 ) {
	                area = sqrt( csi[k][j][i+1].x*csi[k][j][i+1].x + csi[k][j][i+1].y*csi[k][j][i+1].y + csi[k][j][i+1].z*csi[k][j][i+1].z );

			st = sqrt(area);

                        Ta=user->tmprts[0];
			Ua.x=Ua.y=Ua.z=0.0;

                        sc = 2* 0.5/aj[k][j][i+1]/area + 0.5/aj[k][j][i+2]/area;
                        Tc=tmprt[k][j][i+2];

	                sb = 0.5/aj[k][j][i+1]/area;
                        Tb=tmprt[k][j][i+1];
			Ub=ucat[k][j][i+1];

		       	bctype = imin_wmtmprt; 
			Calculate_normal(csi[k][j][i+1], eta[k][j][i+1], zet[k][j][i+1], ni, nj, nk);
			nx =  ni[0], ny =  ni[1], nz =  ni[2];

			Dt_c = pr_t[k][j][i+2];
			Dt_b = pr_t[k][j][i+1];

			num_innergrid = (int)(log(sc*(dhratio_wm-1)/dh1_wm+1.0)/log(dhratio_wm))+1;

                        double *t_in, *z_in, *prt_les, *prt_rans;

                        t_in= (double *) malloc(num_innergrid*sizeof(double));
                        z_in= (double *) malloc(num_innergrid*sizeof(double));
			prt_les= (double *) malloc(num_innergrid*sizeof(double));
			prt_rans= (double *) malloc(num_innergrid*sizeof(double));

                        innergrid( z_in, sc );

//                        double t1x, t1y, t1z, t2x, t2y, t2z;
			wallmodel_tmprt( sb, nx, ny, nz, Ub, Ua, Tb, Ta, bctype, roughness_size, &qs);

			q_scalar[k][j][i+1]=qs;
		
			double ajc = iaj[k][j][i];
			csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
			eta0 = ieta[k][j][i].x, eta1 = ieta[k][j][i].y, eta2 = ieta[k][j][i].z;
			zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;

			Compute_dscalar_i (i, j, k, mx, my, mz, tmprt, nvert, &dtdc, &dtde, &dtdz );

			Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dtdc, dtde, dtdz, &dt_dx, &dt_dy, &dt_dz);

			double dtdn=dt_dx*nx+dt_dy*ny+dt_dz*nz;

			double dtdtau_x=dt_dx-dtdn*nx;
			double dtdtau_y=dt_dy-dtdn*ny;
			double dtdtau_z=dt_dz-dtdn*nz;

                        if ( i==0 || nvert[k][j][i]>0.1 ) Dt = pr_t[k][j][i+1];
                        else if( i==mx-2 || nvert[k][j][i+1]>0.1 ) Dt = pr_t[k][j][i];
                        else Dt = 0.5 * (pr_t[k][j][i] + pr_t[k][j][i+1]);

			double dtdn_wm;
                        if (!infRe) dtdn_wm = -qs/(Dm+Dt);

			if (infRe) dtdn_wm = -qs/(Dm+Dt);
		
			double dtdx_wm=dtdtau_x+dtdn_wm*nx;
			double dtdy_wm=dtdtau_y+dtdn_wm*ny;
			double dtdz_wm=dtdtau_z+dtdn_wm*nz;
			
                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                        Comput_JacobTensor_i(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

			dtdc = dtdx_wm*dxdc+dtdy_wm*dydc+dtdz_wm*dzdc;
			dtde = dtdx_wm*dxde+dtdy_wm*dyde+dtdz_wm*dzde;
			dtdz = dtdx_wm*dxdz+dtdy_wm*dydz+dtdz_wm*dzdz;

			g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
			g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
			g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

			if (infRe) Dm=0.0;

			visc1_wmtmprt[k][j][i] = (g11 * dtdc + g21 * dtde + g31 * dtdz) * ajc * (Dm + Dt);

			free(t_in);
			free(z_in);
			free(prt_les);
			free(prt_rans);
		}

                if ( imax_wmtmprt != 0 && i == mx-2 ) {
	                area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );

			st = sqrt(area);

                        Ta=user->tmprts[1];
			Ua.x=Ua.y=Ua.z=0.0;

                        sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j][i-1]/area;
                        Tc=tmprt[k][j][i-1];

	                sb = 0.5/aj[k][j][i]/area;
                        Tb=tmprt[k][j][i];
			Ub=ucat[k][j][i];

		       	bctype = imax_wmtmprt; 
			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
			nx = -ni[0], ny = -ni[1], nz = -ni[2];

			Dt_c = pr_t[k][j][i-1];
			Dt_b = pr_t[k][j][i];

			num_innergrid = (int)(log(sc*(dhratio_wm-1)/dh1_wm+1.0)/log(dhratio_wm))+1;

                        double *t_in, *z_in, *prt_les, *prt_rans;

                        t_in= (double *) malloc(num_innergrid*sizeof(double));
                        z_in= (double *) malloc(num_innergrid*sizeof(double));
			prt_les= (double *) malloc(num_innergrid*sizeof(double));
			prt_rans= (double *) malloc(num_innergrid*sizeof(double));

                        innergrid( z_in, sc );

			wallmodel_tmprt( sb, nx, ny, nz, Ub, Ua, Tb, Ta, bctype, roughness_size, &qs);

			q_scalar[k][j][i]=qs;
		
			double ajc = iaj[k][j][i];
			csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
			eta0 = ieta[k][j][i].x, eta1 = ieta[k][j][i].y, eta2 = ieta[k][j][i].z;
			zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;

			Compute_dscalar_i (i, j, k, mx, my, mz, tmprt, nvert, &dtdc, &dtde, &dtdz );

			Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dtdc, dtde, dtdz, &dt_dx, &dt_dy, &dt_dz);

			double dtdn=dt_dx*nx+dt_dy*ny+dt_dz*nz;

			double dtdtau_x=dt_dx-dtdn*nx;
			double dtdtau_y=dt_dy-dtdn*ny;
			double dtdtau_z=dt_dz-dtdn*nz;

                        if ( i==0 || nvert[k][j][i]>0.1 ) Dt = pr_t[k][j][i+1];
                        else if( i==mx-2 || nvert[k][j][i+1]>0.1 ) Dt = pr_t[k][j][i];
                        else Dt = 0.5 * (pr_t[k][j][i] + pr_t[k][j][i+1]);

			double dtdn_wm;
                        if (!infRe) dtdn_wm = -qs/(Dm+Dt);

			if (infRe) dtdn_wm = -qs/(Dm+Dt);
		
			double dtdx_wm=dtdtau_x+dtdn_wm*nx;
			double dtdy_wm=dtdtau_y+dtdn_wm*ny;
			double dtdz_wm=dtdtau_z+dtdn_wm*nz;
			
                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                        Comput_JacobTensor_i(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

			dtdc = dtdx_wm*dxdc+dtdy_wm*dydc+dtdz_wm*dzdc;
			dtde = dtdx_wm*dxde+dtdy_wm*dyde+dtdz_wm*dzde;
			dtdz = dtdx_wm*dxdz+dtdy_wm*dydz+dtdz_wm*dzdz;

			g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
			g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
			g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

			if (infRe) Dm=0.0;

			visc1_wmtmprt[k][j][i] = (g11 * dtdc + g21 * dtde + g31 * dtdz) * ajc * (Dm + Dt);

			free(t_in);
			free(z_in);
			free(prt_les);
			free(prt_rans);
			
		}
	
		
	}
  
	// j direction
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		if(i==mx-1 || j==my-1 || k==mz-1) continue;
//		if(i==mx-2 || i==2) continue;
		if(i==0 || k==0) continue;

                if ( jmin_wmtmprt != 0 && j == 0 ) {
	                area = sqrt( eta[k][j+1][i].x*eta[k][j+1][i].x + eta[k][j+1][i].y*eta[k][j+1][i].y + eta[k][j+1][i].z*eta[k][j+1][i].z );

			st = sqrt(area);

                        Ta=user->tmprts[2];
			Ua.x=Ua.y=Ua.z=0.0;

                        sc = 2* 0.5/aj[k][j+1][i]/area + 0.5/aj[k][j+2][i]/area;
                        Tc=tmprt[k][j+2][i];

	                sb = 0.5/aj[k][j+1][i]/area;
                        Tb=tmprt[k][j+1][i];
			Ub=ucat[k][j+1][i];

//	printf("j=%d, Ta=%le, Tb=%le \n", j, Ta, Tb);
//	int aaa;

//	cout << "here \n";
//	cin >> aaa;
		       	bctype = jmin_wmtmprt; 
			Calculate_normal(csi[k][j+1][i], eta[k][j+1][i], zet[k][j+1][i], ni, nj, nk);
			nx = nj[0], ny = nj[1], nz = nj[2];

			Dt_c = pr_t[k][j+2][i];
			Dt_b = pr_t[k][j+1][i];

			num_innergrid = (int)(log(sc*(dhratio_wm-1)/dh1_wm+1.0)/log(dhratio_wm))+1;

                        double *t_in, *z_in, *prt_les, *prt_rans;

                        t_in= (double *) malloc(num_innergrid*sizeof(double));
                        z_in= (double *) malloc(num_innergrid*sizeof(double));
			prt_les= (double *) malloc(num_innergrid*sizeof(double));
			prt_rans= (double *) malloc(num_innergrid*sizeof(double));

                        innergrid( z_in, sc );

			wallmodel_tmprt( sb, nx, ny, nz, Ub, Ua, Tb, Ta, bctype, roughness_size, &qs);

			q_scalar[k][j+1][i]=qs;
		
//	printf("j=%d, Ta=%le, Tb=%le qs=%le \n", j, Ta, Tb, qs);
//	int aaa;

//	cout << "here \n";
//	cin >> aaa;
			double ajc = jaj[k][j][i];
			csi0 = jcsi[k][j][i].x, csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
			eta0 = jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
			zet0 = jzet[k][j][i].x, zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;

			Compute_dscalar_j (i, j, k, mx, my, mz, tmprt, nvert, &dtdc, &dtde, &dtdz );

			Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dtdc, dtde, dtdz, &dt_dx, &dt_dy, &dt_dz);

			double dtdn=dt_dx*nx+dt_dy*ny+dt_dz*nz;

			double dtdtau_x=dt_dx-dtdn*nx;
			double dtdtau_y=dt_dy-dtdn*ny;
			double dtdtau_z=dt_dz-dtdn*nz;

                        if ( j==0 || nvert[k][j][i]>0.1 ) Dt = pr_t[k][j+1][i];
                        else if( j==my-2 || nvert[k][j+1][i]>0.1 ) Dt = pr_t[k][j][i];
                        else Dt = 0.5 * (pr_t[k][j][i] + pr_t[k][j+1][i]);

			double dtdn_wm;
                        if (!infRe) dtdn_wm = -qs/(Dm+Dt);

			if (infRe) dtdn_wm = -qs/(Dm+Dt);
		
			double dtdx_wm=dtdtau_x+dtdn_wm*nx;
			double dtdy_wm=dtdtau_y+dtdn_wm*ny;
			double dtdz_wm=dtdtau_z+dtdn_wm*nz;
			
                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                        Comput_JacobTensor_j(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

			dtdc = dtdx_wm*dxdc+dtdy_wm*dydc+dtdz_wm*dzdc;
			dtde = dtdx_wm*dxde+dtdy_wm*dyde+dtdz_wm*dzde;
			dtdz = dtdx_wm*dxdz+dtdy_wm*dydz+dtdz_wm*dzdz;

			g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
			g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
			g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;

			if (infRe) Dm=0.0;

			visc2_wmtmprt[k][j][i] = (g11 * dtdc + g21 * dtde + g31 * dtdz) * ajc * (Dm + Dt);

			free(t_in);
			free(z_in);
			free(prt_les);
			free(prt_rans);

	
                }


                if ( jmax_wmtmprt != 0 && j == my-2 ) {
	                area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );

			st = sqrt(area);

                        Ta=user->tmprts[3];
			Ua.x=Ua.y=Ua.z=0.0;

                        sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j-1][i]/area;
                        Tc=tmprt[k][j-1][i];

	                sb = 0.5/aj[k][j][i]/area;
                        Tb=tmprt[k][j][i];

			Ub=ucat[k][j][i];

		       	bctype = jmax_wmtmprt; 
			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
			nx = -nj[0], ny = -nj[1], nz = -nj[2];

			Dt_c = pr_t[k][j-1][i];
			Dt_b = pr_t[k][j][i];

			num_innergrid = (int)(log(sc*(dhratio_wm-1)/dh1_wm+1.0)/log(dhratio_wm))+1;

                        double *t_in, *z_in, *prt_les, *prt_rans;

                        t_in= (double *) malloc(num_innergrid*sizeof(double));
                        z_in= (double *) malloc(num_innergrid*sizeof(double));
			prt_les= (double *) malloc(num_innergrid*sizeof(double));
			prt_rans= (double *) malloc(num_innergrid*sizeof(double));

                        innergrid( z_in, sc );

			wallmodel_tmprt( sb, nx, ny, nz, Ub, Ua, Tb, Ta, bctype, roughness_size, &qs);

			q_scalar[k][j][i]=qs;
//	printf("j=%d, Ta=%le, Tb=%le qs=%le \n", j, Ta, Tb, qs);
//	int aaa;

//	cout << "here \n";
//	cin >> aaa;
			double ajc = jaj[k][j][i];
			csi0 = jcsi[k][j][i].x, csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
			eta0 = jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
			zet0 = jzet[k][j][i].x, zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;

			Compute_dscalar_j (i, j, k, mx, my, mz, tmprt, nvert, &dtdc, &dtde, &dtdz );

			Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dtdc, dtde, dtdz, &dt_dx, &dt_dy, &dt_dz);

			double dtdn=dt_dx*nx+dt_dy*ny+dt_dz*nz;

			double dtdtau_x=dt_dx-dtdn*nx;
			double dtdtau_y=dt_dy-dtdn*ny;
			double dtdtau_z=dt_dz-dtdn*nz;

                        if ( j==0 || nvert[k][j][i]>0.1 ) Dt = pr_t[k][j+1][i];
                        else if( j==my-2 || nvert[k][j+1][i]>0.1 ) Dt = pr_t[k][j][i];
                        else Dt = 0.5 * (pr_t[k][j][i] + pr_t[k][j+1][i]);

			double dtdn_wm;
                        if (!infRe) dtdn_wm = -qs/(Dm+Dt);

			if (infRe) dtdn_wm = -qs/(Dm+Dt);
		
			double dtdx_wm=dtdtau_x+dtdn_wm*nx;
			double dtdy_wm=dtdtau_y+dtdn_wm*ny;
			double dtdz_wm=dtdtau_z+dtdn_wm*nz;
			
                        double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                        Comput_JacobTensor_j(i, j, k, mx, my, mz, coor, &dxdc, &dxde, &dxdz, &dydc, &dyde, &dydz, &dzdc, &dzde, &dzdz);

			dtdc = dtdx_wm*dxdc+dtdy_wm*dydc+dtdz_wm*dzdc;
			dtde = dtdx_wm*dxde+dtdy_wm*dyde+dtdz_wm*dzde;
			dtdz = dtdx_wm*dxdz+dtdy_wm*dydz+dtdz_wm*dzdz;

			g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
			g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
			g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;

			if (infRe) Dm=0.0;

			visc2_wmtmprt[k][j][i] = (g11 * dtdc + g21 * dtde + g31 * dtdz) * ajc * (Dm + Dt);

			free(t_in);
			free(z_in);
			free(prt_les);
			free(prt_rans);
                }

	
	}



        for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++) {

                if (jmin_wmtmprt && user->bctype_tmprt[0]==1 && j==0 && i==1) visc2_wmtmprt[k][j][i]=0.0;             
                if (jmin_wmtmprt && user->bctype_tmprt[1]==1 && j==0 && i==mx-2) visc2_wmtmprt[k][j][i]=0.0;                
                if (jmax_wmtmprt && user->bctype_tmprt[0]==1 && j==my-2 && i==1) visc2_wmtmprt[k][j][i]=0.0;                
                if (jmax_wmtmprt && user->bctype_tmprt[1]==1 && j==my-2 && i==mx-2) visc2_wmtmprt[k][j][i]=0.0;                
		
                if (imin_wmtmprt && user->bctype_tmprt[2]==1 && i==0 && j==1) visc1_wmtmprt[k][j][i]=0.0;                
                if (imin_wmtmprt && user->bctype_tmprt[3]==1 && i==0 && j==my-2) visc1_wmtmprt[k][j][i]=0.0;                
                if (imax_wmtmprt && user->bctype_tmprt[2]==1 && i==mx-2 && j==1) visc1_wmtmprt[k][j][i]=0.0;                
                if (imax_wmtmprt && user->bctype_tmprt[3]==1 && i==mx-2 && j==my-2) visc1_wmtmprt[k][j][i]=0.0;
                
        }

	DMDAVecRestoreArray(da, user->lVisc1_wmtmprt, &visc1_wmtmprt);
	DMDAVecRestoreArray(da, user->lVisc2_wmtmprt, &visc2_wmtmprt);
	DMDAVecRestoreArray(da, user->lVisc3_wmtmprt, &visc3_wmtmprt);
	
	DMDALocalToLocalBegin(da, user->lVisc1_wmtmprt, INSERT_VALUES, user->lVisc1_wmtmprt);
	DMDALocalToLocalEnd(da, user->lVisc1_wmtmprt, INSERT_VALUES, user->lVisc1_wmtmprt);
	DMDALocalToLocalBegin(da, user->lVisc2_wmtmprt, INSERT_VALUES, user->lVisc2_wmtmprt);
	DMDALocalToLocalEnd(da, user->lVisc2_wmtmprt, INSERT_VALUES, user->lVisc2_wmtmprt);
	DMDALocalToLocalBegin(da, user->lVisc3_wmtmprt, INSERT_VALUES, user->lVisc3_wmtmprt);
	DMDALocalToLocalEnd(da, user->lVisc3_wmtmprt, INSERT_VALUES, user->lVisc3_wmtmprt);

	
	DMDAVecRestoreArray(fda, Coor, &coor);
	DMDAVecRestoreArray(fda, Csi, &csi);
	DMDAVecRestoreArray(fda, Eta, &eta);
	DMDAVecRestoreArray(fda, Zet, &zet);
	DMDAVecRestoreArray(fda, user->lICsi, &icsi);
	DMDAVecRestoreArray(fda, user->lIEta, &ieta);
	DMDAVecRestoreArray(fda, user->lIZet, &izet);
	DMDAVecRestoreArray(fda, user->lJCsi, &jcsi);
	DMDAVecRestoreArray(fda, user->lJEta, &jeta);
	DMDAVecRestoreArray(fda, user->lJZet, &jzet);
	DMDAVecRestoreArray(fda, user->lKCsi, &kcsi);
	DMDAVecRestoreArray(fda, user->lKEta, &keta);
	DMDAVecRestoreArray(fda, user->lKZet, &kzet);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);

	DMDAVecRestoreArray(fda, Ucat,  &ucat);
	DMDAVecRestoreArray(da, user->lTmprt, &tmprt);
	if(les_prt) DMDAVecRestoreArray(da, user->lPr_t, &pr_t);

        DMDAVecRestoreArray(da, user->lQ_scalar, &q_scalar);
 

	
	DMDAVecGetArray(da, user->lVisc1_wmtmprt, &visc1_wmtmprt);
	DMDAVecGetArray(da, user->lVisc2_wmtmprt, &visc2_wmtmprt);
	DMDAVecGetArray(da, user->lVisc3_wmtmprt, &visc3_wmtmprt);
	
	if(periodic)
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		int a=i, b=j, c=k;

		int flag=0;
		
		if(i_periodic && i==0) a=mx-2, flag=1;
		else if(i_periodic && i==mx-1) a=1, flag=1;
		
		if(j_periodic && j==0) b=my-2, flag=1;
		else if(j_periodic && j==my-1) b=1, flag=1;
		
		if(k_periodic && k==0) c=mz-2, flag=1;
		else if(k_periodic && k==mz-1) c=1, flag=1;
		
		if(ii_periodic && i==0) a=-2, flag=1;
		else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
		
		if(jj_periodic && j==0) b=-2, flag=1;
		else if(jj_periodic && j==my-1) b=my+1, flag=1;
		
		if(kk_periodic && k==0) c=-2, flag=1;
		else if(kk_periodic && k==mz-1) c=mz+1, flag=1;
				
		if(flag) {
			visc1_wmtmprt[k][j][i] = visc1_wmtmprt[c][b][a];
			visc2_wmtmprt[k][j][i] = visc2_wmtmprt[c][b][a];
			visc3_wmtmprt[k][j][i] = visc3_wmtmprt[c][b][a];
		}
	}


        DMDAVecRestoreArray(da, user->lVisc1_wmtmprt, &visc1_wmtmprt);
        DMDAVecRestoreArray(da, user->lVisc2_wmtmprt, &visc2_wmtmprt);
        DMDAVecRestoreArray(da, user->lVisc3_wmtmprt, &visc3_wmtmprt);

	/*
	DMLocalToGlobalBegin(da, user->lVisc1_wmtmprt, INSERT_VALUES, user->Visc1_wmtmprt);
	DMLocalToGlobalEnd(da, user->lVisc1_wmtmprt, INSERT_VALUES, user->Visc1_wmtmprt);
	DMLocalToGlobalBegin(da, user->lVisc2_wmtmprt, INSERT_VALUES, user->Visc2_wmtmprt);
	DMLocalToGlobalEnd(da, user->lVisc2_wmtmprt, INSERT_VALUES, user->Visc2_wmtmprt);
	DMLocalToGlobalBegin(da, user->lVisc3_wmtmprt, INSERT_VALUES, user->Visc3_wmtmprt);
	DMLocalToGlobalEnd(da, user->lVisc3_wmtmprt, INSERT_VALUES, user->Visc3_wmtmprt);
	*/

	extern PetscErrorCode TECIOOut_rhs_da(UserCtx *user, Vec Rhs);

//	TECIOOut_rhs_da(user, user->lVisc2_wmtmprt);
//	exit(0);
//	int aaa;
//	cout << "here \n";
//	cin >> aaa;

//        VecCopy(Visc1_wm, user->Visc1_wm);
//        VecCopy(Visc2_wm, user->Visc2_wm);
//        VecCopy(Visc3_wm, user->Visc3_wm);

//        DAGlobalToLocalBegin(fda, user->Visc1_wm, INSERT_VALUES, user->lVisc1_wm);
//        DAGlobalToLocalEnd(fda, user->Visc1_wm, INSERT_VALUES, user->lVisc1_wm);

//        DAGlobalToLocalBegin(fda, user->Visc2_wm, INSERT_VALUES, user->lVisc2_wm);
//        DAGlobalToLocalEnd(fda, user->Visc2_wm, INSERT_VALUES, user->lVisc2_wm);

//        DAGlobalToLocalBegin(fda, user->Visc3_wm, INSERT_VALUES, user->lVisc3_wm);
//        DAGlobalToLocalEnd(fda, user->Visc3_wm, INSERT_VALUES, user->lVisc3_wm);

	return(0);
};



// must be called from Formfunction_2
PetscErrorCode Formfunction_wm(UserCtx *user, Vec Rhs, double scale)
{
	Vec	Csi = user->lCsi, Eta = user->lEta, Zet = user->lZet;
	Vec 	Ucat=user->lUcat;
	
	Cmpnts	***ucat;

	Cmpnts	***csi, ***eta, ***zet;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;
//	PetscScalar ***p;

	PetscReal ***p;


	PetscReal	***nvert, ***rho, ***mu;

	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	Vec	Visc1, Visc2, Visc3, Fp;
	
	Cmpnts	***visc1, ***visc2, ***visc3, ***fp;
	Cmpnts	***rhs;
	PetscReal	***aj, ***iaj, ***jaj, ***kaj;//, ***vol;

	int	lxs, lxe, lys, lye, lzs, lze;

	PetscReal	dudc, dude, dudz, dvdc, dvde, dvdz, dwdc, dwde, dwdz;
	PetscReal	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2;
	PetscReal	g11, g21, g31;
	PetscReal	r11, r21, r31, r12, r22, r32, r13, r23, r33;

        PetscReal dudc_wm, dvdc_wm, dwdc_wm, dude_wm, dvde_wm, dwde_wm, dudz_wm, dvdz_wm, dwdz_wm;
        PetscReal r11_wm, r21_wm, r31_wm, r12_wm, r22_wm, r32_wm, r13_wm, r23_wm, r33_wm;

	Vec Coor;
	Cmpnts ***coor;

	PetscScalar	solid;

	solid = 0.5;
  
	DMDAGetLocalInfo(da, &info);
	mx = info.mx; my = info.my; mz = info.mz;
	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;

  /* First we calculate the flux on cell surfaces. Stored on the upper integer
     node. For example, along i direction, the flux are stored at node 0:mx-2*/
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;


        DMDAGetCoordinates(da, &Coor);
        DMDAVecGetArray(fda, Coor, &coor);

	DMDAVecGetArray(da, user->lNvert, &nvert);

	DMDAVecGetArray(fda, Rhs,  &rhs);

	DMDAVecGetArray(fda, Csi, &csi);
	DMDAVecGetArray(fda, Eta, &eta);
	DMDAVecGetArray(fda, Zet, &zet);

	
	if(levelset) {
		DMDAVecGetArray(da, user->lDensity, &rho);
		DMDAVecGetArray(da, user->lMu, &mu);
	}
	
	

	Fp = user->Fp;
//	Visc1 = user->Visc1;
//	Visc2 = user->Visc2;
//	Visc3 = user->Visc3;
	
	DMDAVecGetArray(fda, user->lVisc1, &visc1);
	DMDAVecGetArray(fda, user->lVisc2, &visc2);
	DMDAVecGetArray(fda, user->lVisc3, &visc3);
	
	DMDAVecGetArray(da, user->lAj, &aj);
   
	
      
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);

//	Vec Visc1_wm, Visc2_wm, Visc3_wm, Fp_wm;

	Vec Fp_wm;

	Cmpnts ***visc1_wm, ***visc2_wm, ***visc3_wm, ***fp_wm;

//	Visc1_wm=user->Visc1_wm;
//	Visc2_wm=user->Visc2_wm;
//	Visc3_wm=user->Visc3_wm;

	VecDuplicate(Csi, &Fp_wm);

	VecSet(Fp_wm, 0.0);

	DMDAVecGetArray(fda, user->lVisc1_wm, &visc1_wm);
	DMDAVecGetArray(fda, user->lVisc2_wm, &visc2_wm);
	DMDAVecGetArray(fda, user->lVisc3_wm, &visc3_wm);

        for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++) {

                if (fabs(visc1_wm[k][j][i].x)<1.0e-9) visc1_wm[k][j][i].x = visc1[k][j][i].x;
                if (fabs(visc1_wm[k][j][i].y)<1.0e-9) visc1_wm[k][j][i].y = visc1[k][j][i].y;
                if (fabs(visc1_wm[k][j][i].z)<1.0e-9) visc1_wm[k][j][i].z = visc1[k][j][i].z;

                if (fabs(visc2_wm[k][j][i].x)<1.0e-9) visc2_wm[k][j][i].x = visc2[k][j][i].x;
                if (fabs(visc2_wm[k][j][i].y)<1.0e-9) visc2_wm[k][j][i].y = visc2[k][j][i].y;
                if (fabs(visc2_wm[k][j][i].z)<1.0e-9) visc2_wm[k][j][i].z = visc2[k][j][i].z;

                if (fabs(visc3_wm[k][j][i].x)<1.0e-9) visc3_wm[k][j][i].x = visc3[k][j][i].x;
                if (fabs(visc3_wm[k][j][i].y)<1.0e-9) visc3_wm[k][j][i].y = visc3[k][j][i].y;
                if (fabs(visc3_wm[k][j][i].z)<1.0e-9) visc3_wm[k][j][i].z = visc3[k][j][i].z;


//		visc1_wm[k][j][i].x = visc1[k][j][i].x;
//		visc1_wm[k][j][i].y = visc1[k][j][i].y;
//		visc1_wm[k][j][i].z = visc1[k][j][i].z;

//                visc2_wm[k][j][i].x = visc2[k][j][i].x;
//                visc2_wm[k][j][i].y = visc2[k][j][i].y;
//                visc2_wm[k][j][i].z = visc2[k][j][i].z;

//                visc3_wm[k][j][i].x = visc3[k][j][i].x;
//                visc3_wm[k][j][i].y = visc3[k][j][i].y;
//                visc3_wm[k][j][i].z = visc3[k][j][i].z;

	}

	
	DMDAVecGetArray(fda, Fp, &fp);
	DMDAVecGetArray(fda, Fp_wm, &fp_wm);
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		 
		double r=1.0;
			
		if(levelset) {
			r = rho[k][j][i];
		}
		
		if ( inviscid ) {}
		else {
			fp[k][j][i].x = (visc1[k][j][i].x - visc1[k][j][i-1].x + visc2[k][j][i].x - visc2[k][j-1][i].x + visc3[k][j][i].x - visc3[k-1][j][i].x) / r;
			fp[k][j][i].y = (visc1[k][j][i].y - visc1[k][j][i-1].y + visc2[k][j][i].y - visc2[k][j-1][i].y + visc3[k][j][i].y - visc3[k-1][j][i].y) / r;
			fp[k][j][i].z = (visc1[k][j][i].z - visc1[k][j][i-1].z + visc2[k][j][i].z - visc2[k][j-1][i].z + visc3[k][j][i].z - visc3[k-1][j][i].z) / r;


			fp_wm[k][j][i].x = (visc1_wm[k][j][i].x - visc1_wm[k][j][i-1].x + visc2_wm[k][j][i].x - visc2_wm[k][j-1][i].x + visc3_wm[k][j][i].x - visc3_wm[k-1][j][i].x) / r;
			fp_wm[k][j][i].y = (visc1_wm[k][j][i].y - visc1_wm[k][j][i-1].y + visc2_wm[k][j][i].y - visc2_wm[k][j-1][i].y + visc3_wm[k][j][i].y - visc3_wm[k-1][j][i].y) / r;
			fp_wm[k][j][i].z = (visc1_wm[k][j][i].z - visc1_wm[k][j][i-1].z + visc2_wm[k][j][i].z - visc2_wm[k][j-1][i].z + visc3_wm[k][j][i].z - visc3_wm[k-1][j][i].z) / r;
		}
	
	}
	
	DMDAVecRestoreArray(fda, Fp, &fp);
	
	DMDALocalToLocalBegin(fda, Fp, INSERT_VALUES, Fp);
	DMDALocalToLocalEnd(fda, Fp, INSERT_VALUES, Fp);
	
        DMDAVecRestoreArray(fda, Fp_wm, &fp_wm);

        DMDALocalToLocalBegin(fda, Fp_wm, INSERT_VALUES, Fp_wm);
        DMDALocalToLocalEnd(fda, Fp_wm, INSERT_VALUES, Fp_wm);



	DMDAVecGetArray(fda, Fp, &fp);
	DMDAVecGetArray(fda, Fp_wm, &fp_wm);
	
	if(periodic)
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		int flag=0, a=i, b=j, c=k;
                
		if(i_periodic && i==0) a=mx-2, flag=1;
		else if(i_periodic && i==mx-1) a=1, flag=1;
		
		if(j_periodic && j==0) b=my-2, flag=1;
		else if(j_periodic && j==my-1) b=1, flag=1;
		
		if(k_periodic && k==0) c=mz-2, flag=1;
		else if(k_periodic && k==mz-1) c=1, flag=1;
		
		if(ii_periodic && i==0) a=-2, flag=1;
		else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
		
		if(jj_periodic && j==0) b=-2, flag=1;
		else if(jj_periodic && j==my-1) b=my+1, flag=1;
		
		if(kk_periodic && k==0) c=-2, flag=1;
		else if(kk_periodic && k==mz-1) c=mz+1, flag=1;
		
		if(flag) fp[k][j][i] = fp[c][b][a];
		if(flag) fp_wm[k][j][i] = fp_wm[c][b][a];

	}

	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
				
		
		rhs[k][j][i].x = -scale * ( 0.5 * ( csi[k][j][i].x * fp[k][j][i].x + csi[k][j][i].y * fp[k][j][i].y + csi[k][j][i].z * fp[k][j][i].z) + 0.5 * ( csi[k][j][i+1].x * fp[k][j][i+1].x + csi[k][j][i+1].y * fp[k][j][i+1].y + csi[k][j][i+1].z * fp[k][j][i+1].z) ) * iaj[k][j][i];
		rhs[k][j][i].y = -scale * ( 0.5 * ( eta[k][j][i].x * fp[k][j][i].x + eta[k][j][i].y * fp[k][j][i].y + eta[k][j][i].z * fp[k][j][i].z) + 0.5 * ( eta[k][j+1][i].x * fp[k][j+1][i].x + eta[k][j+1][i].y * fp[k][j+1][i].y + eta[k][j+1][i].z * fp[k][j+1][i].z) ) * jaj[k][j][i];
		rhs[k][j][i].z = -scale * ( 0.5 * ( zet[k][j][i].x * fp[k][j][i].x + zet[k][j][i].y * fp[k][j][i].y + zet[k][j][i].z * fp[k][j][i].z) + 0.5 * ( zet[k+1][j][i].x * fp[k+1][j][i].x + zet[k+1][j][i].y * fp[k+1][j][i].y + zet[k+1][j][i].z * fp[k+1][j][i].z) ) * kaj[k][j][i];
		

                rhs[k][j][i].x += scale * ( 0.5 * ( csi[k][j][i].x * fp_wm[k][j][i].x + csi[k][j][i].y * fp_wm[k][j][i].y + csi[k][j][i].z * fp_wm[k][j][i].z) + 0.5 * ( csi[k][j][i+1].x * fp_wm[k][j][i+1].x + csi[k][j][i+1].y * fp_wm[k][j][i+1].y + csi[k][j][i+1].z * fp_wm[k][j][i+1].z) ) * iaj[k][j][i];
                rhs[k][j][i].y += scale * ( 0.5 * ( eta[k][j][i].x * fp_wm[k][j][i].x + eta[k][j][i].y * fp_wm[k][j][i].y + eta[k][j][i].z * fp_wm[k][j][i].z) + 0.5 * ( eta[k][j+1][i].x * fp_wm[k][j+1][i].x + eta[k][j+1][i].y * fp_wm[k][j+1][i].y + eta[k][j+1][i].z * fp_wm[k][j+1][i].z) ) * jaj[k][j][i];
                rhs[k][j][i].z += scale * ( 0.5 * ( zet[k][j][i].x * fp_wm[k][j][i].x + zet[k][j][i].y * fp_wm[k][j][i].y + zet[k][j][i].z * fp_wm[k][j][i].z) + 0.5 * ( zet[k+1][j][i].x * fp_wm[k+1][j][i].x + zet[k+1][j][i].y * fp_wm[k+1][j][i].y + zet[k+1][j][i].z * fp_wm[k+1][j][i].z) ) * kaj[k][j][i];

		
		if(nvert[k][j][i]+nvert[k][j][i+1]>0.1 || (!i_periodic && !ii_periodic && i==mx-2) ) {
			rhs[k][j][i].x = 0;
		}
		if(nvert[k][j][i]+nvert[k][j+1][i]>0.1 || (!j_periodic && !jj_periodic && j==my-2) ) {
			rhs[k][j][i].y = 0;
		}
		if(nvert[k][j][i]+nvert[k+1][j][i]>0.1 || (!k_periodic && !kk_periodic && k==mz-2) ) {
			rhs[k][j][i].z = 0;
		}

	}

	
        DMDAVecRestoreArray(da, user->lIAj, &iaj);
        DMDAVecRestoreArray(da, user->lJAj, &jaj);
        DMDAVecRestoreArray(da, user->lKAj, &kaj);

	if (xs ==0) {
		i = 0;
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++) {
			rhs[k][j][i].x = 0;
			rhs[k][j][i].y = 0;
			rhs[k][j][i].z = 0;

		}
	}

	if (xe == mx) {
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++) {
			if(!i_periodic && !ii_periodic) {
				i = mx-2;
				rhs[k][j][i].x = 0;
			}
			i = mx-1;
			rhs[k][j][i].x = 0;
			rhs[k][j][i].y = 0;
			rhs[k][j][i].z = 0;



		}
	}


	if (ys == 0) {
		for (k=zs; k<ze; k++)
		for (i=xs; i<xe; i++) {
			j=0;
			rhs[k][j][i].x = 0;
			rhs[k][j][i].y = 0;
			rhs[k][j][i].z = 0;
		}
	}
  
	if (ye == my) {
		for (k=zs; k<ze; k++) 
		for (i=xs; i<xe; i++) {
			if(!j_periodic && !jj_periodic) {
				j=my-2;
				rhs[k][j][i].y = 0;
			}
			j=my-1;
			rhs[k][j][i].x = 0;
			rhs[k][j][i].y = 0;
			rhs[k][j][i].z = 0;
		}
	}
	
	
	if (zs == 0) {
		k=0;
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
			rhs[k][j][i].x = 0;
			rhs[k][j][i].y = 0;
			rhs[k][j][i].z = 0;


		}
	}
  
	if (ze == mz) {
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
			if(!k_periodic && !kk_periodic) {
				k=mz-2;
				rhs[k][j][i].z = 0;
			}
			k=mz-1;
			rhs[k][j][i].x = 0;
			rhs[k][j][i].y = 0;
			rhs[k][j][i].z = 0;


		}
	}

	DMDAVecRestoreArray(fda, Rhs,  &rhs);

	DMDAVecRestoreArray(fda, Csi, &csi);
	DMDAVecRestoreArray(fda, Eta, &eta);
	DMDAVecRestoreArray(fda, Zet, &zet);
	
	
	if(levelset) {
		DMDAVecRestoreArray(da, user->lDensity, &rho);
		DMDAVecRestoreArray(da, user->lMu, &mu);
	}
	
  
	DMDAVecRestoreArray(fda, Fp, &fp);
	
	DMDAVecRestoreArray(fda, user->lVisc1, &visc1);
	DMDAVecRestoreArray(fda, user->lVisc2, &visc2);
	DMDAVecRestoreArray(fda, user->lVisc3, &visc3);
	
	DMDAVecRestoreArray(da, user->lAj, &aj);
  
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	
        DMDAVecRestoreArray(fda, user->lVisc1_wm, &visc1_wm);
        DMDAVecRestoreArray(fda, user->lVisc2_wm, &visc2_wm);
        DMDAVecRestoreArray(fda, user->lVisc3_wm, &visc3_wm);
        DMDAVecRestoreArray(fda, Fp_wm, &fp_wm);

        DMDAVecRestoreArray(fda, Coor, &coor);

        VecDestroy(&Fp_wm);

//        TECIOOut_rhs(user, Rhs );

//        int aaa;
//        cout << "here \n";
//        cin >> aaa;



//	TECIOOut_rhs(user, user->lVisc2_wm );

	return(0);
};


// must be called from Formfunction_2
PetscErrorCode Formfunction_wm_tmprt(UserCtx *user, Vec Rhs, double scale)
{
	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	
	PetscReal	***visc1, ***visc2, ***visc3;
	PetscReal 	***visc1_wm, ***visc2_wm, ***visc3_wm;
	PetscReal	***rhs;
	PetscReal	***aj;

	PetscReal	***rho, ***mu;

	int	lxs, lxe, lys, lye, lzs, lze;

	PetscReal	***nvert;
	Vec Coor;
	Cmpnts 		***coor;

	PetscScalar	solid;

	solid = 0.5;
  
	DMDAGetLocalInfo(da, &info);
	mx = info.mx; my = info.my; mz = info.mz;
	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;

  /* First we calculate the flux on cell surfaces. Stored on the upper integer
     node. For example, along i direction, the flux are stored at node 0:mx-2*/
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;


        DMDAGetCoordinates(da, &Coor);

        DMDAVecGetArray(fda, Coor, &coor);
	DMDAVecGetArray(da, Rhs,  &rhs);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	
	if(levelset) {
		DMDAVecGetArray(da, user->lDensity, &rho);
		DMDAVecGetArray(da, user->lMu, &mu);
	}
	
	DMDAVecGetArray(da, user->lVisc1_tmprt, &visc1);
	DMDAVecGetArray(da, user->lVisc2_tmprt, &visc2);
	DMDAVecGetArray(da, user->lVisc3_tmprt, &visc3);
	
	DMDAVecGetArray(da, user->lAj, &aj);

	DMDAVecGetArray(da, user->lVisc1_wmtmprt, &visc1_wm);
	DMDAVecGetArray(da, user->lVisc2_wmtmprt, &visc2_wm);
	DMDAVecGetArray(da, user->lVisc3_wmtmprt, &visc3_wm);

        for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++) {
                if (fabs(visc1_wm[k][j][i])<1.0e-9) visc1_wm[k][j][i] = visc1[k][j][i];
                if (fabs(visc2_wm[k][j][i])<1.0e-9) visc2_wm[k][j][i] = visc2[k][j][i];
                if (fabs(visc3_wm[k][j][i])<1.0e-9) visc3_wm[k][j][i] = visc3[k][j][i];
	}
	
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
				
		double ajc = aj[k][j][i];

		double r = 1.;
			
		if(levelset) r = rho[k][j][i];
			
		rhs[k][j][i] -= ( visc1[k][j][i] - visc1[k][j][i-1] + visc2[k][j][i] - visc2[k][j-1][i] + visc3[k][j][i] - visc3[k-1][j][i]) * ajc / r;
		rhs[k][j][i] += ( visc1_wm[k][j][i] - visc1_wm[k][j][i-1] + visc2_wm[k][j][i] - visc2_wm[k][j-1][i] + visc3_wm[k][j][i] - visc3_wm[k-1][j][i]) * ajc / r;

/*		
		if(nvert[k][j][i]+nvert[k][j][i+1]>0.1 || (!i_periodic && !ii_periodic && i==mx-2) ) {
			rhs[k][j][i] = 0;
		}
		if(nvert[k][j][i]+nvert[k][j+1][i]>0.1 || (!j_periodic && !jj_periodic && j==my-2) ) {
			rhs[k][j][i] = 0;
		}
		if(nvert[k][j][i]+nvert[k+1][j][i]>0.1 || (!k_periodic && !kk_periodic && k==mz-2) ) {
			rhs[k][j][i] = 0;
		}
*/
	}

	if (xs ==0) {
		i = 0;
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++) {
			rhs[k][j][i] = 0;
		}
	}

	if (xe == mx) {
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++) {
//			if(!i_periodic && !ii_periodic) {
//				i = mx-2;
//				rhs[k][j][i] = 0;
//			}
			i = mx-1;
			rhs[k][j][i] = 0;
		}
	}

	if (ys == 0) {
		for (k=zs; k<ze; k++)
		for (i=xs; i<xe; i++) {
			j=0;
			rhs[k][j][i] = 0;
		}
	}
  
	if (ye == my) {
		for (k=zs; k<ze; k++) 
		for (i=xs; i<xe; i++) {
//			if(!j_periodic && !jj_periodic) {
//				j=my-2;
//				rhs[k][j][i] = 0;
//			}
			j=my-1;
			rhs[k][j][i] = 0;
		}
	}
	
	
	if (zs == 0) {
		k=0;
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
			rhs[k][j][i] = 0;
		}
	}
  
	if (ze == mz) {
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
//			if(!k_periodic && !kk_periodic) {
//				k=mz-2;
//				rhs[k][j][i] = 0;
//			}
			k=mz-1;
			rhs[k][j][i] = 0;
		}
	}

        DMDAVecRestoreArray(fda, Coor, &coor);
	DMDAVecRestoreArray(da, Rhs,  &rhs);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	
	if(levelset) {
		DMDAVecRestoreArray(da, user->lDensity, &rho);
		DMDAVecRestoreArray(da, user->lMu, &mu);
	}
	
	DMDAVecRestoreArray(da, user->lVisc1_tmprt, &visc1);
	DMDAVecRestoreArray(da, user->lVisc2_tmprt, &visc2);
	DMDAVecRestoreArray(da, user->lVisc3_tmprt, &visc3);
	
	DMDAVecRestoreArray(da, user->lAj, &aj);

	DMDAVecRestoreArray(da, user->lVisc1_wmtmprt, &visc1_wm);
	DMDAVecRestoreArray(da, user->lVisc2_wmtmprt, &visc2_wm);
	DMDAVecRestoreArray(da, user->lVisc3_wmtmprt, &visc3_wm);


//        TECIOOut_rhs(user, Rhs );

//        int aaa;
//        cout << "here \n";
//        cin >> aaa;



//	TECIOOut_rhs(user, user->lVisc2_wm );

	return(0);
};


// From computational grid to local wall model grid

void Comput_du_wmlocal(double nx, double ny, double nz, double t1x, double t1y, double t1z, double t2x, double t2y, double t2z, double du_dx,double dv_dx,double dw_dx,double du_dy,double dv_dy,double dw_dy,double du_dz,double dv_dz,double dw_dz, double *dut1dn, double *dut2dn, double *dundn, double *dut1dt1, double *dut2dt1, double *dundt1, double *dut1dt2, double *dut2dt2, double *dundt2) {

	double dudn = du_dx*nx+du_dy*ny+du_dz*nz;	
	double dvdn = dv_dx*nx+dv_dy*ny+dv_dz*nz;	
	double dwdn = dw_dx*nx+dw_dy*ny+dw_dz*nz;	

	double dudt1 = du_dx*t1x+du_dy*t1y+du_dz*t1z;	
	double dvdt1 = dv_dx*t1x+dv_dy*t1y+dv_dz*t1z;	
	double dwdt1 = dw_dx*t1x+dw_dy*t1y+dw_dz*t1z;	

        double dudt2 = du_dx*t2x+du_dy*t2y+du_dz*t2z;
        double dvdt2 = dv_dx*t2x+dv_dy*t2y+dv_dz*t2z;
        double dwdt2 = dw_dx*t2x+dw_dy*t2y+dw_dz*t2z;


	*dut1dn=dudn*t1x+dvdn*t1y+dwdn*t1z;	
	*dut2dn=dudn*t2x+dvdn*t2y+dwdn*t2z;	
	*dundn=dudn*nx+dvdn*ny+dwdn*nz;	

        *dut1dt1=dudt1*t1x+dvdt1*t1y+dwdt1*t1z;
        *dut2dt1=dudt1*t2x+dvdt1*t2y+dwdt1*t2z;
        *dundt1=dudt1*nx+dvdt1*ny+dwdt1*nz;


        *dut1dt2=dudt2*t1x+dvdt2*t1y+dwdt2*t1z;
        *dut2dt2=dudt2*t2x+dvdt2*t2y+dwdt2*t2z;
        *dundt2=dudt2*nx+dvdt2*ny+dwdt2*nz;

}


// From local wall model grid to computational grid
void Comput_du_Compgrid(double dxdc, double dxde, double dxdz, double dydc, double dyde, double dydz, double dzdc, double dzde, double dzdz, double nx, double ny, double nz, double t1x, double t1y, double t1z, double t2x, double t2y, double t2z, double dut1dn, double dut2dn, double dundn, double dut1dt1, double dut2dt1, double dundt1, double dut1dt2, double dut2dt2, double dundt2, double *dudc, double *dvdc, double *dwdc, double *dude, double *dvde, double *dwde, double *dudz, double *dvdz, double *dwdz) {

	double dxdn=nx, dydn=ny, dzdn=nz;
	double dxdt1=t1x, dydt1=t1y, dzdt1=t1z;
	double dxdt2=t2x, dydt2=t2y, dzdt2=t2z;

	double dndx = dydt1*dzdt2-dydt2*dzdt1;
	double dt1dx = dydt2*dzdn-dydn*dzdt2;
	double dt2dx = dydn*dzdt1-dydt1*dzdn;

        double dndy = dzdt1*dxdt2-dzdt2*dxdt1;
        double dt1dy = dzdt2*dxdn-dzdn*dxdt2;
        double dt2dy = dzdn*dxdt1-dzdt1*dxdn;

        double dndz = dxdt1*dydt2-dxdt2*dydt1;
        double dt1dz = dxdt2*dydn-dxdn*dydt2;
        double dt2dz = dxdn*dydt1-dxdt1*dydn;


	double dundx = dundn*dndx+dundt1*dt1dx+dundt2*dt2dx;
	double dundy = dundn*dndy+dundt1*dt1dy+dundt2*dt2dy;
	double dundz = dundn*dndz+dundt1*dt1dz+dundt2*dt2dz;

        double dut1dx = dut1dn*dndx+dut1dt1*dt1dx+dut1dt2*dt2dx;
        double dut1dy = dut1dn*dndy+dut1dt1*dt1dy+dut1dt2*dt2dy;
        double dut1dz = dut1dn*dndz+dut1dt1*dt1dz+dut1dt2*dt2dz;

        double dut2dx = dut2dn*dndx+dut2dt1*dt1dx+dut2dt2*dt2dx;
        double dut2dy = dut2dn*dndy+dut2dt1*dt1dy+dut2dt2*dt2dy;
        double dut2dz = dut2dn*dndz+dut2dt1*dt1dz+dut2dt2*dt2dz;


	double du_dx = dundx*nx+dut1dx*t1x+dut2dx*t2x;
	double du_dy = dundy*nx+dut1dy*t1x+dut2dy*t2x;
	double du_dz = dundz*nx+dut1dz*t1x+dut2dz*t2x;

        double dv_dx = dundx*ny+dut1dx*t1y+dut2dx*t2y;
        double dv_dy = dundy*ny+dut1dy*t1y+dut2dy*t2y;
        double dv_dz = dundz*ny+dut1dz*t1y+dut2dz*t2y;

        double dw_dx = dundx*nz+dut1dx*t1z+dut2dx*t2z;
        double dw_dy = dundy*nz+dut1dy*t1z+dut2dy*t2z;
        double dw_dz = dundz*nz+dut1dz*t1z+dut2dz*t2z;

	*dudc = du_dx*dxdc+du_dy*dydc+du_dz*dzdc;
	*dude = du_dx*dxde+du_dy*dyde+du_dz*dzde;
	*dudz = du_dx*dxdz+du_dy*dydz+du_dz*dzdz;

        *dvdc = dv_dx*dxdc+dv_dy*dydc+dv_dz*dzdc;
        *dvde = dv_dx*dxde+dv_dy*dyde+dv_dz*dzde;
        *dvdz = dv_dx*dxdz+dv_dy*dydz+dv_dz*dzdz;

        *dwdc = dw_dx*dxdc+dw_dy*dydc+dw_dz*dzdc;
        *dwde = dw_dx*dxde+dw_dy*dyde+dw_dz*dzde;
        *dwdz = dw_dx*dxdz+dw_dy*dydz+dw_dz*dzdz;



}

void Comput_JacobTensor_i(int i, int j, int k, int mx, int my, int mz, Cmpnts ***coor, double *dxdc, double *dxde, double *dxdz, double *dydc, double *dyde, double *dydz, double *dzdc, double *dzde, double *dzdz) {

        double centx, centy, centz;
        double centx_ip1, centy_ip1, centz_ip1;
        double centx_im1, centy_im1, centz_im1;
        double centx_jp1, centy_jp1, centz_jp1;
        double centx_jm1, centy_jm1, centz_jm1;
        double centx_kp1, centy_kp1, centz_kp1;
        double centx_km1, centy_km1, centz_km1;


	int i1=i,j1=j,k1=k;	

        centx = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
        centy = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
        centz = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;

	if (i!=mx-2) {
	        i1=i+1,j1=j,k1=k;

        	centx_ip1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                	     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
	        centy_ip1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
        	             coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
	        centz_ip1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
        	             coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
	}

	if (i!=0) {
	        i1=i-1,j1=j,k1=k;

	        centx_im1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
        	             coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
	        centy_im1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
        	             coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
	        centz_im1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
        	             coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
	}

	if (j!=my-2) {
        	i1=i,j1=j+1,k1=k;

	        centx_jp1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
        	             coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
	        centy_jp1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
        	             coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
	        centz_jp1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
        	             coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
	}

	if (j!=1) {
	        i1=i,j1=j-1,k1=k;

	        centx_jm1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
        	             coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
	        centy_jm1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                	     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
        	centz_jm1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
	                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
	}

	if (k!=mz-2) {
        	i1=i,j1=j,k1=k+1;

        	centx_kp1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                	     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
	        centy_kp1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
        	             coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
	        centz_kp1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
        	             coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
	}

	if (k!=1) {
        	i1=i,j1=j,k1=k-1;

	        centx_km1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
        	             coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
	        centy_km1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                	     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
	        centz_km1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
	                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
	}
	
	if (i==0) {
	  *dxdc = centx_ip1 - centx;
	  *dydc = centy_ip1 - centy;
	  *dzdc = centz_ip1 - centz;
	}
	else if (i==mx-2) {
	  *dxdc = centx - centx_im1;
	  *dydc = centy - centy_im1;
	  *dzdc = centz - centz_im1;
	}
	else {
	  *dxdc = (centx_ip1 - centx_im1) * 0.5;
	  *dydc = (centy_ip1 - centy_im1) * 0.5;
	  *dzdc = (centz_ip1 - centz_im1) * 0.5;
	}

	
	if (j==1) {
	  *dxde = centx_jp1 - centx;
	  *dyde = centy_jp1 - centy;
	  *dzde = centz_jp1 - centz;
	}
	else if (j==my-2) {
	  *dxde = centx - centx_jm1;
	  *dyde = centy - centy_jm1;
	  *dzde = centz - centz_jm1;
	}
	else {
	  *dxde = (centx_jp1 - centx_jm1) * 0.5;
	  *dyde = (centy_jp1 - centy_jm1) * 0.5;
	  *dzde = (centz_jp1 - centz_jm1) * 0.5;
	}
	
	if (k==1) {
	  *dxdz = (centx_kp1 - centx);
	  *dydz = (centy_kp1 - centy);
	  *dzdz = (centz_kp1 - centz);
	}
	else if (k==mz-2) {
	  *dxdz = (centx - centx_km1);
	  *dydz = (centy - centy_km1);
	  *dzdz = (centz - centz_km1);
	}
	else {
	  *dxdz = (centx_kp1 - centx_km1) * 0.5;
	  *dydz = (centy_kp1 - centy_km1) * 0.5;
	  *dzdz = (centz_kp1 - centz_km1) * 0.5;
	}
	
}



void Comput_JacobTensor_j(int i, int j, int k, int mx, int my, int mz, Cmpnts ***coor, double *dxdc, double *dxde, double *dxdz, double *dydc, double *dyde, double *dydz, double *dzdc, double *dzde, double *dzdz) {
	
	double centx, centy, centz;
	double centx_ip1, centy_ip1, centz_ip1;
	double centx_im1, centy_im1, centz_im1;
	double centx_jp1, centy_jp1, centz_jp1;
	double centx_jm1, centy_jm1, centz_jm1;
	double centx_kp1, centy_kp1, centz_kp1;
	double centx_km1, centy_km1, centz_km1;

	int i1=i,j1=j,k1=k;	
        centx = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                 coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
        centy = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                 coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
        centz = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                 coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

	if (i!=mx-2) {
	        i1=i+1,j1=j,k1=k;

        	centx_ip1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                        coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
	        centy_ip1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                        coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
	        centz_ip1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                        coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

	}

	if (i!=1) {
	        i1=i-1,j1=j,k1=k;

                centx_im1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                        coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
                centy_im1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                        coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
                centz_im1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                        coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

	}

	if (j!=my-2) {
        	i1=i,j1=j+1,k1=k;

                centx_jp1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                        coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
                centy_jp1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                        coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
                centz_jp1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                        coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

	}

	if (j!=0) {
	        i1=i,j1=j-1,k1=k;

                centx_jm1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                             coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
                centy_jm1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                        coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
                centz_jm1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                        coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

	}

	if (k!=mz-2) {
        	i1=i,j1=j,k1=k+1;

                centx_kp1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                        coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
                centy_kp1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                        coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
                centz_kp1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                        coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

	}

	if (k!=1) {
        	i1=i,j1=j,k1=k-1;

                centx_km1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                        coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
                centy_km1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                        coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
                centz_km1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                        coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

	}
	
	if (i==1) {
	  *dxdc = centx_ip1 - centx;
	  *dydc = centy_ip1 - centy;
	  *dzdc = centz_ip1 - centz;
	}
	else if (i==mx-2) {
	  *dxdc = centx - centx_im1;
	  *dydc = centy - centy_im1;
	  *dzdc = centz - centz_im1;
	}
	else {
	  *dxdc = (centx_ip1 - centx_im1) * 0.5;
	  *dydc = (centy_ip1 - centy_im1) * 0.5;
	  *dzdc = (centz_ip1 - centz_im1) * 0.5;
	}

	
	if (j==0) {
	  *dxde = centx_jp1 - centx;
	  *dyde = centy_jp1 - centy;
	  *dzde = centz_jp1 - centz;
	}
	else if (j==my-2) {
	  *dxde = centx - centx_jm1;
	  *dyde = centy - centy_jm1;
	  *dzde = centz - centz_jm1;
	}
	else {
	  *dxde = (centx_jp1 - centx_jm1) * 0.5;
	  *dyde = (centy_jp1 - centy_jm1) * 0.5;
	  *dzde = (centz_jp1 - centz_jm1) * 0.5;
	}
	
	if (k==1) {
	  *dxdz = (centx_kp1 - centx);
	  *dydz = (centy_kp1 - centy);
	  *dzdz = (centz_kp1 - centz);
	}
	else if (k==mz-2) {
	  *dxdz = (centx - centx_km1);
	  *dydz = (centy - centy_km1);
	  *dzdz = (centz - centz_km1);
	}
	else {
	  *dxdz = (centx_kp1 - centx_km1) * 0.5;
	  *dydz = (centy_kp1 - centy_km1) * 0.5;
	  *dzdz = (centz_kp1 - centz_km1) * 0.5;
	}
	
}



void Comput_JacobTensor_k(int i, int j, int k, int mx, int my, int mz, Cmpnts ***coor, double *dxdc, double *dxde, double *dxdz, double *dydc, double *dyde, double *dydz, double *dzdc, double *dzde, double *dzdz) {
	
	double centx, centy, centz;
	double centx_ip1, centy_ip1, centz_ip1;
	double centx_im1, centy_im1, centz_im1;
	double centx_jp1, centy_jp1, centz_jp1;
	double centx_jm1, centy_jm1, centz_jm1;
	double centx_kp1, centy_kp1, centz_kp1;
	double centx_km1, centy_km1, centz_km1;

	int i1=i,j1=j,k1=k;

        centx = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                 coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
        centy = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                 coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
        centz = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                 coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

	if (i!=mx-2) {
	        i1=i+1,j1=j,k1=k;

	        centx_ip1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                 	     coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
        	centy_ip1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                 	     coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
        	centz_ip1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                 	     coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

	}

	if (i!=1) {
	        i1=i-1,j1=j,k1=k;

                centx_im1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                             coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
                centy_im1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                             coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
                centz_im1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                             coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

	}

	if (j!=my-2) {
        	i1=i,j1=j+1,k1=k;

                centx_jp1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                             coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
                centy_jp1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                             coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
                centz_jp1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                             coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

	}

	if (j!=1) {
	        i1=i,j1=j-1,k1=k;

                centx_jm1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                             coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
                centy_jm1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                             coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
                centz_jm1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                             coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

	}

	if (k!=mz-2) {
        	i1=i,j1=j,k1=k+1;

                centx_kp1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                             coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
                centy_kp1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                             coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
                centz_kp1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                             coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

	}

	if (k!=0) {
        	i1=i,j1=j,k1=k-1;

                centx_km1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                             coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
                centy_km1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                             coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
                centz_km1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                             coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

	}
	
	if (i==1) {
	  *dxdc = centx_ip1 - centx;
	  *dydc = centy_ip1 - centy;
	  *dzdc = centz_ip1 - centz;
	}
	else if (i==mx-2) {
	  *dxdc = centx - centx_im1;
	  *dydc = centy - centy_im1;
	  *dzdc = centz - centz_im1;
	}
	else {
	  *dxdc = (centx_ip1 - centx_im1) * 0.5;
	  *dydc = (centy_ip1 - centy_im1) * 0.5;
	  *dzdc = (centz_ip1 - centz_im1) * 0.5;
	}

	
	if (j==1) {
	  *dxde = centx_jp1 - centx;
	  *dyde = centy_jp1 - centy;
	  *dzde = centz_jp1 - centz;
	}
	else if (j==my-2) {
	  *dxde = centx - centx_jm1;
	  *dyde = centy - centy_jm1;
	  *dzde = centz - centz_jm1;
	}
	else {
	  *dxde = (centx_jp1 - centx_jm1) * 0.5;
	  *dyde = (centy_jp1 - centy_jm1) * 0.5;
	  *dzde = (centz_jp1 - centz_jm1) * 0.5;
	}
	
	if (k==0) {
	  *dxdz = (centx_kp1 - centx);
	  *dydz = (centy_kp1 - centy);
	  *dzdz = (centz_kp1 - centz);
	}
	else if (k==mz-2) {
	  *dxdz = (centx - centx_km1);
	  *dydz = (centy - centy_km1);
	  *dzdz = (centz - centz_km1);
	}
	else {
	  *dxdz = (centx_kp1 - centx_km1) * 0.5;
	  *dydz = (centy_kp1 - centy_km1) * 0.5;
	  *dzdz = (centz_kp1 - centz_km1) * 0.5;
	}
	
}



void Compute1_du_i (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
{
	
	if ((nvert[k][j][i])> 1.1 || (nvert[k][j][i+1])> 1.1) {
		*dudc = 0.0;
                *dvdc = 0.0;
                *dwdc = 0.0;

                *dude = 0.0;
                *dvde = 0.0;
                *dwde = 0.0;

                *dudz = 0.0;
                *dvdz = 0.0;
                *dwdz = 0.0;

	} else {
		*dudc = ucat[k][j][i+1].x - ucat[k][j][i].x;
		*dvdc = ucat[k][j][i+1].y - ucat[k][j][i].y;
		*dwdc = ucat[k][j][i+1].z - ucat[k][j][i].z;

		double dude1, dude2, dvde1, dvde2, dwde1, dwde2;

		if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])< 1.1) {
			dude1 =  ucat[k][j][i].x - ucat[k][j-1][i].x;
			dvde1 =  ucat[k][j][i].y - ucat[k][j-1][i].y;
			dwde1 =  ucat[k][j][i].z - ucat[k][j-1][i].z;
		} else if ((nvert[k][j+1][i])< 1.1 && (nvert[k][j-1][i])> 1.1) 	{
                        dude1 =  ucat[k][j+1][i].x - ucat[k][j][i].x;
                        dvde1 =  ucat[k][j+1][i].y - ucat[k][j][i].y;
                        dwde1 =  ucat[k][j+1][i].z - ucat[k][j][i].z;
		} else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
                        dude1 =  0.0;
                        dvde1 =  0.0;
                        dwde1 =  0.0;
		}else {
                        dude1 =  0.5*(ucat[k][j+1][i].x - ucat[k][j-1][i].x);
                        dvde1 =  0.5*(ucat[k][j+1][i].y - ucat[k][j-1][i].y);
                        dwde1 =  0.5*(ucat[k][j+1][i].z - ucat[k][j-1][i].z);
		}

                if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])< 1.1) {
                        dude2 =  ucat[k][j][i+1].x - ucat[k][j-1][i+1].x;
                        dvde2 =  ucat[k][j][i+1].y - ucat[k][j-1][i+1].y;
                        dwde2 =  ucat[k][j][i+1].z - ucat[k][j-1][i+1].z;
                } else if ((nvert[k][j+1][i+1])< 1.1 && (nvert[k][j-1][i+1])> 1.1)  {
                        dude2 =  ucat[k][j+1][i+1].x - ucat[k][j][i+1].x;
                        dvde2 =  ucat[k][j+1][i+1].y - ucat[k][j][i+1].y;
                        dwde2 =  ucat[k][j+1][i+1].z - ucat[k][j][i+1].z;
                } else if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])> 1.1) {
                        dude2 =  0.0;
                        dvde2 =  0.0;
                        dwde2 =  0.0;
                }else {
                        dude2 =  0.5*(ucat[k][j+1][i+1].x - ucat[k][j-1][i+1].x);
                        dvde2 =  0.5*(ucat[k][j+1][i+1].y - ucat[k][j-1][i+1].y);
                        dwde2 =  0.5*(ucat[k][j+1][i+1].z - ucat[k][j-1][i+1].z);
                }

		
		if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])> 1.1) {
			*dude = dude1;
			*dvde = dvde1;
			*dwde = dwde1;
		} else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
                        *dude = dude2;
                        *dvde = dvde2;
                        *dwde = dwde2;
		} else {
                        *dude = 0.5*(dude1+dude2);
                        *dvde = 0.5*(dvde1+dvde2);
                        *dwde = 0.5*(dwde1+dwde2);
		}


		double dudz1, dudz2, dvdz1, dvdz2, dwdz1, dwdz2;

		if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])< 1.1) {
			dudz1 =  ucat[k][j][i].x - ucat[k-1][j][i].x;
			dvdz1 =  ucat[k][j][i].y - ucat[k-1][j][i].y;
			dwdz1 =  ucat[k][j][i].z - ucat[k-1][j][i].z;
		} else if ((nvert[k+1][j][i])< 1.1 && (nvert[k-1][j][i])> 1.1) 	{
                        dudz1 =  ucat[k+1][j][i].x - ucat[k][j][i].x;
                        dvdz1 =  ucat[k+1][j][i].y - ucat[k][j][i].y;
                        dwdz1 =  ucat[k+1][j][i].z - ucat[k][j][i].z;
		} else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
                        dudz1 =  0.0;
                        dvdz1 =  0.0;
                        dwdz1 =  0.0;
		}else {
                        dudz1 =  0.5*(ucat[k+1][j][i].x - ucat[k-1][j][i].x);
                        dvdz1 =  0.5*(ucat[k+1][j][i].y - ucat[k-1][j][i].y);
                        dwdz1 =  0.5*(ucat[k+1][j][i].z - ucat[k-1][j][i].z);
		}

                if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])< 1.1) {
                        dudz2 =  ucat[k][j][i+1].x - ucat[k-1][j][i+1].x;
                        dvdz2 =  ucat[k][j][i+1].y - ucat[k-1][j][i+1].y;
                        dwdz2 =  ucat[k][j][i+1].z - ucat[k-1][j][i+1].z;
                } else if ((nvert[k+1][j][i+1])< 1.1 && (nvert[k-1][j][i+1])> 1.1)  {
                        dudz2 =  ucat[k+1][j][i+1].x - ucat[k][j][i+1].x;
                        dvdz2 =  ucat[k+1][j][i+1].y - ucat[k][j][i+1].y;
                        dwdz2 =  ucat[k+1][j][i+1].z - ucat[k][j][i+1].z;
                } else if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])> 1.1) {
                        dudz2 =  0.0;
                        dvdz2 =  0.0;
                        dwdz2 =  0.0;
                }else {
                        dudz2 =  0.5*(ucat[k+1][j][i+1].x - ucat[k-1][j][i+1].x);
                        dvdz2 =  0.5*(ucat[k+1][j][i+1].y - ucat[k-1][j][i+1].y);
                        dwdz2 =  0.5*(ucat[k+1][j][i+1].z - ucat[k-1][j][i+1].z);
                }
		
		if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])> 1.1) {
			*dudz = dudz1;
			*dvdz = dvdz1;
			*dwdz = dwdz1;
		} else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
                        *dudz = dudz2;
                        *dvdz = dvdz2;
                        *dwdz = dwdz2;
		} else {
                        *dudz = 0.5*(dudz1+dudz2);
                        *dvdz = 0.5*(dvdz1+dvdz2);
                        *dwdz = 0.5*(dwdz1+dwdz2);
		}




	}

}


void Compute1_du_j (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
{
	
	if ((nvert[k][j][i])> 1.1 || (nvert[k][j+1][i])> 1.1) {
		*dudc = 0.0;
                *dvdc = 0.0;
                *dwdc = 0.0;

                *dude = 0.0;
                *dvde = 0.0;
                *dwde = 0.0;

                *dudz = 0.0;
                *dvdz = 0.0;
                *dwdz = 0.0;

	} else {

		double dudc1, dudc2, dvdc1, dvdc2, dwdc1, dwdc2;

		if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])< 1.1) {
			dudc1 =  ucat[k][j][i].x - ucat[k][j][i-1].x;
			dvdc1 =  ucat[k][j][i].y - ucat[k][j][i-1].y;
			dwdc1 =  ucat[k][j][i].z - ucat[k][j][i-1].z;
		} else if ((nvert[k][j][i+1])< 1.1 && (nvert[k][j][i-1])> 1.1) 	{
                        dudc1 =  ucat[k][j][i+1].x - ucat[k][j][i].x;
                        dvdc1 =  ucat[k][j][i+1].y - ucat[k][j][i].y;
                        dwdc1 =  ucat[k][j][i+1].z - ucat[k][j][i].z;
		} else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
                        dudc1 =  0.0;
                        dvdc1 =  0.0;
                        dwdc1 =  0.0;
		}else {
                        dudc1 =  0.5*(ucat[k][j][i+1].x - ucat[k][j][i-1].x);
                        dvdc1 =  0.5*(ucat[k][j][i+1].y - ucat[k][j][i-1].y);
                        dwdc1 =  0.5*(ucat[k][j][i+1].z - ucat[k][j][i-1].z);
		}

                if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])< 1.1) {
                        dudc2 =  ucat[k][j+1][i].x - ucat[k][j+1][i-1].x;
                        dvdc2 =  ucat[k][j+1][i].y - ucat[k][j+1][i-1].y;
                        dwdc2 =  ucat[k][j+1][i].z - ucat[k][j+1][i-1].z;
                } else if ((nvert[k][j+1][i+1])< 1.1 && (nvert[k][j+1][i-1])> 1.1)  {
                        dudc2 =  ucat[k][j+1][i+1].x - ucat[k][j+1][i].x;
                        dvdc2 =  ucat[k][j+1][i+1].y - ucat[k][j+1][i].y;
                        dwdc2 =  ucat[k][j+1][i+1].z - ucat[k][j+1][i].z;
                } else if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])> 1.1) {
                        dudc2 =  0.0;
                        dvdc2 =  0.0;
                        dwdc2 =  0.0;
                }else {
                        dudc2 =  0.5*(ucat[k][j+1][i+1].x - ucat[k][j+1][i-1].x);
                        dvdc2 =  0.5*(ucat[k][j+1][i+1].y - ucat[k][j+1][i-1].y);
                        dwdc2 =  0.5*(ucat[k][j+1][i+1].z - ucat[k][j+1][i-1].z);
                }

		if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])> 1.1) {
			*dudc = dudc1;
			*dvdc = dvdc1;
			*dwdc = dwdc1;
		} else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
                        *dudc = dudc2;
                        *dvdc = dvdc2;
                        *dwdc = dwdc2;
		} else {
                        *dudc = 0.5*(dudc1+dudc2);
                        *dvdc = 0.5*(dvdc1+dvdc2);
                        *dwdc = 0.5*(dwdc1+dwdc2);
		}



                *dude = ucat[k][j+1][i].x - ucat[k][j][i].x;
                *dvde = ucat[k][j+1][i].y - ucat[k][j][i].y;
                *dwde = ucat[k][j+1][i].z - ucat[k][j][i].z;


		double dudz1, dudz2, dvdz1, dvdz2, dwdz1, dwdz2;

		if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])< 1.1) {
			dudz1 =  ucat[k][j][i].x - ucat[k-1][j][i].x;
			dvdz1 =  ucat[k][j][i].y - ucat[k-1][j][i].y;
			dwdz1 =  ucat[k][j][i].z - ucat[k-1][j][i].z;
		} else if ((nvert[k+1][j][i])< 1.1 && (nvert[k-1][j][i])> 1.1) 	{
                        dudz1 =  ucat[k+1][j][i].x - ucat[k][j][i].x;
                        dvdz1 =  ucat[k+1][j][i].y - ucat[k][j][i].y;
                        dwdz1 =  ucat[k+1][j][i].z - ucat[k][j][i].z;
		} else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
                        dudz1 =  0.0;
                        dvdz1 =  0.0;
                        dwdz1 =  0.0;
		}else {
                        dudz1 =  0.5*(ucat[k+1][j][i].x - ucat[k-1][j][i].x);
                        dvdz1 =  0.5*(ucat[k+1][j][i].y - ucat[k-1][j][i].y);
                        dwdz1 =  0.5*(ucat[k+1][j][i].z - ucat[k-1][j][i].z);
		}

                if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])< 1.1) {
                        dudz2 =  ucat[k][j+1][i].x - ucat[k-1][j+1][i].x;
                        dvdz2 =  ucat[k][j+1][i].y - ucat[k-1][j+1][i].y;
                        dwdz2 =  ucat[k][j+1][i].z - ucat[k-1][j+1][i].z;
                } else if ((nvert[k+1][j+1][i])< 1.1 && (nvert[k-1][j+1][i])> 1.1)  {
                        dudz2 =  ucat[k+1][j+1][i].x - ucat[k][j+1][i].x;
                        dvdz2 =  ucat[k+1][j+1][i].y - ucat[k][j+1][i].y;
                        dwdz2 =  ucat[k+1][j+1][i].z - ucat[k][j+1][i].z;
                } else if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])> 1.1) {
                        dudz2 =  0.0;
                        dvdz2 =  0.0;
                        dwdz2 =  0.0;
                }else {
                        dudz2 =  0.5*(ucat[k+1][j+1][i].x - ucat[k-1][j+1][i].x);
                        dvdz2 =  0.5*(ucat[k+1][j+1][i].y - ucat[k-1][j+1][i].y);
                        dwdz2 =  0.5*(ucat[k+1][j+1][i].z - ucat[k-1][j+1][i].z);
                }
		
		if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])> 1.1) {
			*dudz = dudz1;
			*dvdz = dvdz1;
			*dwdz = dwdz1;
		} else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
                        *dudz = dudz2;
                        *dvdz = dvdz2;
                        *dwdz = dwdz2;
		} else {
                        *dudz = 0.5*(dudz1+dudz2);
                        *dvdz = 0.5*(dvdz1+dvdz2);
                        *dwdz = 0.5*(dwdz1+dwdz2);
		}




	}

}


void Compute1_du_k (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
{
	
	if ((nvert[k][j][i])> 1.1 || (nvert[k+1][j][i])> 1.1) {
		*dudc = 0.0;
                *dvdc = 0.0;
                *dwdc = 0.0;

                *dude = 0.0;
                *dvde = 0.0;
                *dwde = 0.0;

                *dudz = 0.0;
                *dvdz = 0.0;
                *dwdz = 0.0;

	} else {

		double dudc1, dudc2, dvdc1, dvdc2, dwdc1, dwdc2;

		if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])< 1.1) {
			dudc1 =  ucat[k][j][i].x - ucat[k][j][i-1].x;
			dvdc1 =  ucat[k][j][i].y - ucat[k][j][i-1].y;
			dwdc1 =  ucat[k][j][i].z - ucat[k][j][i-1].z;
		} else if ((nvert[k][j][i+1])< 1.1 && (nvert[k][j][i-1])> 1.1) 	{
                        dudc1 =  ucat[k][j][i+1].x - ucat[k][j][i].x;
                        dvdc1 =  ucat[k][j][i+1].y - ucat[k][j][i].y;
                        dwdc1 =  ucat[k][j][i+1].z - ucat[k][j][i].z;
		} else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
                        dudc1 =  0.0;
                        dvdc1 =  0.0;
                        dwdc1 =  0.0;
		}else {
                        dudc1 =  0.5*(ucat[k][j][i+1].x - ucat[k][j][i-1].x);
                        dvdc1 =  0.5*(ucat[k][j][i+1].y - ucat[k][j][i-1].y);
                        dwdc1 =  0.5*(ucat[k][j][i+1].z - ucat[k][j][i-1].z);
		}

                if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])< 1.1) {
                        dudc2 =  ucat[k+1][j][i].x - ucat[k+1][j][i-1].x;
                        dvdc2 =  ucat[k+1][j][i].y - ucat[k+1][j][i-1].y;
                        dwdc2 =  ucat[k+1][j][i].z - ucat[k+1][j][i-1].z;
                } else if ((nvert[k+1][j][i+1])< 1.1 && (nvert[k+1][j][i-1])> 1.1)  {
                        dudc2 =  ucat[k+1][j][i+1].x - ucat[k+1][j][i].x;
                        dvdc2 =  ucat[k+1][j][i+1].y - ucat[k+1][j][i].y;
                        dwdc2 =  ucat[k+1][j][i+1].z - ucat[k+1][j][i].z;
                } else if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])> 1.1) {
                        dudc2 =  0.0;
                        dvdc2 =  0.0;
                        dwdc2 =  0.0;
                }else {
                        dudc2 =  0.5*(ucat[k+1][j][i+1].x - ucat[k+1][j][i-1].x);
                        dvdc2 =  0.5*(ucat[k+1][j][i+1].y - ucat[k+1][j][i-1].y);
                        dwdc2 =  0.5*(ucat[k+1][j][i+1].z - ucat[k+1][j][i-1].z);
                }

		if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])> 1.1) {
			*dudc = dudc1;
			*dvdc = dvdc1;
			*dwdc = dwdc1;
		} else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
                        *dudc = dudc2;
                        *dvdc = dvdc2;
                        *dwdc = dwdc2;
		} else {
                        *dudc = 0.5*(dudc1+dudc2);
                        *dvdc = 0.5*(dvdc1+dvdc2);
                        *dwdc = 0.5*(dwdc1+dwdc2);
		}


		double dude1, dude2, dvde1, dvde2, dwde1, dwde2;

		if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])< 1.1) {
			dude1 =  ucat[k][j][i].x - ucat[k][j-1][i].x;
			dvde1 =  ucat[k][j][i].y - ucat[k][j-1][i].y;
			dwde1 =  ucat[k][j][i].z - ucat[k][j-1][i].z;
		} else if ((nvert[k][j+1][i])< 1.1 && (nvert[k][j-1][i])> 1.1) 	{
                        dude1 =  ucat[k][j+1][i].x - ucat[k][j][i].x;
                        dvde1 =  ucat[k][j+1][i].y - ucat[k][j][i].y;
                        dwde1 =  ucat[k][j+1][i].z - ucat[k][j][i].z;
		} else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
                        dude1 =  0.0;
                        dvde1 =  0.0;
                        dwde1 =  0.0;
		}else {
                        dude1 =  0.5*(ucat[k][j+1][i].x - ucat[k][j-1][i].x);
                        dvde1 =  0.5*(ucat[k][j+1][i].y - ucat[k][j-1][i].y);
                        dwde1 =  0.5*(ucat[k][j+1][i].z - ucat[k][j-1][i].z);
		}

                if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])< 1.1) {
                        dude2 =  ucat[k+1][j][i].x - ucat[k+1][j-1][i].x;
                        dvde2 =  ucat[k+1][j][i].y - ucat[k+1][j-1][i].y;
                        dwde2 =  ucat[k+1][j][i].z - ucat[k+1][j-1][i].z;
                } else if ((nvert[k+1][j+1][i])< 1.1 && (nvert[k+1][j-1][i])> 1.1)  {
                        dude2 =  ucat[k+1][j+1][i].x - ucat[k+1][j][i].x;
                        dvde2 =  ucat[k+1][j+1][i].y - ucat[k+1][j][i].y;
                        dwde2 =  ucat[k+1][j+1][i].z - ucat[k+1][j][i].z;
                } else if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])> 1.1) {
                        dude2 =  0.0;
                        dvde2 =  0.0;
                        dwde2 =  0.0;
                }else {
                        dude2 =  0.5*(ucat[k+1][j+1][i].x - ucat[k+1][j-1][i].x);
                        dvde2 =  0.5*(ucat[k+1][j+1][i].y - ucat[k+1][j-1][i].y);
                        dwde2 =  0.5*(ucat[k+1][j+1][i].z - ucat[k+1][j-1][i].z);
                }

		
		if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])> 1.1) {
			*dude = dude1;
			*dvde = dvde1;
			*dwde = dwde1;
		} else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
                        *dude = dude2;
                        *dvde = dvde2;
                        *dwde = dwde2;
		} else {
                        *dude = 0.5*(dude1+dude2);
                        *dvde = 0.5*(dvde1+dvde2);
                        *dwde = 0.5*(dwde1+dwde2);
		}


                *dudz = ucat[k+1][j][i].x - ucat[k][j][i].x;
                *dvdz = ucat[k+1][j][i].y - ucat[k][j][i].y;
                *dwdz = ucat[k+1][j][i].z - ucat[k][j][i].z;





	}

}



void Computewm_du_i (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz,
				double dudc_wm, double dvdc_wm, double dwdc_wm,
                                double dude_wm, double dvde_wm, double dwde_wm,
                                double dudz_wm, double dvdz_wm, double dwdz_wm)
{
	
	if ((nvert[k][j][i])> 1.1 || (nvert[k][j][i+1])> 1.1) {
		*dudc = 0.0;
                *dvdc = 0.0;
                *dwdc = 0.0;

                *dude = 0.0;
                *dvde = 0.0;
                *dwde = 0.0;

                *dudz = 0.0;
                *dvdz = 0.0;
                *dwdz = 0.0;

	} else {

		if ((nvert[k][j][i])> 0.1 || (nvert[k][j][i+1])> 0.1) {
                        *dudc = dudc_wm;
                        *dvdc = dvdc_wm;
                        *dwdc = dwdc_wm;		
		} else {	
			*dudc = ucat[k][j][i+1].x - ucat[k][j][i].x;
			*dvdc = ucat[k][j][i+1].y - ucat[k][j][i].y;
			*dwdc = ucat[k][j][i+1].z - ucat[k][j][i].z;
		}
		double dude1, dude2, dvde1, dvde2, dwde1, dwde2;

		if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])< 1.1) {

			if ((nvert[k][j][i])> 0.1 || (nvert[k][j-1][i])> 0.1) {
				dude1 = dude_wm;
				dvde1 = dvde_wm;
				dwde1 = dwde_wm;
			}else {
				dude1 =  ucat[k][j][i].x - ucat[k][j-1][i].x;
				dvde1 =  ucat[k][j][i].y - ucat[k][j-1][i].y;
				dwde1 =  ucat[k][j][i].z - ucat[k][j-1][i].z;
			}
		} else if ((nvert[k][j+1][i])< 1.1 && (nvert[k][j-1][i])> 1.1) 	{
                        if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j][i])> 0.1) {
                                dude1 = dude_wm;
                                dvde1 = dvde_wm;
                                dwde1 = dwde_wm;
                        }else {
                        	dude1 =  ucat[k][j+1][i].x - ucat[k][j][i].x;
	                        dvde1 =  ucat[k][j+1][i].y - ucat[k][j][i].y;
        	                dwde1 =  ucat[k][j+1][i].z - ucat[k][j][i].z;
			}
		} else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
                        dude1 =  0.0;
                        dvde1 =  0.0;
                        dwde1 =  0.0;
		}else {
                        if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j-1][i])> 0.1) {
                                dude1 = dude_wm;
                                dvde1 = dvde_wm;
                                dwde1 = dwde_wm;
                        }else {
	                        dude1 =  0.5*(ucat[k][j+1][i].x - ucat[k][j-1][i].x);
	                        dvde1 =  0.5*(ucat[k][j+1][i].y - ucat[k][j-1][i].y);
        	                dwde1 =  0.5*(ucat[k][j+1][i].z - ucat[k][j-1][i].z);
			}
		}

                if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])< 1.1) {
                        if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j-1][i+1])> 0.1) {
                                dude2 = dude_wm;
                                dvde2 = dvde_wm;
                                dwde2 = dwde_wm;
                        }else {
                        	dude2 =  ucat[k][j][i+1].x - ucat[k][j-1][i+1].x;
	                        dvde2 =  ucat[k][j][i+1].y - ucat[k][j-1][i+1].y;
        	                dwde2 =  ucat[k][j][i+1].z - ucat[k][j-1][i+1].z;
			}
                } else if ((nvert[k][j+1][i+1])< 1.1 && (nvert[k][j-1][i+1])> 1.1)  {
                        if ((nvert[k][j+1][i+1])> 0.1 || (nvert[k][j][i+1])> 0.1) {
                                dude2 = dude_wm;
                                dvde2 = dvde_wm;
                                dwde2 = dwde_wm;
                        }else {
	                        dude2 =  ucat[k][j+1][i+1].x - ucat[k][j][i+1].x;
        	                dvde2 =  ucat[k][j+1][i+1].y - ucat[k][j][i+1].y;
                	        dwde2 =  ucat[k][j+1][i+1].z - ucat[k][j][i+1].z;
			}
                } else if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])> 1.1) {
                        dude2 =  0.0;
                        dvde2 =  0.0;
                        dwde2 =  0.0;
                }else {
                        if ((nvert[k][j+1][i+1])> 0.1 || (nvert[k][j-1][i+1])> 0.1) {
                                dude2 = dude_wm;
                                dvde2 = dvde_wm;
                                dwde2 = dwde_wm;
                        }else {
                	        dude2 =  0.5*(ucat[k][j+1][i+1].x - ucat[k][j-1][i+1].x);
        	                dvde2 =  0.5*(ucat[k][j+1][i+1].y - ucat[k][j-1][i+1].y);
	                        dwde2 =  0.5*(ucat[k][j+1][i+1].z - ucat[k][j-1][i+1].z);
			}
                }

		
		if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])> 1.1) {
			*dude = dude1;
			*dvde = dvde1;
			*dwde = dwde1;
		} else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
                        *dude = dude2;
                        *dvde = dvde2;
                        *dwde = dwde2;
		} else {
                        *dude = 0.5*(dude1+dude2);
                        *dvde = 0.5*(dvde1+dvde2);
                        *dwde = 0.5*(dwde1+dwde2);
		}


		double dudz1, dudz2, dvdz1, dvdz2, dwdz1, dwdz2;

		if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])< 1.1) {
                        if ((nvert[k][j][i])> 0.1 || (nvert[k-1][j][i])> 0.1) {
                                dudz1 = dudz_wm;
                                dvdz1 = dvdz_wm;
                                dwdz1 = dwdz_wm;
                        }else {
				dudz1 =  ucat[k][j][i].x - ucat[k-1][j][i].x;
				dvdz1 =  ucat[k][j][i].y - ucat[k-1][j][i].y;
				dwdz1 =  ucat[k][j][i].z - ucat[k-1][j][i].z;
			}
		} else if ((nvert[k+1][j][i])< 1.1 && (nvert[k-1][j][i])> 1.1) 	{
                        if ((nvert[k+1][j][i])> 0.1 || (nvert[k][j][i])> 0.1) {
                                dudz1 = dudz_wm;
                                dvdz1 = dvdz_wm;
                                dwdz1 = dwdz_wm;
                        }else {
	                        dudz1 =  ucat[k+1][j][i].x - ucat[k][j][i].x;
        	                dvdz1 =  ucat[k+1][j][i].y - ucat[k][j][i].y;
                	        dwdz1 =  ucat[k+1][j][i].z - ucat[k][j][i].z;
			}
		} else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
                        dudz1 =  0.0;
                        dvdz1 =  0.0;
                        dwdz1 =  0.0;
		}else {
                        if ((nvert[k+1][j][i])> 0.1 || (nvert[k-1][j][i])> 0.1) {
                                dudz1 = dudz_wm;
                                dvdz1 = dvdz_wm;
                                dwdz1 = dwdz_wm;
                        }else {
                        	dudz1 =  0.5*(ucat[k+1][j][i].x - ucat[k-1][j][i].x);
	                        dvdz1 =  0.5*(ucat[k+1][j][i].y - ucat[k-1][j][i].y);
        	                dwdz1 =  0.5*(ucat[k+1][j][i].z - ucat[k-1][j][i].z);
			}
		}

                if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])< 1.1) {
                        if ((nvert[k][j][i+1])> 0.1 || (nvert[k-1][j][i+1])> 0.1) {
                                dudz2 = dudz_wm;
                                dvdz2 = dvdz_wm;
                                dwdz2 = dwdz_wm;
                        }else {
	                        dudz2 =  ucat[k][j][i+1].x - ucat[k-1][j][i+1].x;
        	                dvdz2 =  ucat[k][j][i+1].y - ucat[k-1][j][i+1].y;
                	        dwdz2 =  ucat[k][j][i+1].z - ucat[k-1][j][i+1].z;
			}
                } else if ((nvert[k+1][j][i+1])< 1.1 && (nvert[k-1][j][i+1])> 1.1)  {
                        if ((nvert[k+1][j][i+1])> 0.1 || (nvert[k][j][i+1])> 0.1) {
                                dudz2 = dudz_wm;
                                dvdz2 = dvdz_wm;
                                dwdz2 = dwdz_wm;
                        }else {
	                        dudz2 =  ucat[k+1][j][i+1].x - ucat[k][j][i+1].x;
        	                dvdz2 =  ucat[k+1][j][i+1].y - ucat[k][j][i+1].y;
                	        dwdz2 =  ucat[k+1][j][i+1].z - ucat[k][j][i+1].z;
			}
                } else if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])> 1.1) {
                        dudz2 =  0.0;
                        dvdz2 =  0.0;
                        dwdz2 =  0.0;
                }else {
                        if ((nvert[k+1][j][i+1])> 0.1 || (nvert[k-1][j][i+1])> 0.1) {
                                dudz2 = dudz_wm;
                                dvdz2 = dvdz_wm;
                                dwdz2 = dwdz_wm;
                        }else {
	                        dudz2 =  0.5*(ucat[k+1][j][i+1].x - ucat[k-1][j][i+1].x);
        	                dvdz2 =  0.5*(ucat[k+1][j][i+1].y - ucat[k-1][j][i+1].y);
                	        dwdz2 =  0.5*(ucat[k+1][j][i+1].z - ucat[k-1][j][i+1].z);
			}
                }
		
		if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])> 1.1) {
			*dudz = dudz1;
			*dvdz = dvdz1;
			*dwdz = dwdz1;
		} else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
                        *dudz = dudz2;
                        *dvdz = dvdz2;
                        *dwdz = dwdz2;
		} else {
                        *dudz = 0.5*(dudz1+dudz2);
                        *dvdz = 0.5*(dvdz1+dvdz2);
                        *dwdz = 0.5*(dwdz1+dwdz2);
		}




	}

}


void Computewm_du_j (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz,
				double dudc_wm, double dvdc_wm, double dwdc_wm,
                                double dude_wm, double dvde_wm, double dwde_wm,
                                double dudz_wm, double dvdz_wm, double dwdz_wm)


{
	
	if ((nvert[k][j][i])> 1.1 || (nvert[k][j+1][i])> 1.1) {
		*dudc = 0.0;
                *dvdc = 0.0;
                *dwdc = 0.0;

                *dude = 0.0;
                *dvde = 0.0;
                *dwde = 0.0;

                *dudz = 0.0;
                *dvdz = 0.0;
                *dwdz = 0.0;

	} else {

		double dudc1, dudc2, dvdc1, dvdc2, dwdc1, dwdc2;

		if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])< 1.1) {
                        if ((nvert[k][j][i])> 0.1 || (nvert[k][j][i-1])> 0.1) {
                                dudc1 = dudc_wm;
                                dvdc1 = dvdc_wm;
                                dwdc1 = dwdc_wm;
                        }else {
				dudc1 =  ucat[k][j][i].x - ucat[k][j][i-1].x;
				dvdc1 =  ucat[k][j][i].y - ucat[k][j][i-1].y;
				dwdc1 =  ucat[k][j][i].z - ucat[k][j][i-1].z;
			}
		} else if ((nvert[k][j][i+1])< 1.1 && (nvert[k][j][i-1])> 1.1) 	{
                        if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j][i])> 0.1) {
                                dudc1 = dudc_wm;
                                dvdc1 = dvdc_wm;
                                dwdc1 = dwdc_wm;
                        }else {
                        	dudc1 =  ucat[k][j][i+1].x - ucat[k][j][i].x;
	                        dvdc1 =  ucat[k][j][i+1].y - ucat[k][j][i].y;
        	                dwdc1 =  ucat[k][j][i+1].z - ucat[k][j][i].z;
			}
		} else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
                        dudc1 =  0.0;
                        dvdc1 =  0.0;
                        dwdc1 =  0.0;
		}else {
                        if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j][i-1])> 0.1) {
                                dudc1 = dudc_wm;
                                dvdc1 = dvdc_wm;
                                dwdc1 = dwdc_wm;
                        }else {
        	                dudc1 =  0.5*(ucat[k][j][i+1].x - ucat[k][j][i-1].x);
                	        dvdc1 =  0.5*(ucat[k][j][i+1].y - ucat[k][j][i-1].y);
                        	dwdc1 =  0.5*(ucat[k][j][i+1].z - ucat[k][j][i-1].z);
			}
		}

                if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])< 1.1) {
                        if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j+1][i-1])> 0.1) {
                                dudc2 = dudc_wm;
                                dvdc2 = dvdc_wm;
                                dwdc2 = dwdc_wm;
                        }else {
	                        dudc2 =  ucat[k][j+1][i].x - ucat[k][j+1][i-1].x;
        	                dvdc2 =  ucat[k][j+1][i].y - ucat[k][j+1][i-1].y;
                	        dwdc2 =  ucat[k][j+1][i].z - ucat[k][j+1][i-1].z;
			}
                } else if ((nvert[k][j+1][i+1])< 1.1 && (nvert[k][j+1][i-1])> 1.1)  {
                        if ((nvert[k][j+1][i+1])> 0.1 || (nvert[k][j+1][i])> 0.1) {
                                dudc2 = dudc_wm;
                                dvdc2 = dvdc_wm;
                                dwdc2 = dwdc_wm;
                        }else {
	                        dudc2 =  ucat[k][j+1][i+1].x - ucat[k][j+1][i].x;
        	                dvdc2 =  ucat[k][j+1][i+1].y - ucat[k][j+1][i].y;
                	        dwdc2 =  ucat[k][j+1][i+1].z - ucat[k][j+1][i].z;
			}
                } else if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])> 1.1) {
                        dudc2 =  0.0;
                        dvdc2 =  0.0;
                        dwdc2 =  0.0;
                }else {

                        if ((nvert[k][j+1][i+1])> 0.1 || (nvert[k][j+1][i-1])> 0.1) {
                                dudc2 = dudc_wm;
                                dvdc2 = dvdc_wm;
                                dwdc2 = dwdc_wm;
                        }else {
	                        dudc2 =  0.5*(ucat[k][j+1][i+1].x - ucat[k][j+1][i-1].x);
        	                dvdc2 =  0.5*(ucat[k][j+1][i+1].y - ucat[k][j+1][i-1].y);
                	        dwdc2 =  0.5*(ucat[k][j+1][i+1].z - ucat[k][j+1][i-1].z);
			}
                }

		if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])> 1.1) {
			*dudc = dudc1;
			*dvdc = dvdc1;
			*dwdc = dwdc1;
		} else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
                        *dudc = dudc2;
                        *dvdc = dvdc2;
                        *dwdc = dwdc2;
		} else {
                        *dudc = 0.5*(dudc1+dudc2);
                        *dvdc = 0.5*(dvdc1+dvdc2);
                        *dwdc = 0.5*(dwdc1+dwdc2);
		}

                if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j][i])> 0.1) {
                        *dude = dude_wm;
                        *dvde = dvde_wm;
                        *dwde = dwde_wm;
                } else {
	                *dude = ucat[k][j+1][i].x - ucat[k][j][i].x;
        	        *dvde = ucat[k][j+1][i].y - ucat[k][j][i].y;
                	*dwde = ucat[k][j+1][i].z - ucat[k][j][i].z;
		}

		double dudz1, dudz2, dvdz1, dvdz2, dwdz1, dwdz2;

		if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])< 1.1) {
                        if ((nvert[k][j][i])> 0.1 || (nvert[k-1][j][i])> 0.1) {
                                dudz1 = dudz_wm;
                                dvdz1 = dvdz_wm;
                                dwdz1 = dwdz_wm;
                        }else {
				dudz1 =  ucat[k][j][i].x - ucat[k-1][j][i].x;
				dvdz1 =  ucat[k][j][i].y - ucat[k-1][j][i].y;
				dwdz1 =  ucat[k][j][i].z - ucat[k-1][j][i].z;
			}
		} else if ((nvert[k+1][j][i])< 1.1 && (nvert[k-1][j][i])> 1.1) 	{
                        if ((nvert[k+1][j][i])> 0.1 || (nvert[k][j][i])> 0.1) {
                                dudz1 = dudz_wm;
                                dvdz1 = dvdz_wm;
                                dwdz1 = dwdz_wm;
                        }else {
	                        dudz1 =  ucat[k+1][j][i].x - ucat[k][j][i].x;
        	                dvdz1 =  ucat[k+1][j][i].y - ucat[k][j][i].y;
                	        dwdz1 =  ucat[k+1][j][i].z - ucat[k][j][i].z;
			}
		} else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
                        dudz1 =  0.0;
                        dvdz1 =  0.0;
                        dwdz1 =  0.0;
		}else {
                        if ((nvert[k+1][j][i])> 0.1 || (nvert[k][j][i])> 0.1) {
                                dudz1 = dudz_wm;
                                dvdz1 = dvdz_wm;
                                dwdz1 = dwdz_wm;
                        }else {
	                        dudz1 =  0.5*(ucat[k+1][j][i].x - ucat[k-1][j][i].x);
        	                dvdz1 =  0.5*(ucat[k+1][j][i].y - ucat[k-1][j][i].y);
                	        dwdz1 =  0.5*(ucat[k+1][j][i].z - ucat[k-1][j][i].z);
			}
		}

                if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])< 1.1) {
                        if ((nvert[k][j+1][i])> 0.1 || (nvert[k-1][j+1][i])> 0.1) {
                                dudz2 = dudz_wm;
                                dvdz2 = dvdz_wm;
                                dwdz2 = dwdz_wm;
                        }else {
	                        dudz2 =  ucat[k][j+1][i].x - ucat[k-1][j+1][i].x;
        	                dvdz2 =  ucat[k][j+1][i].y - ucat[k-1][j+1][i].y;
                	        dwdz2 =  ucat[k][j+1][i].z - ucat[k-1][j+1][i].z;
			}
                } else if ((nvert[k+1][j+1][i])< 1.1 && (nvert[k-1][j+1][i])> 1.1)  {
                        if ((nvert[k+1][j+1][i])> 0.1 || (nvert[k][j+1][i])> 0.1) {
                                dudz2 = dudz_wm;
                                dvdz2 = dvdz_wm;
                                dwdz2 = dwdz_wm;
                        }else {
                        	dudz2 =  ucat[k+1][j+1][i].x - ucat[k][j+1][i].x;
	                        dvdz2 =  ucat[k+1][j+1][i].y - ucat[k][j+1][i].y;
        	                dwdz2 =  ucat[k+1][j+1][i].z - ucat[k][j+1][i].z;
			}
                } else if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])> 1.1) {
                        dudz2 =  0.0;
                        dvdz2 =  0.0;
                        dwdz2 =  0.0;
                }else {
                        if ((nvert[k+1][j+1][i])> 0.1 || (nvert[k-1][j+1][i])> 0.1) {
                                dudz2 = dudz_wm;
                                dvdz2 = dvdz_wm;
                                dwdz2 = dwdz_wm;
                        }else {
	                        dudz2 =  0.5*(ucat[k+1][j+1][i].x - ucat[k-1][j+1][i].x);
        	                dvdz2 =  0.5*(ucat[k+1][j+1][i].y - ucat[k-1][j+1][i].y);
                	        dwdz2 =  0.5*(ucat[k+1][j+1][i].z - ucat[k-1][j+1][i].z);
			}
                }
		
		if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])> 1.1) {
			*dudz = dudz1;
			*dvdz = dvdz1;
			*dwdz = dwdz1;
		} else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
                        *dudz = dudz2;
                        *dvdz = dvdz2;
                        *dwdz = dwdz2;
		} else {
                        *dudz = 0.5*(dudz1+dudz2);
                        *dvdz = 0.5*(dvdz1+dvdz2);
                        *dwdz = 0.5*(dwdz1+dwdz2);
		}




	}

}


void Computewm_du_k (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz,
 				double dudc_wm, double dvdc_wm, double dwdc_wm,
                                double dude_wm, double dvde_wm, double dwde_wm,
                                double dudz_wm, double dvdz_wm, double dwdz_wm)
{
	
	if ((nvert[k][j][i])> 1.1 || (nvert[k+1][j][i])> 1.1) {
		*dudc = 0.0;
                *dvdc = 0.0;
                *dwdc = 0.0;

                *dude = 0.0;
                *dvde = 0.0;
                *dwde = 0.0;

                *dudz = 0.0;
                *dvdz = 0.0;
                *dwdz = 0.0;

	} else {

		double dudc1, dudc2, dvdc1, dvdc2, dwdc1, dwdc2;

		if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])< 1.1) {
                        if ((nvert[k][j][i])> 0.1 || (nvert[k][j][i-1])> 0.1) {
                                dudc1 = dudc_wm;
                                dvdc1 = dvdc_wm;
                                dwdc1 = dwdc_wm;
                        }else {
				dudc1 =  ucat[k][j][i].x - ucat[k][j][i-1].x;
				dvdc1 =  ucat[k][j][i].y - ucat[k][j][i-1].y;
				dwdc1 =  ucat[k][j][i].z - ucat[k][j][i-1].z;
			}
		} else if ((nvert[k][j][i+1])< 1.1 && (nvert[k][j][i-1])> 1.1) 	{
                        if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j][i])> 0.1) {
                                dudc1 = dudc_wm;
                                dvdc1 = dvdc_wm;
                                dwdc1 = dwdc_wm;
                        }else {
	                        dudc1 =  ucat[k][j][i+1].x - ucat[k][j][i].x;
        	                dvdc1 =  ucat[k][j][i+1].y - ucat[k][j][i].y;
                	        dwdc1 =  ucat[k][j][i+1].z - ucat[k][j][i].z;
			}
		} else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
                        dudc1 =  0.0;
                        dvdc1 =  0.0;
                        dwdc1 =  0.0;
		}else {
                        if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j][i-1])> 0.1) {
                                dudc1 = dudc_wm;
                                dvdc1 = dvdc_wm;
                                dwdc1 = dwdc_wm;
                        }else {
	                        dudc1 =  0.5*(ucat[k][j][i+1].x - ucat[k][j][i-1].x);
        	                dvdc1 =  0.5*(ucat[k][j][i+1].y - ucat[k][j][i-1].y);
                	        dwdc1 =  0.5*(ucat[k][j][i+1].z - ucat[k][j][i-1].z);
			}
		}

                if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])< 1.1) {
                        if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j][i-1])> 0.1) {
                                dudc2 = dudc_wm;
                                dvdc2 = dvdc_wm;
                                dwdc2 = dwdc_wm;
                        }else {
	                        dudc2 =  ucat[k+1][j][i].x - ucat[k+1][j][i-1].x;
        	                dvdc2 =  ucat[k+1][j][i].y - ucat[k+1][j][i-1].y;
                	        dwdc2 =  ucat[k+1][j][i].z - ucat[k+1][j][i-1].z;
			}
                } else if ((nvert[k+1][j][i+1])< 1.1 && (nvert[k+1][j][i-1])> 1.1)  {
                        if ((nvert[k+1][j][i+1])> 0.1 || (nvert[k+1][j][i])> 0.1) {
                                dudc2 = dudc_wm;
                                dvdc2 = dvdc_wm;
                                dwdc2 = dwdc_wm;
                        }else {
	                        dudc2 =  ucat[k+1][j][i+1].x - ucat[k+1][j][i].x;
        	                dvdc2 =  ucat[k+1][j][i+1].y - ucat[k+1][j][i].y;
                	        dwdc2 =  ucat[k+1][j][i+1].z - ucat[k+1][j][i].z;
			}
                } else if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])> 1.1) {
                        dudc2 =  0.0;
                        dvdc2 =  0.0;
                        dwdc2 =  0.0;
                }else {
                        if ((nvert[k+1][j][i+1])> 0.1 || (nvert[k+1][j][i-1])> 0.1) {
                                dudc2 = dudc_wm;
                                dvdc2 = dvdc_wm;
                                dwdc2 = dwdc_wm;
                        }else {
	                        dudc2 =  0.5*(ucat[k+1][j][i+1].x - ucat[k+1][j][i-1].x);
        	                dvdc2 =  0.5*(ucat[k+1][j][i+1].y - ucat[k+1][j][i-1].y);
                	        dwdc2 =  0.5*(ucat[k+1][j][i+1].z - ucat[k+1][j][i-1].z);
			}
                }

		if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])> 1.1) {
			*dudc = dudc1;
			*dvdc = dvdc1;
			*dwdc = dwdc1;
		} else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
                        *dudc = dudc2;
                        *dvdc = dvdc2;
                        *dwdc = dwdc2;
		} else {
                        *dudc = 0.5*(dudc1+dudc2);
                        *dvdc = 0.5*(dvdc1+dvdc2);
                        *dwdc = 0.5*(dwdc1+dwdc2);
		}


		double dude1, dude2, dvde1, dvde2, dwde1, dwde2;

		if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])< 1.1) {
                        if ((nvert[k][j][i])> 0.1 || (nvert[k][j-1][i])> 0.1) {
                                dude1 = dude_wm;
                                dvde1 = dvde_wm;
                                dwde1 = dwde_wm;
                        }else {
				dude1 =  ucat[k][j][i].x - ucat[k][j-1][i].x;
				dvde1 =  ucat[k][j][i].y - ucat[k][j-1][i].y;
				dwde1 =  ucat[k][j][i].z - ucat[k][j-1][i].z;
			}
		} else if ((nvert[k][j+1][i])< 1.1 && (nvert[k][j-1][i])> 1.1) 	{
                        if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j][i])> 0.1) {
                                dude1 = dude_wm;
                                dvde1 = dvde_wm;
                                dwde1 = dwde_wm;
                        }else {
	                        dude1 =  ucat[k][j+1][i].x - ucat[k][j][i].x;
        	                dvde1 =  ucat[k][j+1][i].y - ucat[k][j][i].y;
                	        dwde1 =  ucat[k][j+1][i].z - ucat[k][j][i].z;
			}
		} else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
                        dude1 =  0.0;
                        dvde1 =  0.0;
                        dwde1 =  0.0;
		}else {
                        if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j-1][i])> 0.1) {
                                dude1 = dude_wm;
                                dvde1 = dvde_wm;
                                dwde1 = dwde_wm;
                        }else {
	                        dude1 =  0.5*(ucat[k][j+1][i].x - ucat[k][j-1][i].x);
        	                dvde1 =  0.5*(ucat[k][j+1][i].y - ucat[k][j-1][i].y);
                	        dwde1 =  0.5*(ucat[k][j+1][i].z - ucat[k][j-1][i].z);
			}
		}

                if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])< 1.1) {
                        if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j-1][i])> 0.1) {
                                dude2 = dude_wm;
                                dvde2 = dvde_wm;
                                dwde2 = dwde_wm;
                        }else {
	                        dude2 =  ucat[k+1][j][i].x - ucat[k+1][j-1][i].x;
        	                dvde2 =  ucat[k+1][j][i].y - ucat[k+1][j-1][i].y;
                	        dwde2 =  ucat[k+1][j][i].z - ucat[k+1][j-1][i].z;
			}
                } else if ((nvert[k+1][j+1][i])< 1.1 && (nvert[k+1][j-1][i])> 1.1)  {
                        if ((nvert[k+1][j+1][i])> 0.1 || (nvert[k+1][j][i])> 0.1) {
                                dude2 = dude_wm;
                                dvde2 = dvde_wm;
                                dwde2 = dwde_wm;
                        }else {
	                        dude2 =  ucat[k+1][j+1][i].x - ucat[k+1][j][i].x;
        	                dvde2 =  ucat[k+1][j+1][i].y - ucat[k+1][j][i].y;
                	        dwde2 =  ucat[k+1][j+1][i].z - ucat[k+1][j][i].z;
			}
                } else if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])> 1.1) {
                        dude2 =  0.0;
                        dvde2 =  0.0;
                        dwde2 =  0.0;
                }else {
                        if ((nvert[k+1][j+1][i])> 0.1 || (nvert[k+1][j-1][i])> 0.1) {
                                dude2 = dude_wm;
                                dvde2 = dvde_wm;
                                dwde2 = dwde_wm;
                        }else {
	                        dude2 =  0.5*(ucat[k+1][j+1][i].x - ucat[k+1][j-1][i].x);
        	                dvde2 =  0.5*(ucat[k+1][j+1][i].y - ucat[k+1][j-1][i].y);
                	        dwde2 =  0.5*(ucat[k+1][j+1][i].z - ucat[k+1][j-1][i].z);
			}
                }

		
		if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])> 1.1) {
			*dude = dude1;
			*dvde = dvde1;
			*dwde = dwde1;
		} else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
                        *dude = dude2;
                        *dvde = dvde2;
                        *dwde = dwde2;
		} else {
                        *dude = 0.5*(dude1+dude2);
                        *dvde = 0.5*(dvde1+dvde2);
                        *dwde = 0.5*(dwde1+dwde2);
		}

                if ((nvert[k+1][j][i])> 0.1 || (nvert[k][j][i])> 0.1) {
                        *dudz = dudz_wm;
                        *dvdz = dvdz_wm;
                        *dwdz = dwdz_wm;
                }else {
	                *dudz = ucat[k+1][j][i].x - ucat[k][j][i].x;
        	        *dvdz = ucat[k+1][j][i].y - ucat[k][j][i].y;
                	*dwdz = ucat[k+1][j][i].z - ucat[k][j][i].z;
		}




	}

}


