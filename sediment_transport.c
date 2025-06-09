#include "petsc.h"
#include "variables.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <valarray>
//#include <algorithm>

extern PetscInt  NumberOfBodies, rstart, input_ib_depth, periodic_morpho;
extern PetscInt tistart, ti, bed_roughness, STRONG_COUPLING;
extern PetscInt projection_method;
extern PetscReal Nseg;
extern double inlet_sediment_flux;
#define  PI 3.14159265

double E_coeff (double utau, double ks, double nu)
{
  double kplus=utau*ks/nu, dB;
  double kappa=0.41, B=5.5;
  if(kplus<=2.25) dB = 0.0;
  else if(kplus>2.25 && kplus<90.) dB = (B-8.5+1./kappa*log(kplus))*sin(0.4258*(log(kplus)-0.811));
  else if(kplus>=90.) dB = B-8.5+1./kappa*log(kplus);
  return exp(kappa*(B-dB));
}

double u_hydset_roughness(double nu, double y, double utau, double ks)
{
 double y0plus=11.53,f;
 double kappa=0.41, B=5.5;
 double yplus = utau*y/nu;
 
  if(yplus<=y0plus){f= utau * yplus;}
  else {f= utau/kappa*log(E_coeff(utau,ks,nu)*yplus);}
  return f;
}

double f_hydset(double nu, double u, double y, double utau0, double ks) 
{
double y0plus=11.53, f;
double kappa=0.41, B=5.5;
double yplus=utau0*y/nu;
if (yplus<=y0plus) {f= utau0*yplus-u;}
else {f= utau0*(1./kappa*log(E_coeff(utau0,ks,nu)*yplus))-u;}
return f;
}

double df_hydset (double nu, double u, double y, double utau0, double ks)
{
double eps=1.e-7;
return (f_hydset(nu, u, y, utau0 + eps, ks) - f_hydset(nu, u, y, utau0 - eps, ks))/(2.*eps);
}


double find_utau_hydset(double nu,double u, double y, double utau_guess, double ks)
{
 double utau,utau0 = utau_guess;  
 int ii;
 for(ii=0;ii<30;ii++){
 utau=utau0-f_hydset(nu,u,y,utau0,ks)/df_hydset(nu,u,y,utau0,ks);
 if (fabs(utau-utau0)<1.e-7)break;
 utau0=utau;
  }
return utau;
};


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


void wall_function_roughness_a (UserCtx *user, double ks, double sc, double sb, Cmpnts Ua, Cmpnts Uc, Cmpnts *Ub, PetscReal *ustar, double nx, double ny, double nz)
{
	double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
	double un = u_c * nx + v_c * ny + w_c * nz;
	double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
	double ut_mag = sqrt( ut*ut + vt*vt + wt*wt );
	//*ustar = find_utau_Cabot_roughness(1./user->ren, ut_mag, sc, 0.01, 0, ks);
        *ustar = find_utau_hydset(1./user->ren, ut_mag, sc, 0.01, ks);
	//double ut_mag_modeled = u_Cabot_roughness(1./user->ren, sb, *ustar, 0, ks);
	double ut_mag_modeled = u_hydset_roughness(1./user->ren, sb, *ustar, ks);

	// xiaolei add
	/*	
	double nu = 1./user->ren;
	double A=8.3;
	double B=0.1; //1.0/7.0; //1.085/log(user->ren)+6.535/pow(log(user->ren),2);
        *ustar = pow( ut_mag * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));
	double ybp = sb*(*ustar)/nu;
	double ycp = sc*(*ustar)/nu;
	double ut_mag_modeled;
	if (ycp>12.0) {
        	*ustar = pow( ut_mag * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));
		ut_mag_modeled = A*(*ustar)*pow(ybp,B);
	} 
	else {
		*ustar = sqrt(fabs(nu*ut_mag/sc));
		ut_mag_modeled = ut_mag*sb/sc;	
	}
	*/	

        //*ustar = pow( ut_mag * pow(nu, 1.0/7.0) / (8.3 * pow(sc, 1.0/7.0)),  7.0/8.0);
	//double ut_mag_modeled = 8.3*(*ustar)*pow(sb*(*ustar)/nu,1.0/7.0);
	// end add


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


// xiaolei   
PetscErrorCode flow_variables_ref_level (UserCtx *user, IBMNodes *ibm, int ti, PetscInt tistart)
{
	DM		 da  = user->da, fda  = user->fda;
	DMDALocalInfo      info  = user->info;

	int	 xs  = info.xs, xe  = info.xs + info.xm;
	int	 ys  = info.ys, ye  = info.ys + info.ym;
	int	 zs  = info.zs, ze  = info.zs + info.zm;
	int	 mx  = info.mx, my  = info.my, mz  = info.mz;
	int	 lxs, lys, lzs, lxe, lye, lze;
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	if( xs == 0 )lxs = xs + 1;
	if( ys == 0 )lys = ys + 1;
	if( zs == 0 )lzs = zs + 1;

	if( xe == mx )lxe = xe - 1;
	if( ye == my )lye = ye - 1;
	if( ze == mz )lze = ze - 1;

        //IBMNodes *ibm = &ibm_ptr[ibi];
	Cmpnts	   *** ucat, *** lucat;
	PetscReal  *** ustar;
	PetscReal  *** conc, ***lnu_t;
	int       n_elmt  = ibm->n_elmt, 
		       n_v  = ibm->n_v;
	int	   i, j, k;
        PetscReal	   ref_level;
	//PetscInt	   ip1,	ip2, ip3, jp1, jp2, jp3, kp1, kp2, kp3;
	//PetscReal	   sk1,	sk2, sk3, cv1, cv2, cv3;
	// PetscReal  *** nvert, *** nvert_o;
//	PetscReal	cs1, cs2, cs3;
//	PetscInt	nii;
	IBMInfo     *   ibminfo;
//	PetscInt	NumberOfBodies = 0, ibi;
	//	PetscReal	nfx, 
	//		nfy, 
	//		nfz;
         int   ibi;              
         PetscReal   sbbb = 0.025; // all stream structures  0.025;
	 double cc; 

	if(ti==tistart || ti==0) PetscPrintf(PETSC_COMM_WORLD, "sbb: %e \n",sbbb);

	if (ti==tistart) {
		for (i=0;i<ibm->n_elmt;i++) {
			ibm[0].Bvel[i].x=0.0;
			ibm[0].Bvel[i].y=0.0;
			ibm[0].Bvel[i].z=0.0;
			ibm[0].Shvel[i]=0.0;
			if(LiveBed) ibm[0].C[i]=0.0;
		}
	}




		DMDAVecGetArray(da, user->lUstar, &ustar);
		DMDAVecGetArray(fda, user->Ucat,  &ucat);
		DMDAVecGetArray(fda, user->lUcat,  &lucat);
		if(LiveBed) DMDAVecGetArray(da, user->lConc, &conc);
		// if(LiveBed) DMDAVecGetArray(da, user->lNu_t, &lnu_t);

		for(ibi=0; ibi<NumberOfBodies; ibi++)
		{

			double count[ibm[ibi].n_elmt],_u[ibm[ibi].n_elmt],_v[ibm[ibi].n_elmt],_w[ibm[ibi].n_elmt],_ustar[ibm[ibi].n_elmt],_C[ibm[ibi].n_elmt];
			double count_sum[ibm[ibi].n_elmt],_u_sum[ibm[ibi].n_elmt],_v_sum[ibm[ibi].n_elmt],_w_sum[ibm[ibi].n_elmt],_ustar_sum[ibm[ibi].n_elmt],_C_sum[ibm[ibi].n_elmt];
			int il;
			for(il=0;il<ibm[ibi].n_elmt;il++) {
				count[il]=0.0;
         			_u[il] = 0.0;
         			_v[il] = 0.0;
         			_w[il] = 0.0;
         			_ustar[il] = 0.0;
         			_C[il] = 0.0;
			}

			IBMListNode *current;
			current = user->ibmlist[ibi].head;
			while (current) {
				IBMInfo *ibminfo = &current->ibm_intp;
				current = current->next;
				
				int ni = ibminfo->cell;
				int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
				int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
				int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
				i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;
				
				double sb = ibminfo->d_s, sc = sb + ibminfo->d_i;
				double sk1  = ibminfo->cr1, sk2 = ibminfo->cr2, sk3 = ibminfo->cr3;
				double cs1 = ibminfo->cs1, cs2 = ibminfo->cs2, cs3 = ibminfo->cs3;
                                
	//			  if (ti==tistart){ref_level = sb;} else {ref_level = deltab;}

				count[ni]+=1.0;
				Cmpnts Ua, Uc;

			        int rigid = ibm->Rigidity[ni];
                                	
        			if (ni>=0) {
					Ua.x = ibm->u[ibm->nv1[ni]].x * cs1 + ibm->u[ibm->nv2[ni]].x * cs2 + ibm->u[ibm->nv3[ni]].x * cs3;
					Ua.y = ibm->u[ibm->nv1[ni]].y * cs1 + ibm->u[ibm->nv2[ni]].y * cs2 + ibm->u[ibm->nv3[ni]].y * cs3;
					Ua.z = ibm->u[ibm->nv1[ni]].z * cs1 + ibm->u[ibm->nv2[ni]].z * cs2 + ibm->u[ibm->nv3[ni]].z * cs3;
			                   }
				else       {
					Ua.x = Ua.y = Ua.z = 0.;
	                                   }
	 			
        			Uc.x = (lucat[kp1][jp1][ip1].x * sk1 + lucat[kp2][jp2][ip2].x * sk2 +lucat[kp3][jp3][ip3].x * sk3);
				Uc.y = (lucat[kp1][jp1][ip1].y * sk1 + lucat[kp2][jp2][ip2].y * sk2 + lucat[kp3][jp3][ip3].y * sk3);
				Uc.z = (lucat[kp1][jp1][ip1].z * sk1 + lucat[kp2][jp2][ip2].z * sk2 + lucat[kp3][jp3][ip3].z * sk3);
				if(LiveBed) {
                                             double c1 = (conc[kp1][jp1][ip1] * sk1 + conc[kp2][jp2][ip2] * sk2 + conc[kp3][jp3][ip3] * sk3);
                                             double c  = PetscMax (c1 , conc[k][j][i]);
                                             double nu = 1./user->ren;
                                             double sigma_phi = 0.75;
                                             double nu_t = lnu_t[kp1][jp1][ip1] * sk1 + lnu_t[kp2][jp2][ip2] * sk2 +lnu_t[kp3][jp3][ip3] * sk3;
                                                    nu_t = PetscMax (nu_t, 0.0);
                                             double nu_tt = nu + sigma_phi * nu_t;
                                             double alfaa = fabs (sc - sb);
                                                    alfaa = alfaa * w_s / nu_tt;
                                                    alfaa = exp (alfaa);
                                                    alfaa = PetscMin (1.0, alfaa);
                                                    alfaa = 1.0 - alfaa;
                                             if(!rigid) {cc = c + ibm->SCont[ni] * alfaa;} else {cc = c;}
                                            }
				
				double nu=1.0/user->ren;
				Cmpnts Ub;


				double nx, ny, nz;
				nx= ibm->nf_x[ni];
				ny= ibm->nf_y[ni];
				nz= ibm->nf_z[ni];
	
				double ustar1;
				double rs_sedi;
				rs_sedi=rs_bed*roughness_size;
				wall_function_s (nu, rs_sedi, sc, sb_bed, Ua, Uc,&Ub, &ustar1, nx,ny,nz);

				//wall_function_roughness_a (user, rs_sedi, sc, sb_bed, Ua, Uc, &Ub, &ustar1, nx,ny,nz);

				//wall_function_s (nu, roughness_size, sc, sb_bed, Ua, Uc, &Ub, &ustar[k][j][i], ibm->nf_x[ni], ibm->nf_y[ni], ibm->nf_z[ni]);
    				//if(!bed_roughness)wall_function_roughness_a (user, 0.001, sc, sbbb, Ua, Uc, &Ub, &ustar[k][j][i], ibm->nf_x[ni], ibm->nf_y[ni], ibm->nf_z[ni]);
				//if(bed_roughness)wall_function_roughness_a (user, k_ss, sc, sbbb, Ua, Uc, &Ub, &ustar[k][j][i], ibm->nf_x[ni], ibm->nf_y[ni], ibm->nf_z[ni]);
     
 				if( i >= lxs && i < lxe && j >= lys && j < lye && k >= lzs && k < lze ) {
         				_u[ni] += Ub.x;
         				_v[ni] += Ub.y;
         				_w[ni] += Ub.z;
         				_ustar[ni] += ustar1;
         				if(LiveBed) _C[ni] += cc;
   				}
			}


  			MPI_Allreduce( &(_u[0]), &(_u_sum[0]), ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		  	MPI_Allreduce( &(_v[0]), &(_v_sum[0]), ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		  	MPI_Allreduce( &(_w[0]), &(_w_sum[0]), ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		  	MPI_Allreduce( &(_ustar[0]), &(_ustar_sum[0]), ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		  	if(LiveBed) MPI_Allreduce( &(_C[0]), &(_C_sum[0]), ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		  	MPI_Allreduce( &(count[0]), &(count_sum[0]), ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

			for(il=0;il<ibm[ibi].n_elmt;il++) {
				//ibm[ibi].Bvel[il].x+=_u_sum[il]/(count_sum[il]+1.e-19);
				//ibm[ibi].Bvel[il].y+=_v_sum[il]/(count_sum[il]+1.e-19);
				//ibm[ibi].Bvel[il].z+=_w_sum[il]/(count_sum[il]+1.e-19);
				//ibm[ibi].Shvel[il]+=_ustar_sum[il]/(count_sum[il]+1.e-19);
				//if(LiveBed) ibm[ibi].C[il]+=_C_sum[il]/(count_sum[il]+1.e-19);
				_u[il]=_u_sum[il]/(count_sum[il]+1.e-19);
				_v[il]=_v_sum[il]/(count_sum[il]+1.e-19);
				_w[il]=_w_sum[il]/(count_sum[il]+1.e-19);
				_ustar[il]=_ustar_sum[il]/(count_sum[il]+1.e-19);
				if(LiveBed) _C[il]=_C_sum[il]/(count_sum[il]+1.e-19);

			}



			int elmt;
			int itemp;
			for (itemp=0;itemp<10;itemp++) {
			for( elmt = 0; elmt < ibm[ibi].n_elmt; elmt++ )
			{

			double nfz = ibm[ibi].nf_z[ elmt ];

			if (count_sum[elmt]<0.9 && nfz > 1.e-7 && ibm[ibi].elmt_depth[elmt] < 0.3) {

				double riter=0.0;
				double bvelu_sum=0.0;
				double bvelv_sum=0.0;
				double bvelw_sum=0.0;
				double shvel_sum=0.0;
				double c_sum=0.0;

				int ii,jj,nii;
				for(ii=0;ii<100;ii++)
				{
					nii=ibm[ibi].c2ac[elmt].c[ii];	// xiaolei add
					if (nii>=0) {    // xiaolei add 
					if(ibm->nf_z[nii]<1.e-6 || ibm->elmt_depth[nii]>0.3 || count_sum[nii] <0.9)
                                        { 
                                        }
                                        else
                                        {
						count_sum[elmt]=1.0;
						riter = riter + 1.;
						shvel_sum+=_ustar[nii];
						if(LiveBed) c_sum+=_C[nii];
						bvelu_sum+=_u[nii];
						bvelv_sum+=_v[nii];
						bvelw_sum+=_w[nii];
			              	}
				      	}	
			      	}


				_ustar[ elmt ] = shvel_sum / (riter+1.e-17);
				if(LiveBed) _C[ elmt ] = c_sum / (riter+1.e-17);
				_u[ elmt] = bvelu_sum / (riter+1.e-17);
				_v[ elmt] = bvelv_sum / (riter+1.e-17);
				_w[ elmt] = bvelw_sum / (riter+1.e-17);		

			}
			}
			}

			for( elmt = 0; elmt < ibm[ibi].n_elmt; elmt++ )
			{
			double nfx = ibm->nf_x[ elmt ];
			double nfy = ibm->nf_y[ elmt ];
			double nfz = ibm->nf_z[ elmt ];

			double ucx = _u[ elmt ];
			double ucy = _v[ elmt ];
			double ucz = _w[ elmt ];


			if( nfz < 1.e-7 || ibm[ibi].elmt_depth[elmt] > 0.3)
			{
				_u[ elmt ] = 0.;
				_v[ elmt ] = 0.;
				_w[ elmt ] = 0.;
                	        _ustar[ elmt ] = 0.;
				if(LiveBed) _C[elmt]=0.;
	       		}
			else  
			{
				_u[ elmt ] = ucx - (ucx * nfx + ucy * nfy + ucz * nfz) * nfx;
	        		_v[ elmt ] = ucy - (ucx * nfx + ucy * nfy + ucz * nfz) * nfy;
				_w[ elmt ] = ucz - (ucx * nfx + ucy * nfy + ucz * nfz) * nfz;
			}
			}

			double fac;
			if (ti<=tistart+1) double fac=1.0/((double)bed_avg+1.0);
			else fac=1.0/((double)bed_avg);
			for( il = 0; il < ibm[ibi].n_elmt; il++ ) {
				
				ibm[ibi].Bvel[il].x+=_u[il]*fac;
				ibm[ibi].Bvel[il].y+=_v[il]*fac;
				ibm[ibi].Bvel[il].z+=_w[il]*fac;
				ibm[ibi].Shvel[il]+=_ustar[il]*fac;
				if(LiveBed) ibm[ibi].C[il]+=_C[il]*fac;

			}

		}

	
		if(LiveBed) DMDAVecRestoreArray(da, user->lConc, &conc);
		DMDAVecRestoreArray(da, user->lUstar, &ustar);
		DMDAVecRestoreArray(fda, user->Ucat,  &ucat);
		DMDAVecRestoreArray(fda, user->lUcat,  &lucat);
		DMGlobalToLocalBegin(fda, user->Ucat, INSERT_VALUES, user->lUcat);
		DMGlobalToLocalEnd(fda, user->Ucat, INSERT_VALUES, user->lUcat);
	
 return (0);
}

// xiaolei change SEDI
PetscErrorCode Finalizing_Projecting( UserCtx * user, IBMNodes * ibm ,int ti, PetscInt tistart)
{
	DM		da  = user->da, fda  = user->fda;
	DMDALocalInfo     info  = user->info;
        IBMInfo         *ibminfo;
	int        ibi; 
       
	Cmpnts	        *Bvel;
	PetscReal       *Shvel;
	PetscReal       *C;
	int        n_elmt = ibm->n_elmt, 
		        n_v = ibm->n_v;
	int        i, j, k;
	int        nii;
	PetscReal       ucx, ucy, ucz, riter;
        
        int	elmt,nelmt;

        PetscReal	nfx, 
			nfy, 
			nfz;

	Cmpnts *Bvel_tmp;
	PetscReal *C_tmp;
	PetscReal *Shvel_tmp;
	PetscMalloc(n_elmt*sizeof(Cmpnts), &(Bvel_tmp));
      	PetscMalloc(n_elmt*sizeof(PetscReal), &(C_tmp));
      	PetscMalloc(n_elmt*sizeof(PetscReal), &(Shvel_tmp));


	for( nelmt = 0; nelmt < ibm->n_elmt; nelmt++ ) {

		Shvel_tmp[nelmt]=ibm->Shvel[ nelmt ];
		if(LiveBed) C_tmp[nelmt]=ibm->C[ nelmt ];
		Bvel_tmp[nelmt].x=ibm->Bvel[ nelmt].x;
		Bvel_tmp[nelmt].y=ibm->Bvel[ nelmt].y;
		Bvel_tmp[nelmt].z=ibm->Bvel[ nelmt].z;
	
	}

	for( nelmt = 0; nelmt < ibm->n_elmt; nelmt++ )
	{
		if(ibm->nf_z[nelmt] < 1.e-7 || ibm->elmt_depth[nelmt] > 0.3)
		{
			ibm->Shvel[ nelmt ] = 0.;
			ibm->Bvel[ nelmt ].x = 0.;
			ibm->Bvel[ nelmt ].y = 0.;
			ibm->Bvel[ nelmt ].z = 0.;
		}
		else
		{
		 	double test;
		 	if(LiveBed){ test = fabs(ibm->Shvel[nelmt])+fabs(ibm->Bvel[nelmt].x)+fabs(ibm->Bvel[nelmt].y)+fabs(ibm->Bvel[nelmt].z) + fabs(ibm->C[nelmt]);}
                 	else { test = fabs(ibm->Shvel[nelmt])+fabs(ibm->Bvel[nelmt].x)+fabs(ibm->Bvel[nelmt].y)+fabs(ibm->Bvel[nelmt].z);}
		 	if(test < 1.e-7)
			{  
				riter = 0.;
				double shvel_sum=0.0;
				double bvelu_sum=0.0;
				double bvelv_sum=0.0;
				double bvelw_sum=0.0;
				double c_sum=0.0;
	
				int ii,jj;
				for(ii=0;ii<100;ii++)
				{
					nii=ibm->c2ac[nelmt].c[ii];	// xiaolei add
					if (nii>=0) {    // xiaolei add 
					if(ibm->nf_z[nii]<1.e-6 || ibm->elmt_depth[nii]>0.3)
                                         { 
                                         }
                                        else
                                         {
		 				double testi;
		 				if(LiveBed){ testi = fabs(ibm->Shvel[nii])+fabs(ibm->Bvel[nii].x)+fabs(ibm->Bvel[nii].y)+fabs(ibm->Bvel[nii].z) + fabs(ibm->C[nii]);}
                 				else { testi = fabs(ibm->Shvel[nii])+fabs(ibm->Bvel[nii].x)+fabs(ibm->Bvel[nii].y)+fabs(ibm->Bvel[nii].z);}
                                           	if( nii == nelmt || testi < 1.e-7 )
                                           	{
                                           	}
                                           	else
                                           	{					                         			
							riter = riter + 1.;
							shvel_sum+=Shvel_tmp[nii];
							if(LiveBed) c_sum+=C_tmp[nii];
							bvelu_sum+=Bvel_tmp[nii].x;
							bvelv_sum+=Bvel_tmp[nii].y;
							bvelw_sum+=Bvel_tmp[nii].z;
					   	}
			              	}
				      	}	
			      	}


				ibm->Shvel[ nelmt ] = shvel_sum / (riter+1.e-17);
				if(LiveBed) ibm->C[ nelmt ] = c_sum / (riter+1.e-17);
				ibm->Bvel[ nelmt].x = bvelu_sum / (riter+1.e-17);
				ibm->Bvel[ nelmt].y = bvelv_sum / (riter+1.e-17);
				ibm->Bvel[ nelmt].z = bvelw_sum / (riter+1.e-17);		


	              }
          	}
    	}


	//Smoothing the variables at all cells to correct the high gradient regions
	for(i=0;i<number_smooth;i++){

		for( nelmt = 0; nelmt < ibm->n_elmt; nelmt++ ) {

			Shvel_tmp[nelmt]=ibm->Shvel[ nelmt ];
			if(LiveBed) C_tmp[nelmt]=ibm->C[ nelmt ];
			Bvel_tmp[nelmt].x=ibm->Bvel[ nelmt].x;
			Bvel_tmp[nelmt].y=ibm->Bvel[ nelmt].y;
			Bvel_tmp[nelmt].z=ibm->Bvel[ nelmt].z;
	
		}


		for( nelmt = 0; nelmt < ibm->n_elmt; nelmt++ )
		{
			if(ibm->nf_z[nelmt] < 1.e-7 || ibm->elmt_depth[nelmt] > 0.3)
			{
				ibm->Shvel[ nelmt ] = 0.;
				ibm->Bvel[ nelmt ].x = 0.;
				ibm->Bvel[ nelmt ].y = 0.;
				ibm->Bvel[ nelmt ].z = 0.;
			}
			else
			{
				riter = 1.;
				double shvel_sum=ibm->Shvel[ nelmt ] ;
				double bvelu_sum=ibm->Bvel[ nelmt ].x;
				double bvelv_sum=ibm->Bvel[ nelmt ].y;
				double bvelw_sum=ibm->Bvel[ nelmt ].z;
				double c_sum=0.0;
				if (LiveBed) c_sum=ibm->C[ nelmt ];
	
				int ii,jj;
				for(ii=0;ii<100;ii++)
				{
					nii=ibm->c2ac[nelmt].c[ii];	// xiaolei add
					if (nii>=0) {    // xiaolei add 
					if(ibm->nf_z[nii]<1.e-6 || ibm->elmt_depth[nii]>0.3)
                                        { 
                                        }
                                        else
                                        {
						riter = riter + 1.;
						shvel_sum+=Shvel_tmp[nii];
						if(LiveBed) c_sum+=C_tmp[nii];
						bvelu_sum+=Bvel_tmp[nii].x;
						bvelv_sum+=Bvel_tmp[nii].y;
						bvelw_sum+=Bvel_tmp[nii].z;
			              	}
				      	}	
			      	}


				ibm->Shvel[ nelmt ] = shvel_sum / (riter+1.e-17);
				if(LiveBed) ibm->C[ nelmt ] = c_sum / (riter+1.e-17);
				ibm->Bvel[ nelmt].x = bvelu_sum / (riter+1.e-17);
				ibm->Bvel[ nelmt].y = bvelv_sum / (riter+1.e-17);
				ibm->Bvel[ nelmt].z = bvelw_sum / (riter+1.e-17);		


	              	}
          	}
    	}


	// finalizing the projection of velocity vector and bed shear stress on body  surface 
	for( elmt = 0; elmt < n_elmt; elmt++ )
	{
		nfx = ibm->nf_x[ elmt ];
		nfy = ibm->nf_y[ elmt ];
		nfz = ibm->nf_z[ elmt ];

		ucx = ibm->Bvel[ elmt ].x;
		ucy = ibm->Bvel[ elmt ].y;
		ucz = ibm->Bvel[ elmt ].z;


		if( nfz < 1.e-7 || ibm->elmt_depth[elmt] > 0.3)
		{
			ibm->Bvel[ elmt ].x = 0.;
			ibm->Bvel[ elmt ].y = 0.;
			ibm->Bvel[ elmt ].z = 0.;
                        ibm->Shvel[ elmt ] = 0.;
       		}
		else   // U_par = U - (U.n)n => (U_par)i = Ui - (Uj.nj).ni  : for every i we have j=1,2,3
		{
			ibm->Bvel[ elmt ].x = ucx - (ucx * nfx + ucy * nfy + ucz * nfz) * nfx;
	        	ibm->Bvel[ elmt ].y = ucy - (ucx * nfx + ucy * nfy + ucz * nfz) * nfy;
			ibm->Bvel[ elmt ].z = ucz - (ucx * nfx + ucy * nfy + ucz * nfz) * nfz;
                        //ibm->Shvel[ elmt ] = ibm->Shvel[elmt]; 
		}
	}

	PetscFree(Bvel_tmp);
      	PetscFree(C_tmp);
      	PetscFree(Shvel_tmp);


	return ( 0 );
}





double integrating (double a , double b, double rp)
{
int i;
double x,N=30;
double delx=(b-a)/(2.*N);
    double sumeven=0.;
    for(i=1;i<N; i++)
       {
        x=a+delx*2.*i;
        sumeven += pow(x,1.5) * exp(-0.5*(x-rp)*(x-rp));
       }

    double sumodd =0.;
    for(i=1;i<N+1; i++)
       {
        x=a+delx*(2.*i-1.);
        sumodd += pow(x,1.5) * exp(-0.5*(x-rp)*(x-rp));
       }
 
return delx*(pow(a,1.5) * exp(-0.5*(a-rp)*(a-rp)) + pow(b,1.5) * exp(-0.5*(b-rp)*(b-rp))+2.*sumeven+4.*sumodd)/3.;
}



PetscErrorCode	bed_concentration( UserCtx * user, IBMNodes * ibm, int ti, PetscInt tistart )
{
	DM	 da = user->da, 
		 fda = user->fda;
	Cmpnts		 *	Bvel;
	PetscReal	 *	BShS, *Shvel, 
			 *	BCShS;
	PetscReal	 *	SCont;
	int		n_elmt	= ibm->n_elmt, 
	         		n_v  = ibm->n_v;
	int		i, j, k;
	PetscReal		sb,sc;
	int		ni,nii;
	PetscReal		ucx,ucy,ucz;
	IBMInfo  	 *	ibminfo;
	IBMListNode      *	current;
	int		elmt;
	PetscReal		riter;
	PetscReal		nfx, nfy, nfz;
	PetscReal		xc,yc, zc;
	PetscReal		tmp, tt;
	PetscReal               D50		 = 0.0018;///jhoke-crossvane-rockvane 0.0011 bends 0.0005
	PetscReal               D90		 = 0.05;

	D50=particle_D50;
	
	ibm->Deltab = 0.01;//0.04;// 2.* D50 ;//2.0 * D50 * 7./2.5 ;
	PetscReal       fdepth	= 1.0, //1.15, // jhoke 0.172, //cross-vane 0.179, // 0.312, //0.4											    
	                particle_sg  = 1.922,											    
			fluid_density  = 1000.0,						  			    
			particle_ws,											    
			Shields_param,														    
			CShields_param,  												    
			Dstar,														    
			visk  = 1.e-6,											    
			gi	= 9.806,									  			    
			sgd_tmp,				  										    
			Ctau_0,  												    
			Custar_0,												    
			U_bulk_vel	= 1.0, // jhoke  0.355,//cross-vane 0.33, //0.285*1.5, //times 1.						    
			max_bed_slope,										    
			angle_of_dev,
                        ttau;
       	PetscReal        bita=PI*0.51/6.,DD;

	particle_sg=particle_dens/fluid_density;

       int         stochastic = 0, Fredsoe=1, VanRijn=0;
       if(stochastic){Fredsoe=0; VanRijn=1;} 
                  						    
      if(ti==tistart || ti==0){ PetscPrintf(PETSC_COMM_WORLD, "Stochastic: %d \n",stochastic);
      PetscPrintf(PETSC_COMM_WORLD, "Fredsoe: %d \n",Fredsoe);
      PetscPrintf(PETSC_COMM_WORLD, "VanRijn: %d \n",VanRijn);
      PetscPrintf(PETSC_COMM_WORLD, "D50: %e \n",D50);
      PetscPrintf(PETSC_COMM_WORLD, "Delta_b: %e \n",deltab);
      PetscPrintf(PETSC_COMM_WORLD, "Flow_Depth (m): %e \n",fdepth);
      PetscPrintf(PETSC_COMM_WORLD, "Sediment Particles Density (kg/m3): %e \n",particle_dens);
      PetscPrintf(PETSC_COMM_WORLD, "Flow_Bulk_Velocity (m/sec): %e \n",U_bulk_vel);}
//PetscReal   z_minimum,z_maximum,depth_elmt;
//---------------------------------------------------------------------------
// computing critical bed shear stress on a flat bed, by shields curve critera 

	sgd_tmp   = ( particle_sg - 1. ) * gi * D50;
	particle_ws
	= sqrt( sgd_tmp )
	   * ( sqrt( 2. / 3. + 36. * visk * visk / ( sgd_tmp * D50 * D50 ) )
			- 6. * visk / ( sqrt( sgd_tmp ) * D50 ) );
	Dstar = D50 * pow(((particle_sg-1.) * gi / ( visk * visk )), 0.3333333333 );


      	PetscPrintf(PETSC_COMM_WORLD, "#### Dstar: %le \n",Dstar);

        int shields = 0, soulsby = 1, old_method = 0, whitehouse = 1; // soulseby and Whitehouse methods: see J Geoph. Res. vol 115, 2010, Chou & Fringer

        if(shields){

	if( Dstar <= 4. )
		CShields_param = 0.24 / Dstar;
	if( Dstar > 4. && Dstar <= 10. )
		CShields_param = 0.14 / pow( Dstar, 0.64 );
	if( Dstar > 10. && Dstar <= 20. )
		CShields_param = 0.04 / pow( Dstar, 0.1 );
	if( Dstar > 20. && Dstar <= 150. )
		CShields_param = 0.013 * pow( Dstar, 0.29 );
	if( Dstar > 150. )
		CShields_param = 0.055;
        } 
        if (soulsby) {
        CShields_param = 0.3/ ( 1. + 1.2 * Dstar) + 0.055 * (1. - exp (-0.02 * Dstar)) ;
        }
        
	if(VanRijn)
        {
         Ctau_0 = CShields_param * sgd_tmp * fluid_density;
	 Custar_0	 = sqrt( Ctau_0 / fluid_density );// change Nov 17 2009---canceled nov 18 2009
        }
      
	if(Fredsoe)
        {
         Ctau_0 = CShields_param; 
	}
	
        // non-dimensionalizing
	//Custar_0	/= U_bulk_vel;
	//Ctau_0		 = Custar_0 * Custar_0;
	particle_ws /= U_bulk_vel;
	D50  		/= fdepth;
	D90  		/= fdepth;
	ibm->Deltab		/= fdepth;
	
// computing critical bed shear stress on inclined bed
	for( elmt = 0; elmt < n_elmt; elmt++ )
	{
		//xc	= ibm->cent_x[ elmt ];
		//yc	= ibm->cent_y[ elmt ];
		//zc	= ibm->cent_z[ elmt ];

		nfx = ibm->nf_x[ elmt ];
		nfy = ibm->nf_y[ elmt ];
		nfz = ibm->nf_z[ elmt ];

		ucx = ibm->Bvel[ elmt ].x;
		ucy = ibm->Bvel[ elmt ].y;
		ucz = ibm->Bvel[ elmt ].z;

		if( nfz < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt])
		{
			ibm->BCShS[ elmt ] = 100.0;
		}
		else
		{
			if (old_method) {
                        max_bed_slope
			= sqrt( ( nfy / nfz ) * ( nfy / nfz ) + ( nfx / nfz ) * ( nfx / nfz ) );

			if(fabs(max_bed_slope)<1.e-6)
                        {
                        tmp=1.0;
                        }
                        else
                        {
                        max_bed_slope = atan( max_bed_slope );

			angle_of_dev  = ( nfx / (nfz) ) * ucx + ( nfy / (nfz) ) * ucy;

                          //PetscPrintf(PETSC_COMM_WORLD, "angle of dev %le \n",angle_of_dev);
                        angle_of_dev /= ( sqrt( ( nfy / (nfz) ) * ( nfy / (nfz) ) + ( nfx / (nfz) ) * ( nfx / (nfz) )));

			angle_of_dev /= ( sqrt( ucx * ucx + ucy * ucy ));
                        
                        if(fabs(angle_of_dev)>1.)angle_of_dev=angle_of_dev*PetscMax(.99,angle_of_dev)/fabs(angle_of_dev);
			
                        angle_of_dev = acos( angle_of_dev );

			tmp = sin( angle_of_dev ) * sin( angle_of_dev ) * tan( max_bed_slope )
				   * tan( max_bed_slope );
			tmp /= ( tan( Angle_repose ) * tan( Angle_repose ) );
			tmp = PetscMax((1. - tmp),0.);
			tmp = cos( max_bed_slope ) * sqrt( tmp );
			tmp = tmp - cos( angle_of_dev ) * sin( max_bed_slope ) / tan( Angle_repose );
                        }
                 	
                        //ibm->BCShS[ elmt ] = Ctau_0 *tmp*nfz*0.75/tmp; //times a 0.75 to reduce the critical bed shear stress
                 	//ibm->BCShS[ elmt ] = Ctau_0 *tmp; //times a 0.75 to reduce the critical bed shear stress
                 	ibm->BCShS[ elmt ] = Ctau_0; //times a 0.75 to reduce the critical bed shear stress
                // 	ibm->BCShS[ elmt ] = 0.03; //times a 0.75 to reduce the critical bed shear stress
                // 	DD=Ctau_0/tan(Angle_repose);
                         }
 
                         if (whitehouse){
                         double tita_x = atan ( - nfx / nfz );
                         double tita_y = atan ( - nfy / nfz );
                         double tita1  = ucx * sin (tita_x) + ucy * sin (tita_y); 
                         double tita2  = sqrt (ucx * ucx + ucy * ucy);
                         double tita3  = tita1 / (tita2+1.e-19);
			double tita;
                         if (fabs(tita3) >= 1. || fabs(tita3)<= -1.) {tmp =1.;} else {
                         tita   = asin (tita3);
                         tmp           = sin ( tita + Angle_repose ) / sin (Angle_repose);}
                         
                         ibm->BCShS [ elmt ] = Ctau_0 * tmp;
                                       }
		}
	}

// Smoothening the computed critical bed shear stress over entire domian
//BCShS_Smoothening(ibm, ti, tistart);


// computing bed concentration by Van Rijn relation for equilibrium condition // Fredsoe method added as well
	for( elmt = 0; elmt < n_elmt; elmt++ )
	{
		//xc	= ibm->cent_x[ elmt ];
		//yc	= ibm->cent_y[ elmt ];
		//zc	= ibm->cent_z[ elmt ];
                
                int rigid = ibm->Rigidity[elmt];
                 
		nfx = ibm->nf_x[ elmt ];
		nfy = ibm->nf_y[ elmt ];
		nfz = ibm->nf_z[ elmt ];

	//	if( nfz < 1.e-6 )
		if( nfz < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || rigid)
		{
		        ibm->SCont[ elmt ] = 0.0;
                        ibm->BShS[elmt] = 0.0;
                        ibm->BCShS[elmt] = 100.0;
		}
		else
		{
                     if(VanRijn){
                        //ttau=sqrt(ibm->BShS[elmt])*U_bulk_vel*1.; //inlet vel. increase by 1.5 times
                        //ttau=ttau*ttau*fluid_density;    // dimensionalizing bed shear stress
                        //ibm->BShS[elmt]=ttau;
                        ibm->BShS[elmt] = fluid_density*(ibm->Shvel[elmt]*U_bulk_vel)*(ibm->Shvel[elmt]*U_bulk_vel);//+ DD*(nfx*nfz+nfy*nfz+nfz*nfz-1.);
                        

                                  if(!stochastic)
                                              {

			tt = (ibm->BShS[ elmt ] - ibm->BCShS[ elmt ])/ibm->BCShS[elmt];

			tt = PetscMax( tt, 0. );
                                       
                       //*************************** 1. first strategy ********************************************************************
                         ibm->SCont[ elmt ] =  0.015 *(D50/deltab)*pow( tt, 1.5 )/pow( Dstar,0.3);
                                                                     
                       //**************************** 1-2. strategy ******************************************************************

		       //      ibm->SCont[ elmt ] = PetscMax(0.,(0.015 * (D50 / deltab ) * pow( tt, 1.5 ) / pow( Dstar, 0.3 )+
                       //                            ibm->netflux_old[elmt]*user->dt/(ibm->dA[elmt]*deltab)));

                       //**************************** 2. second strategy ******************************************************************
                       // SCont is the portable part of bed cell concent. Depending on the last time step deposition or scour
                       // the current time step cell concentration can be more or less than the Ca by Van Rijn formula here 
                       // but if tt>0 it gonna be always greater than zero:
                       /* if (tt>0.)
                            {
                       
		             ibm->SCont[ elmt ] = PetscMax(0.,(0.015 * (D50 / deltab ) * pow( tt, 1.5 ) / pow( Dstar, 0.3 )+
                                                   ibm->netflux_old[elmt]*user->dt/(ibm->dA[elmt]*deltab)));
                                                //   ibm->netflux_old[elmt]*3.0/(ibm->dA[elmt]*deltab)));
                            }
                        else
                            {

		             ibm->SCont[ elmt ] = 0.015 * (D50 / deltab ) * pow( tt, 1.5 ) / pow( Dstar, 0.3 )+
                                                  PetscMax(0.,((ibm->netflux_old[elmt])*(user->dt)/(ibm->dA[elmt]*deltab)));
                                                  //PetscMax(0.,((ibm->netflux_old[elmt])*(3.0)/(ibm->dA[elmt]*deltab)));
                            }*/
                       //**************************** 3. Third one  ************************************************************************
                       // SCont is the portable part of bed cell sediemnt cont anb the portable sed-concent
                       // at a cell can not be more than the Ca which is the max. portable sed-concent by Van Rijn formula here 
                       // but because of deposition at last time step it could be less than Ca :
                       /* if (tt>0.)
                            {
                                  ibm->SCont[ elmt ] = 0.015 * (D50 / deltab ) * pow( tt, 1.5 ) / pow( Dstar, 0.3 )+
                                                   PetsMin(0., (ibm->netflux_old[elmt]*user->dt/(ibm->dA[elmt]*deltab)));
                            }
                        else
                            {
                             ibm->SCont[ elmt ] =  0.015 * (D50 / deltab ) * pow( tt, 1.5 ) / pow( Dstar, 0.3 )+
                                                  PetscMax(0.,((ibm->netflux_old[elmt])*(user->dt)/(ibm->dA[elmt]*deltab)));
                            }*/
                       //****************************************************************************************************************                                            
                                      }
                                      if(stochastic)
                                      {
                                        double Int1, Int2;
                                        double sigma = 0.4*fabs(ibm->BShS[elmt]);
                                        double Tau_Cr= 1.5*Ctau_0;
                                        double Tau_Cr1= ibm->BCShS[elmt];
                                        double Tau_Cr2= - ibm->BCShS[elmt];     
                                        double r = (ibm->BShS[elmt]-Tau_Cr1)/(sigma+1.e-8);
                                        double p = (-ibm->BShS[elmt]-Tau_Cr2)/(sigma+1.e-8);
                                        double a1 = PetscMax(0.0,r - 4.);
                                        double b1 = r + 5.;            
                                        double a2 = PetscMax(0.0,p - 4.);
                                        double b2 = p + 5.;       
                                        if(r <  -3.0)  Int1 = 0.0;     
                                        if(r >= -3.0)  Int1 = integrating (a1,b1,r); 
                                        if(p <  -3.0)  Int2 = 0.0;
                                        if(p >= -3.0)  Int2 = integrating (a2,b2,p);
                                        
                                        ibm->SCont[elmt] =  0.03 * (D50/deltab)*(Int1+Int2)/pow( Dstar,0.3);
                                      }                                             
                         }
                  
        
                if(Fredsoe) // ali 10 March 2010
                         {
                          ttau=(ibm->Shvel[elmt]*U_bulk_vel)*(ibm->Shvel[elmt]*U_bulk_vel); // the inlet vel. increase by 1.5 times
                          ibm->BShS[elmt]=ttau/(sgd_tmp);

                         
			  tt = ibm->BShS[ elmt ] - ibm->BCShS[ elmt ];

			  tt = PetscMax( tt, 1.e-7 );
                         
                          PetscReal bita=PI*0.51/6.;
                          
                          tt= tt*tt*tt*tt;
                          bita=bita*bita*bita*bita;
                          
                          ibm->SCont[elmt]=pow((1.+ bita/tt),-0.25)*PI/6.;
               
			  if (ibm->BShS[ elmt ] - ibm->BCShS[ elmt ]<=0.0) ibm->SCont[elmt]=0.0;  // xiaolei
 
                          //Hitting the rigid bed; if a cell is on the rigid bed then the bed concentration is zero
                          //if(ibm->cent_x[elmt]<= 2) ibm->SCont[elmt]=0.0;                                   
                         }

		}
        // if(RigidBed && ibm->cent_z[elmt] <0.225) {ibm->SCont[elmt] = 0.0;}                                                                
	}

return ( 0 );		// Ali completed 2 nov. 2009		
}


                       
PetscErrorCode	sediment_flux( UserCtx * user, IBMNodes * ibm, int ti, PetscInt tistart)
{
	DM			   da  = user->da, 
					fda  = user->fda;

	Cmpnts		 * Bvel;
	PetscReal	 * F_flux_12, 
				 * F_flux_13, 
				 * F_flux_23;
	PetscReal	 * SCont;
	int	   n_elmt  = ibm->n_elmt, 
					n_v  = ibm->n_v;
	IBMInfo  	 * ibminfo;
	IBMListNode  * current;
	int   elmt,nbelmt,	n1e,n2e,n3e;
	PetscReal	   riter;
	PetscReal	   nfx, 
					nfy, 
					nfz, 
					dx12, 
					dy12, 
					dz12, 
					dx13, 
					dy13, 
					dz13, 
					dx23, 
					dy23, 
					dz23, 
					dr, 
					ds;
	PetscReal	   xc, yc, zc;
	PetscReal	   tmp;
	PetscReal	  caver, Ave_face12_u, 
					Ave_face12_v, 
					Ave_face12_w, 
					Ave_face13_u, 
					Ave_face13_v, 
					Ave_face13_w, 
					Ave_face23_u, 
					Ave_face23_v, 
					Ave_face23_w, 
					nfx12, 
					nfy12, 
					nfx13, 
					nfy13, 
					nfx23, 
					nfy23, 
					nor_vel_to_face12, 
					nor_vel_to_face13, 
					nor_vel_to_face23, 
					le12x, 
					le12y, 
					le13x, 
					le13y, 
					le23x, 
					le23y, 
					e12x, 
					e12y, 
					e13x, 
					e13y, 
					e23x, 
					e23y, 
					le_dot_n12, 
					le_dot_n13, 
					le_dot_n23;
      PetscReal  d_x,d_y,delZ, delX;
      PetscReal  u_face, v_face, c_face, landau,landav, landac,phiu, phiv, phic, phi1u, phi2u, phi1v, phi2v, phi1c, phi2c, RL, fR, fL, fx, beta=1./3.;
      PetscReal  cof=0.;
      int   upwind_v = 0;
      int   central_v = 1;
      int   upwind_c = 0;
      int   central_c = 1;
      int   gamma_c = 0;
      int   gamma_v = 0;
      int   SOU_v = 0;
      if(ti==tistart || ti==0){ PetscPrintf(PETSC_COMM_WORLD, "COF: %e \n",cof);
       PetscPrintf(PETSC_COMM_WORLD, "upwind_v and upwind_c: %d %d \n",upwind_v,upwind_c);
       PetscPrintf(PETSC_COMM_WORLD, "central_v and central_c: %d %d \n",central_v,central_c);
       PetscPrintf(PETSC_COMM_WORLD, "GAMMA_v and GAMMA_c: %d %d \n",gamma_v,gamma_c);}

	double Good, Number; 
	double Good_max=0.0; 
	Good=0.0;
	Number=0.0;

	/*
	double **a, *b1, *b2, *b3, *b4, *c1, *c2, *c3, *c4;

	a = (double**) malloc(sizeof(double)*6);
	b1 = (double*) malloc(sizeof(double)*6);
	b2 = (double*) malloc(sizeof(double)*6);
	b3 = (double*) malloc(sizeof(double)*6);
	b4 = (double*) malloc(sizeof(double)*6);
	c1 = (double*) malloc(sizeof(double)*6);
	c2 = (double*) malloc(sizeof(double)*6);
	c3 = (double*) malloc(sizeof(double)*6);
	c4 = (double*) malloc(sizeof(double)*6);

	for(int k=0;k<6;k++) {
		a[k] = (double*) malloc(sizeof(double)*6);
	}
	*/



	/*
	double a[6][6];
	double a1[6][6];


	double b1[6], c1[6];
	double b2[6], c2[6];
	double b3[6], c3[6];
	double b4[6], c4[6];
	double xi[100], yi[100], zi[100];
	double u_x[100], u_y[100], u_z[100], cc[100];
	int label[100];
	double _xc, _yc, _zc;

	double ux_avg, uy_avg, uz_avg, cc_avg;
	int N_ac;	

	double u_node[ibm->n_v],v_node[ibm->n_v],w_node[ibm->n_v],C_node[ibm->n_v];
	//double a[6][6], b[6], c[6];
        for( elmt = 0; elmt < n_elmt; elmt++ ) {
                int rigid = ibm->Rigidity[elmt];
		if(ibm->nf_z[elmt]<1.e-6 || ibm->elmt_depth[elmt]>0.3 || rigid) 
		{
		}
		else {
			n1e = ibm->nv1[ elmt ];
			n2e = ibm->nv2[ elmt ];
			n3e = ibm->nv3[ elmt ];

			N_ac=0;
			_xc=0.0;
			_yc=0.0;
			_zc=0.0;
			ux_avg=0.0;
			uy_avg=0.0;
			uz_avg=0.0;
			cc_avg=0.0;
			int ii,jj, nii;
			for(jj=0;jj<6;jj++) {
				for(ii=0;ii<6;ii++) {
					a[jj][ii]=0.0;			
				}
				b1[jj]=0.0;
				c1[jj]=0.0;
				b2[jj]=0.0;
				c2[jj]=0.0;
				b3[jj]=0.0;
				c3[jj]=0.0;
				b4[jj]=0.0;
				c4[jj]=0.0;


			}


			xi[0]=ibm->cent_x[elmt];
			yi[0]=ibm->cent_y[elmt];
			zi[0]=ibm->cent_z[elmt];
			u_x[0]=ibm->Bvel[elmt].x;
			u_y[0]=ibm->Bvel[elmt].y;
			u_z[0]=ibm->Bvel[elmt].z;
			cc[0]=ibm->SCont[elmt];
			_xc=ibm->cent_x[elmt];
			_yc=ibm->cent_y[elmt];
			_zc=ibm->cent_z[elmt];
			ux_avg=ibm->Bvel[elmt].x;
			uy_avg=ibm->Bvel[elmt].y;
			uz_avg=ibm->Bvel[elmt].z;
			cc_avg=ibm->SCont[elmt];
			N_ac=1;
			label[0]=1;

	
			for(ii=1;ii<100;ii++) {
				xi[ii]=0.0;			
				yi[ii]=0.0;			
				zi[ii]=0.0;			
				label[ii]=0.0;			
				nii=ibm->c2ac[elmt].c[ii];	// xiaolei add
				if (nii>=0) {    // xiaolei add 
                			int rigid = ibm->Rigidity[nii];
					if(ibm->nf_z[nii]<1.e-6 || ibm->elmt_depth[nii]>0.3 || rigid ) 
                	                { 
                        	        }
	                                else
        	                        {
						xi[ii]=ibm->cent_x[nii];
						yi[ii]=ibm->cent_y[nii];
						zi[ii]=ibm->cent_z[nii];
						u_x[ii]=ibm->Bvel[nii].x;
						u_y[ii]=ibm->Bvel[nii].y;
						u_z[ii]=ibm->Bvel[nii].z;
						cc[ii]=ibm->SCont[nii];
						_xc+=ibm->cent_x[nii];
						_yc+=ibm->cent_y[nii];
						_zc+=ibm->cent_z[nii];
						ux_avg+=ibm->Bvel[nii].x;
						uy_avg+=ibm->Bvel[nii].y;
						uz_avg+=ibm->Bvel[nii].z;
						cc_avg+=ibm->SCont[nii];
						N_ac+=1;
						label[ii]=1;
					}
				}
			}

			_xc/=double(N_ac);
			_yc/=double(N_ac);
			_zc/=double(N_ac);
			ux_avg/=double(N_ac);
			uy_avg/=double(N_ac);
			uz_avg/=double(N_ac);
			cc_avg/=double(N_ac);


			for(ii=0;ii<100;ii++) {
				if (label[ii]) {
					xi[ii]-=_xc;
					yi[ii]-=_yc;
				}
			}



			if (N_ac<6) {
		        	u_node[n1e] = ux_avg;
		         	v_node[n1e] = uy_avg;
		         	w_node[n1e] = uz_avg;
		         	C_node[n1e] = cc_avg;

		        	u_node[n2e] = ux_avg;
		         	v_node[n2e] = uy_avg;
		         	w_node[n2e] = uz_avg;
		         	C_node[n2e] = cc_avg;

		        	u_node[n3e] = ux_avg;
		         	v_node[n3e] = uy_avg;
		         	w_node[n3e] = uz_avg;
		         	C_node[n3e] = cc_avg;



			} else {
				for(ii=0;ii<100;ii++) {
					if (label[ii]) {
						b1[0]+=u_x[ii];
						b1[1]+=u_x[ii]*xi[ii];
						b1[2]+=u_x[ii]*yi[ii];
						b1[3]+=u_x[ii]*xi[ii]*yi[ii];
						b1[4]+=u_x[ii]*xi[ii]*xi[ii];
						b1[5]+=u_x[ii]*yi[ii]*yi[ii];

						b2[0]+=u_y[ii];
						b2[1]+=u_y[ii]*xi[ii];
						b2[2]+=u_y[ii]*yi[ii];
						b2[3]+=u_y[ii]*xi[ii]*yi[ii];
						b2[4]+=u_y[ii]*xi[ii]*xi[ii];
						b2[5]+=u_y[ii]*yi[ii]*yi[ii];

						b3[0]+=u_z[ii];
						b3[1]+=u_z[ii]*xi[ii];
						b3[2]+=u_z[ii]*yi[ii];
						b3[3]+=u_z[ii]*xi[ii]*yi[ii];
						b3[4]+=u_z[ii]*xi[ii]*xi[ii];
						b3[5]+=u_z[ii]*yi[ii]*yi[ii];

						b4[0]+=cc[ii];
						b4[1]+=cc[ii]*xi[ii];
						b4[2]+=cc[ii]*yi[ii];
						b4[3]+=cc[ii]*xi[ii]*yi[ii];
						b4[4]+=cc[ii]*xi[ii]*xi[ii];
						b4[5]+=cc[ii]*yi[ii]*yi[ii];



						double _a0=1, _a1=xi[ii], _a2=yi[ii], _a3=xi[ii]*yi[ii], _a4=xi[ii]*xi[ii], _a5=yi[ii]*yi[ii];
						a[0][0]+=_a0; a[0][1]+=_a1; a[0][2]+=_a2; a[0][3]+=_a3; a[0][4]+=_a4; a[0][5]+=_a5;
						a[1][0]+=_a0*xi[ii]; a[1][1]+=_a1*xi[ii]; a[1][2]+=_a2*xi[ii]; a[1][3]+=_a3*xi[ii]; a[1][4]+=_a4*xi[ii]; a[1][5]+=_a5*xi[ii];
						a[2][0]+=_a0*yi[ii]; a[2][1]+=_a1*yi[ii]; a[2][2]+=_a2*yi[ii]; a[2][3]+=_a3*yi[ii]; a[2][4]+=_a4*yi[ii]; a[2][5]+=_a5*yi[ii];
						a[3][0]+=_a0*xi[ii]*yi[ii]; a[3][1]+=_a1*xi[ii]*yi[ii]; a[3][2]+=_a2*xi[ii]*yi[ii]; a[3][3]+=_a3*xi[ii]*yi[ii]; a[3][4]+=_a4*xi[ii]*yi[ii]; a[3][5]+=_a5*xi[ii]*yi[ii];
						a[4][0]+=_a0*xi[ii]*xi[ii]; a[4][1]+=_a1*xi[ii]*xi[ii]; a[4][2]+=_a2*xi[ii]*xi[ii]; a[4][3]+=_a3*xi[ii]*xi[ii]; a[4][4]+=_a4*xi[ii]*xi[ii]; a[4][5]+=_a5*xi[ii]*xi[ii];
						a[5][0]+=_a0*yi[ii]*yi[ii]; a[5][1]+=_a1*yi[ii]*yi[ii]; a[5][2]+=_a2*yi[ii]*yi[ii]; a[5][3]+=_a3*yi[ii]*yi[ii]; a[5][4]+=_a4*yi[ii]*yi[ii]; a[5][5]+=_a5*yi[ii]*yi[ii];
					}
				}			

				int ia, ja;
				for(ia=0;ia<6;ia++)
				for(ja=0;ja<6;ja++) {
					a1[ja][ia]=a[ja][ia];
				}

				AGAUS (a, b1, c1);

				for(ia=0;ia<6;ia++)
				for(ja=0;ja<6;ja++) {
					a[ja][ia]=a1[ja][ia];
				}

				AGAUS (a, b2, c2);

				for(ia=0;ia<6;ia++)
				for(ja=0;ja<6;ja++) {
					a[ja][ia]=a1[ja][ia];
				}

				AGAUS (a, b3, c3);

				for(ia=0;ia<6;ia++)
				for(ja=0;ja<6;ja++) {
					a[ja][ia]=a1[ja][ia];
				}

				AGAUS (a, b4, c4);
		
				double xx, yy;

				xx=ibm->x_bp[n1e]-_xc; yy=ibm->y_bp[n1e]-_yc; 
				u_node[n1e]=c1[0]+c1[1]*xx+c1[2]*yy+c1[3]*xx*yy+c1[4]*xx*xx+c1[5]*yy*yy;
				v_node[n1e]=c2[0]+c2[1]*xx+c2[2]*yy+c2[3]*xx*yy+c2[4]*xx*xx+c2[5]*yy*yy;
				w_node[n1e]=c3[0]+c3[1]*xx+c3[2]*yy+c3[3]*xx*yy+c3[4]*xx*xx+c3[5]*yy*yy;
				C_node[n1e]=c4[0]+c4[1]*xx+c4[2]*yy+c4[3]*xx*yy+c4[4]*xx*xx+c4[5]*yy*yy;

				xx=ibm->x_bp[n2e]-_xc; yy=ibm->y_bp[n2e]-_yc; 
				u_node[n2e]=c1[0]+c1[1]*xx+c1[2]*yy+c1[3]*xx*yy+c1[4]*xx*xx+c1[5]*yy*yy;
				v_node[n2e]=c2[0]+c2[1]*xx+c2[2]*yy+c2[3]*xx*yy+c2[4]*xx*xx+c2[5]*yy*yy;
				w_node[n2e]=c3[0]+c3[1]*xx+c3[2]*yy+c3[3]*xx*yy+c3[4]*xx*xx+c3[5]*yy*yy;
				C_node[n2e]=c4[0]+c4[1]*xx+c4[2]*yy+c4[3]*xx*yy+c4[4]*xx*xx+c4[5]*yy*yy;


				xx=ibm->x_bp[n3e]-_xc; yy=ibm->y_bp[n3e]-_yc; 
				u_node[n3e]=c1[0]+c1[1]*xx+c1[2]*yy+c1[3]*xx*yy+c1[4]*xx*xx+c1[5]*yy*yy;
				v_node[n3e]=c2[0]+c2[1]*xx+c2[2]*yy+c2[3]*xx*yy+c2[4]*xx*xx+c2[5]*yy*yy;
				w_node[n3e]=c3[0]+c3[1]*xx+c3[2]*yy+c3[3]*xx*yy+c3[4]*xx*xx+c3[5]*yy*yy;
				C_node[n3e]=c4[0]+c4[1]*xx+c4[2]*yy+c4[3]*xx*yy+c4[4]*xx*xx+c4[5]*yy*yy;

			}

		}

	}



        for( elmt = 0; elmt < n_elmt; elmt++ ) {
                int rigid = ibm->Rigidity[elmt];
		if(ibm->nf_z[elmt]<1.e-6 || ibm->elmt_depth[elmt]>0.3 || rigid) 
		{
		}
		else {
			n1e = ibm->nv1[ elmt ];
			n2e = ibm->nv2[ elmt ];
			n3e = ibm->nv3[ elmt ];
	
			ibm->Bvel[elmt].x=(u_node[n1e]+u_node[n2e]+u_node[n3e])/3;
			ibm->Bvel[elmt].y=(v_node[n1e]+v_node[n2e]+v_node[n3e])/3;
			ibm->Bvel[elmt].z=(w_node[n1e]+w_node[n2e]+w_node[n3e])/3;
			ibm->SCont[elmt]=(C_node[n1e]+C_node[n2e]+C_node[n3e])/3;
		}

	}

	*/

	// computing fluxes through all three faces: 1-->2 , 1-->3 , 2-->3
	for( elmt = 0; elmt < n_elmt; elmt++ )
	{
		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt])
		{
			ibm->F_flux_12[ elmt ] = 0.;
			ibm->F_flux_13[ elmt ] = 0.;
			ibm->F_flux_23[ elmt ] = 0.;
		}
		else
		{
			n1e  = ibm->nv1[ elmt ];
			n2e  = ibm->nv2[ elmt ];
			n3e  = ibm->nv3[ elmt ];
			
                        xc	 = ibm->cent_x[ elmt ];
			yc	 = ibm->cent_y[ elmt ];
			zc	 = ibm->cent_z[ elmt ];

			dx12 = -(ibm->x_bp[ n2e ] - ibm->x_bp[ n1e ]);
			dy12 = -(ibm->y_bp[ n2e ] - ibm->y_bp[ n1e ]);
			dz12 = -(ibm->z_bp[ n2e ] - ibm->z_bp[ n1e ]);


			dx13 = -(ibm->x_bp[ n1e ] - ibm->x_bp[ n3e ]);
			dy13 = -(ibm->y_bp[ n1e ] - ibm->y_bp[ n3e ]);
			dz13 = -(ibm->z_bp[ n1e ] - ibm->z_bp[ n3e ]);


			dx23 = -(ibm->x_bp[ n3e ] - ibm->x_bp[ n2e ]);
			dy23 = -(ibm->y_bp[ n3e ] - ibm->y_bp[ n2e ]);
			dz23 = -(ibm->z_bp[ n3e ] - ibm->z_bp[ n2e ]);


			nbelmt=ibm->c2c[elmt].c1;  // xiaolei add SEDI
			if (nbelmt>=0)  // xiaolei add 
			{

		            	if(ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)// || ibm->Rigidity[nbelmt])
				{
					ibm->F_flux_12[ elmt ] = 0.;
				}
				else
				{
				  	double u_face = 0.5 * ( ibm->Bvel[ elmt ].x + ibm->Bvel[ nbelmt ].x );
					double v_face = 0.5 * ( ibm->Bvel[ elmt ].y + ibm->Bvel[ nbelmt ].y );
					double w_face = 0.5 * ( ibm->Bvel[ elmt ].z + ibm->Bvel[ nbelmt ].z );
                                        double C_face = 0.5 * ( ibm->SCont[ elmt ] + ibm->SCont[ nbelmt ] );

					/*
				  	double u_face = 0.5 * ( u_node[n1e] + u_node[n2e]);
				  	double v_face = 0.5 * ( v_node[n1e] + v_node[n2e]);
				  	double w_face = 0.5 * ( w_node[n1e] + w_node[n2e]);
				  	double C_face = 0.5 * ( C_node[n1e] + C_node[n2e]);
					*/

					dr = sqrt( dx12 * dx12 + dy12 * dy12 );
                                        ds = dr;
					double nfx = dy12/dr;
					double nfy = -dx12/dr;

					double un_face  = nfx * u_face + nfy * v_face;
					ibm->F_flux_12[elmt]= un_face * ds * C_face;      
				}
			}


			nbelmt=ibm->c2c[elmt].c2;  // xiaolei add SEDI
			if (nbelmt>=0)  // xiaolei add 
			{

		            	if(ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)// || ibm->Rigidity[nbelmt])
				{
					ibm->F_flux_13[ elmt ] = 0.;
				}
				else
				{
				  	double u_face = 0.5 * ( ibm->Bvel[ elmt ].x + ibm->Bvel[ nbelmt ].x );
					double v_face = 0.5 * ( ibm->Bvel[ elmt ].y + ibm->Bvel[ nbelmt ].y );
					double w_face = 0.5 * ( ibm->Bvel[ elmt ].z + ibm->Bvel[ nbelmt ].z );
                                        double C_face = 0.5 * ( ibm->SCont[ elmt ] + ibm->SCont[ nbelmt ] );

					/*                         
				  	double u_face = 0.5 * ( u_node[n1e] + u_node[n3e]);
				  	double v_face = 0.5 * ( v_node[n1e] + v_node[n3e]);
				  	double w_face = 0.5 * ( w_node[n1e] + w_node[n3e]);
				  	double C_face = 0.5 * ( C_node[n1e] + C_node[n3e]);
					*/

					dr = sqrt( dx13 * dx13 + dy13 * dy13 );
                                        ds = dr;
					double nfx = dy13/dr;
					double nfy = -dx13/dr;

					double un_face  = nfx * u_face + nfy * v_face;

					ibm->F_flux_13[elmt]= un_face * ds * C_face;      
				}
			}


			nbelmt=ibm->c2c[elmt].c3;  // xiaolei add SEDI
			if (nbelmt>=0)  // xiaolei add 
			{

		            	if(ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)// || ibm->Rigidity[nbelmt])
				{
					ibm->F_flux_23[ elmt ] = 0.;
				}
				else
				{
				  	double u_face = 0.5 * ( ibm->Bvel[ elmt ].x + ibm->Bvel[ nbelmt ].x );
					double v_face = 0.5 * ( ibm->Bvel[ elmt ].y + ibm->Bvel[ nbelmt ].y );
					double w_face = 0.5 * ( ibm->Bvel[ elmt ].z + ibm->Bvel[ nbelmt ].z );
                                        double C_face = 0.5 * ( ibm->SCont[ elmt ] + ibm->SCont[ nbelmt ] );

					/*
				  	double u_face = 0.5 * ( u_node[n2e] + u_node[n3e]);
				  	double v_face = 0.5 * ( v_node[n2e] + v_node[n3e]);
				  	double w_face = 0.5 * ( w_node[n2e] + w_node[n3e]);
				  	double C_face = 0.5 * ( C_node[n2e] + C_node[n3e]);
					*/

					dr = sqrt( dx23 * dx23 + dy23 * dy23 );
                                        ds = dr;
					double nfx = dy23/dr;
					double nfy = -dx23/dr;

					double un_face  = nfx * u_face + nfy * v_face;

					ibm->F_flux_23[elmt]= un_face * ds * C_face;      
				}
			}




		}
	}

	return ( 0 );												//	ali completed on 3 nov. 2009				    
}



PetscErrorCode	outlet_sediment_flux_bend( UserCtx * user, IBMNodes * ibm )
{
	DM	 da  = user->da, 
		 fda  = user->fda;

	Cmpnts		 * Bvel;
	PetscReal	 * F_flux_12, 
				 * F_flux_13, 
				 * F_flux_23;
	PetscReal	 * SCont;
	int	   n_elmt  = ibm->n_elmt, 
					n_v  = ibm->n_v;
	PetscReal	   sb, 
					sc;
	IBMInfo  	 * ibminfo;
	IBMListNode  * current;
	int   elmt,nbelmt,	n1e,n2e,n3e;
	PetscReal	   riter;
	PetscReal	   nfx, 
					nfy, 
					nfz, 
					dx12, 
					dy12, 
					dz12, 
					dx13, 
					dy13, 
					dz13, 
					dx23, 
					dy23, 
					dz23, 
					dr, 
					ds;
       	PetscReal	   tmp;
	PetscReal	        	nfx12, 
					nfy12, 
					nfx13, 
					nfy13, 
					nfx23, 
					nfy23; 
	PetscReal                       y_outlet = 3.41,
                                        y_plus_x_outlet=29.11,                                            
                                        someX_minus_Y_outlet = 33.07459,                                            
                                        //someX_plus_Y_inlet = 100.05944;                                            
                                        someX_plus_Y_inlet = 90.16268;                                            
        PetscReal                       ydif1, ydif2, ydif3,xydif1,xydif2,xydif3;
        PetscReal                       sxydif1,sxydif2,sxydif3;
        PetscReal                       ssxydif1,ssxydif2,ssxydif3;
	PetscReal                       Ave_face12_u,
                                        Ave_face13_u,
                                        Ave_face23_u,
                                        Ave_face12_v, 
                                        Ave_face13_v, 
                                        Ave_face23_v,
                                        nor_vel_to_face12,
                                        nor_vel_to_face13,
                                        nor_vel_to_face23,
                                     	le12x,
                                        le12y,
                                        le13x,
                                        le13y,
                                        le23x,
                                        le23y,
                                        le_dot_n12,
                                        le_dot_n13,
                                        le_dot_n23,
                                        xc,yc,zc;
                

// computing fluxes through oulet faces which could be 1->2,3 or 2->3
// the outlet charachteristics is that the y is 3.41 there, later I must find a better characteristics for this, 18 Nov. 2009, ali
      
	for( elmt = 0; elmt < n_elmt; elmt++ )
	   {
	         	n1e  = ibm->nv1[ elmt ];
			n2e  = ibm->nv2[ elmt ];
			n3e  = ibm->nv3[ elmt ];
                        
                        xc	 = ibm->cent_x[ elmt ];
			yc	 = ibm->cent_y[ elmt ];
		//	zc       = ( ibm->z_bp[ n1e ] + ibm->z_bp[ n2e ] + ibm->z_bp[ n3e ] ) / 3.;
			zc	 = ibm->cent_z[ elmt ];
                        
                        ydif1=fabs(y_outlet-ibm->y_bp[n1e]);
			ydif2=fabs(y_outlet-ibm->y_bp[n2e]); 
	                ydif3=fabs(y_outlet-ibm->y_bp[n3e]);
                      
                        xydif1=fabs(y_plus_x_outlet-ibm->y_bp[n1e]-ibm->x_bp[n1e]);
                        xydif2=fabs(y_plus_x_outlet-ibm->y_bp[n2e]-ibm->x_bp[n2e]);
                        xydif3=fabs(y_plus_x_outlet-ibm->y_bp[n3e]-ibm->x_bp[n3e]);
		     
                        sxydif1=fabs(someX_minus_Y_outlet-(1.13087*ibm->x_bp[n1e]-ibm->y_bp[n1e]));
                        sxydif2=fabs(someX_minus_Y_outlet-(1.13087*ibm->x_bp[n2e]-ibm->y_bp[n2e]));
                        sxydif3=fabs(someX_minus_Y_outlet-(1.13087*ibm->x_bp[n3e]-ibm->y_bp[n3e]));

                        //ssxydif1=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n1e]+ibm->y_bp[n1e]));
                        //ssxydif2=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n2e]+ibm->y_bp[n2e]));
                        //ssxydif3=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n3e]+ibm->y_bp[n3e]));
                        ssxydif1=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n1e]+ibm->y_bp[n1e]));
                        ssxydif2=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n2e]+ibm->y_bp[n2e]));
                        ssxydif3=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n3e]+ibm->y_bp[n3e]));
//    Bend 90 degree                                       
	     // if((ydif1<1.e-6 && ydif2<1.e-6)||(ydif1<1.e-6 && ydif3<1.e-6)||(ydif2<1.e-6 && ydif3<1.e-6)) 


 //   Bend 135 degree
	     // if((xydif1<1.e-6 && xydif2<1.e-6)||(xydif1<1.e-6 && xydif3<1.e-6)||(xydif2<1.e-6 && xydif3<1.e-6))  
 

//   OSL
	      if((sxydif1<1.e-4 && sxydif2<1.e-4)||(sxydif1<1.e-4 && sxydif3<1.e-4)||(sxydif2<1.e-4 && sxydif3<1.e-4)) 
               // if((ssxydif1<1.e-4 && ssxydif2<1.e-4)||(ssxydif1<1.e-4 && ssxydif3<1.e-4)||(ssxydif2<1.e-4 && sxydif3<1.e-4))  
	      {


		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt])
		{
			ibm->F_flux_12[ elmt ] = 0.;
			ibm->F_flux_13[ elmt ] = 0.;
			ibm->F_flux_23[ elmt ] = 0.;
		}
		else
		{
		        dx12 = ibm->x_bp[ n2e ] - ibm->x_bp[ n1e ];
			dy12 = ibm->y_bp[ n2e ] - ibm->y_bp[ n1e ];
			dz12 = ibm->z_bp[ n2e ] - ibm->z_bp[ n1e ];

			dx13 = ibm->x_bp[ n3e ] - ibm->x_bp[ n1e ];
			dy13 = ibm->y_bp[ n3e ] - ibm->y_bp[ n1e ];
			dz13 = ibm->z_bp[ n3e ] - ibm->z_bp[ n1e ];

			dx23 = ibm->x_bp[ n3e ] - ibm->x_bp[ n2e ];
			dy23 = ibm->y_bp[ n3e ] - ibm->y_bp[ n2e ];
			dz23 = ibm->z_bp[ n3e ] - ibm->z_bp[ n2e ];

// finding neighbor cells and related inflow and outflow sediment flux for current elmt 
			// for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ )  // xiaolei deactivate 
			{
				//					1---->2 edge flux
				//
				/* // xiaolei deactivate 
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c1; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
	      	                        if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])
      					{

					       	Ave_face12_u = ibm->Bvel[ elmt ].x;

						Ave_face12_v = ibm->Bvel[ elmt ].y;

						nfx12 = dy12;
						nfy12 = -dx12;
						//ds    = sqrt( dx12 * dx12 + dy12 * dy12 + dz12 * dz12 );
						dr    = sqrt( dx12 * dx12 + dy12 * dy12 );
                                                ds = dr;
						nfx12 /= dr;
						nfy12 /= dr;

						nor_vel_to_face12  = nfx12 * Ave_face12_u + nfy12 * Ave_face12_v;

						le12x = 0.5 * ( ibm->x_bp[ n2e ] + ibm->x_bp[ n1e ] ) - ibm->cent_x[ elmt ];
						le12y = 0.5 * ( ibm->y_bp[ n2e ] + ibm->y_bp[ n1e ] ) - ibm->cent_y[ elmt ];

						le_dot_n12 = nfx12 * le12x + nfy12 * le12y;


						if( le_dot_n12 > 0. )	//  the current cell "elmt" is on the L side or U/S of neighbor cell "nbelmt"
                                                {
							if( nor_vel_to_face12 > 0. )	//the vel is in positive direction    
							{
								tmp = ibm->Bvel[ elmt ].x * nfx12 + ibm->Bvel[ elmt ].y * nfy12;
								ibm->F_flux_12[ elmt ] = -tmp * ds * ibm->SCont[ elmt ];
              						}
							else	//the velocity is in negative direction					    
							{
								tmp = ibm->Bvel[ nbelmt ].x * nfx12 + ibm->Bvel[ nbelmt ].y * nfy12;
								ibm->F_flux_12[ elmt ]
								= -tmp * ds * ibm->SCont[ nbelmt ];
							}
						}
						else    	// the current cell "elmt" is on the R side or D/S of neighbor cell "nbelmt"		
						{
							if( nor_vel_to_face12 > 0. )	//the vel is in positive direction    
                                                      	{
								tmp = ibm->Bvel[ nbelmt ].x * nfx12 + ibm->Bvel[ nbelmt ].y * nfy12;
								ibm->F_flux_12[ elmt ] = tmp * ds * ibm->SCont[ nbelmt ];
							}
							else	//the velocity is in negative direction		
                                                        {
								tmp = ibm->Bvel[ elmt ].x * nfx12 + ibm->Bvel[ elmt ].y * nfy12;
								ibm->F_flux_12[ elmt ] = tmp * ds * ibm->SCont[ elmt ];
							}
						}
                                             
					}
					else
					{		
					}
				}

				//					1---->3 edge flux 

				/*
				if( nbelmt != elmt   // xiaolei deactivate 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c2; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])
					{

                                                Ave_face13_u = ibm->Bvel[ elmt ].x;

						Ave_face13_v = ibm->Bvel[ elmt ].y;

						nfx13 = dy13;
						nfy13 = -dx13;
						//ds    = sqrt( dx13 * dx13 + dy13 * dy13 + dz13 * dz13 );
						dr    = sqrt( dx13 * dx13 + dy13 * dy13 );
                                                ds = dr;
						nfx13 /= dr;
						nfy13 /= dr;

						nor_vel_to_face13  = nfx13 * Ave_face13_u + nfy13 * Ave_face13_v;

						le13x = 0.5 * ( ibm->x_bp[ n3e ] + ibm->x_bp[ n1e ] ) - ibm->cent_x[ elmt ];
						le13y = 0.5 * ( ibm->y_bp[ n3e ] + ibm->y_bp[ n1e ] ) - ibm->cent_y[ elmt ];

						le_dot_n13 = nfx13 * le13x + nfy13 * le13y;


						if( le_dot_n13 > 0. )	// the current cell "elmt" is on the L side or U/S of neighbor cell "nbelmt"
                                                {
							if( nor_vel_to_face13 > 0. )	//the vel is in positive direction    
							{
								tmp = ibm->Bvel[ elmt ].x * nfx13 + ibm->Bvel[ elmt ].y * nfy13;
								ibm->F_flux_13[ elmt ] = -tmp * ds * ibm->SCont[ elmt ];
							}
							else	//the velocity is in negative direction		
							{
								tmp = ibm->Bvel[ nbelmt ].x * nfx13 + ibm->Bvel[ nbelmt ].y * nfy13;
								ibm->F_flux_13[ elmt ]
								= -tmp * ds * ibm->SCont[ nbelmt ];
							}
						}
						else	// the current cell "elmt" is on the R side or D/S of neighbor cell  "nbelmt"
                                                {
							if( nor_vel_to_face13 > 0. )	//the vel is in positive direction    
                                                        {
								tmp = ibm->Bvel[ nbelmt ].x * nfx13 + ibm->Bvel[ nbelmt ].y * nfy13;
								ibm->F_flux_13[ elmt ] = tmp * ds * ibm->SCont[ nbelmt ];
							}
							else	//the velocity is in negative direction	
                                 			{
								tmp = ibm->Bvel[ elmt ].x * nfx13 + ibm->Bvel[ elmt ].y * nfy13;
								ibm->F_flux_13[ elmt ] = tmp * ds * ibm->SCont[ elmt ];
							}
						}

					}
					else
               				{
					}
				}

				//					2---->3 edge  flux
				/*  // xiaolei deactivate 
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c3; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
	                            	if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])
					{
                                              	Ave_face23_u = ibm->Bvel[ elmt ].x;

						Ave_face23_v = ibm->Bvel[ elmt ].y;

						nfx23 = dy23;
						nfy23 = -dx23;
						//ds    = sqrt( dx23 * dx23 + dy23 * dy23 + dz23 * dz23 );
						dr    = sqrt( dx23 * dx23 + dy23 * dy23 );
                                                ds = dr;
						nfx23 /= dr;
						nfy23 /= dr;

						nor_vel_to_face23  = nfx23 * Ave_face23_u + nfy23 * Ave_face23_v;

						le23x = 0.5 * ( ibm->x_bp[ n3e ] + ibm->x_bp[ n2e ] ) - ibm->cent_x[ elmt ];
						le23y = 0.5 * ( ibm->y_bp[ n3e ] + ibm->y_bp[ n2e ] ) - ibm->cent_y[ elmt ];

						le_dot_n23 = nfx23 * le23x + nfy23 * le23y;


						if( le_dot_n23 > 0. )	// the current cell "elmt" is on the L side or U/S of neighbor cell "nbelmt"
                                                {
							if( nor_vel_to_face23 > 0. )	//the vel is in positive direction    
							{
								tmp = ibm->Bvel[ elmt ].x * nfx23 + ibm->Bvel[ elmt ].y * nfy23;
								ibm->F_flux_23[ elmt ] = -tmp * ds * ibm->SCont[ elmt ];
							}
							else	//the velocity is in negative direction				
							{
								tmp = ibm->Bvel[ nbelmt ].x * nfx23 + ibm->Bvel[ nbelmt ].y * nfy23;
								ibm->F_flux_23[ elmt ]
								= -tmp * ds * ibm->SCont[ nbelmt ];
							}
						}
						else	// the current cell "elmt" is on the R side or D/S of neighbor cell "nbelmt"
                                                {
							if( nor_vel_to_face23 > 0. )	//the vel is in positive direction    
							{
								tmp = ibm->Bvel[ nbelmt ].x * nfx23 + ibm->Bvel[ nbelmt ].y * nfy23;
								ibm->F_flux_23[ elmt ]
								= tmp * ds * ibm->SCont[ nbelmt ];
							}
							else	//the velocity is in negative direction					    
							{
								tmp = ibm->Bvel[ elmt ].x * nfx23 + ibm->Bvel[ elmt ].y * nfy23;
								ibm->F_flux_23[ elmt ] = tmp * ds * ibm->SCont[ elmt ];
							}
						}
						
					}
					else
					{
					}
				}
			} // for loop, nbelmt cycling
		}  // if,to check if elmt is on bed  
	 } // if,to check the cell face if at outlet
   }  // for loop, elmt cycling

	return ( 0 );		//	ali completed on 19 nov. 2009				    
}


PetscErrorCode	BC_elevation_1( UserCtx * user, IBMNodes * ibm )
{
	DM	 da  = user->da, 
		 fda  = user->fda;

	int	   n_elmt  = ibm->n_elmt, n_v  = ibm->n_v;
	IBMInfo  	 * ibminfo;
	int   elmt,nbelmt,	n1e,n2e,n3e;
	PetscReal                       x_outlet = 10.0; //total osl//25.; // Jhoke Crossvane Rockvane 50.;//49.71;
	PetscReal                       x_inlet = 0.0; //total osl//25.; // Jhoke Crossvane Rockvane 50.;//49.71; // xiaolei add
        PetscReal                       xdif1,xdif2,xdif3;

     
	for( elmt = 0; elmt < n_elmt; elmt++ ) {
         	n1e  = ibm->nv1[ elmt ];
		n2e  = ibm->nv2[ elmt ];
		n3e  = ibm->nv3[ elmt ];

		// outlet		                                                                   
                xdif1=fabs(x_outlet-ibm->x_bp[n1e]);
		xdif2=fabs(x_outlet-ibm->x_bp[n2e]); 
	        xdif3=fabs(x_outlet-ibm->x_bp[n3e]);

	      	if((xdif1<1.e-4 && xdif2<1.e-4)||(xdif1<1.e-4 && xdif3<1.e-4)||(xdif2<1.e-4 && xdif3<1.e-4))  {

			int c1_=ibm->c2c[elmt].c1; 
			int c2_=ibm->c2c[elmt].c2; 
			int c3_=ibm->c2c[elmt].c3; 

			double fac1=0.0, fac2=0.0, fac3=0.0;

 	                if( c1_>0 && ibm->nf_z[c1_] > 1.e-6 && ibm->elmt_depth[c1_] < 0.3) fac1=1.0;
      	                if( c2_>0 && ibm->nf_z[c2_] > 1.e-6 && ibm->elmt_depth[c2_] < 0.3) fac2=1.0;
      	                if( c3_>0 && ibm->nf_z[c3_] > 1.e-6 && ibm->elmt_depth[c3_] < 0.3) fac3=1.0;

			double fac=fac1+fac2+fac3+1.e-20;

			fac1=fac1/fac;
			fac2=fac2/fac;
			fac3=fac3/fac;

			ibm->cent_z[elmt]=ibm->cent_z[c1_]*fac1+ibm->cent_z[c2_]*fac2+ibm->cent_z[c3_]*fac3;  

			/*
 	                if( c1_>0 && (ibm->nf_z[c1_] < 1.e-6 || ibm->elmt_depth[c1_] > 0.3) ) {
				ibm->z_bp[n1e]=ibm->cent_z[elmt];
				ibm->z_bp[n2e]=ibm->cent_z[elmt];
			}

 	                if( c2_>0 && (ibm->nf_z[c2_] < 1.e-6 || ibm->elmt_depth[c2_] > 0.3) ) {
				ibm->z_bp[n1e]=ibm->cent_z[elmt];
				ibm->z_bp[n3e]=ibm->cent_z[elmt];
			}
 	                if( c3_>0 && (ibm->nf_z[c3_] < 1.e-6 || ibm->elmt_depth[c3_] > 0.3) ) {
				ibm->z_bp[n2e]=ibm->cent_z[elmt];
				ibm->z_bp[n3e]=ibm->cent_z[elmt];
			}

			*/
		}


		// inlet                                                                
                xdif1=fabs(x_inlet-ibm->x_bp[n1e]);
		xdif2=fabs(x_inlet-ibm->x_bp[n2e]); 
	        xdif3=fabs(x_inlet-ibm->x_bp[n3e]);

	      	if((xdif1<1.e-4 && xdif2<1.e-4)||(xdif1<1.e-4 && xdif3<1.e-4)||(xdif2<1.e-4 && xdif3<1.e-4))  {

			int c1_=ibm->c2c[elmt].c1; 
			int c2_=ibm->c2c[elmt].c2; 
			int c3_=ibm->c2c[elmt].c3; 

			double fac1=0.0, fac2=0.0, fac3=0.0;

 	                if( c1_>0 && ibm->nf_z[c1_] > 1.e-6 && ibm->elmt_depth[c1_] < 0.3) fac1=1.0;
      	                if( c2_>0 && ibm->nf_z[c2_] > 1.e-6 && ibm->elmt_depth[c2_] < 0.3) fac2=1.0;
      	                if( c3_>0 && ibm->nf_z[c3_] > 1.e-6 && ibm->elmt_depth[c3_] < 0.3) fac3=1.0;

			double fac=fac1+fac2+fac3+1.e-20;

			fac1=fac1/fac;
			fac2=fac2/fac;
			fac3=fac3/fac;

			ibm->cent_z[elmt]=ibm->cent_z[c1_]*fac1+ibm->cent_z[c2_]*fac2+ibm->cent_z[c3_]*fac3;  

			/*
 	                if( c1_>0 && (ibm->nf_z[c1_] < 1.e-6 || ibm->elmt_depth[c1_] > 0.3) ) {
				ibm->z_bp[n1e]=ibm->cent_z[elmt];
				ibm->z_bp[n2e]=ibm->cent_z[elmt];
			}

 	                if( c2_>0 && (ibm->nf_z[c2_] < 1.e-6 || ibm->elmt_depth[c2_] > 0.3) ) {
				ibm->z_bp[n1e]=ibm->cent_z[elmt];
				ibm->z_bp[n3e]=ibm->cent_z[elmt];
			}
 	                if( c3_>0 && (ibm->nf_z[c3_] < 1.e-6 || ibm->elmt_depth[c3_] > 0.3) ) {
				ibm->z_bp[n2e]=ibm->cent_z[elmt];
				ibm->z_bp[n3e]=ibm->cent_z[elmt];
			}
			*/


		}

	} 


	return 0; 
}


PetscErrorCode	BC_delz( UserCtx * user, IBMNodes * ibm )
{

	int	n_elmt  = ibm->n_elmt;
	int   	elmt;
     
	PetscReal *dz;
      	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(dz));
	for( elmt = 0; elmt < n_elmt; elmt++ ) {
		dz[elmt] = ibm->Delz[ elmt ];
	}

	for( elmt = 0; elmt < n_elmt; elmt++ ) {

		if (!(ibm->Rigidity[elmt])) {

			int rigid = 0;

			int ii, nii;
			for(ii=0;ii<100;ii++) {
				nii=ibm->c2ac[elmt].c[ii];	// xiaolei add
				if (nii>=0) rigid += ibm->Rigidity[nii];
			}


			double Delz_sum=0.0;
			double count=0.0;
		
			if(rigid) {
				for(ii=0;ii<100;ii++) {
					nii=ibm->c2ac[elmt].c[ii];	// xiaolei add

					if (nii>=0 && !(ibm->Rigidity[nii])) {
						Delz_sum+=dz[ nii ];
						count+=1.0;
					}
				}
				ibm->Delz[ elmt ]=Delz_sum/(count+1.e-9);
			}	

		}

	}

	PetscFree(dz);

	return 0; 
}


PetscErrorCode	BC_elevation( UserCtx * user, IBMNodes * ibm )
{

	int	n_elmt  = ibm->n_elmt;
	int   	elmt, n1e, n2e, n3e;
     
	PetscReal *centz;
      	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(centz));
	for( elmt = 0; elmt < n_elmt; elmt++ ) {
		centz[elmt] = ibm->cent_z[ elmt ];
	}

	for( elmt = 0; elmt < n_elmt; elmt++ ) {

		if (!(ibm->Rigidity[elmt])) {
	         	n1e  = ibm->nv1[ elmt ];
			n2e  = ibm->nv2[ elmt ];
			n3e  = ibm->nv3[ elmt ];


			int rigid = 0;

			int ii, nii;
			for(ii=0;ii<100;ii++) {
				nii=ibm->c2ac[elmt].c[ii];	// xiaolei add
				if (nii>=0) rigid += ibm->Rigidity[nii];
			}


			double z_sum=0.0;
			double count=0.0;
		
			if(rigid) {
				for(ii=0;ii<100;ii++) {
					nii=ibm->c2ac[elmt].c[ii];	// xiaolei add

					if (nii>=0 && !(ibm->Rigidity[nii])) {
						z_sum+=centz[ nii ];
						count+=1.0;
					}
				}
				ibm->cent_z[ elmt ]=z_sum/(count+1.e-9);
				ibm->z_bp[ n1e ]=ibm->cent_z[ elmt ];
				ibm->z_bp[ n2e ]=ibm->cent_z[ elmt ];
				ibm->z_bp[ n3e ]=ibm->cent_z[ elmt ];
			}	

		}

	}

	PetscFree(centz);

	return 0; 
}



PetscErrorCode	BC_elevation_2( UserCtx * user, IBMNodes * ibm )
{
	DM	 da  = user->da, 
		 fda  = user->fda;

	int	   n_elmt  = ibm->n_elmt, n_v  = ibm->n_v;
	IBMInfo  	 * ibminfo;
	int   elmt,nbelmt,	n1e,n2e,n3e;
	PetscReal                       x_outlet = 10.0; //total osl//25.; // Jhoke Crossvane Rockvane 50.;//49.71;
	PetscReal                       x_inlet = 0.0; //total osl//25.; // Jhoke Crossvane Rockvane 50.;//49.71; // xiaolei add
        PetscReal                       xdif1,xdif2,xdif3;

     
	for( elmt = 0; elmt < n_elmt; elmt++ ) {

		int c1_=ibm->c2c[elmt].c1; 
		int c2_=ibm->c2c[elmt].c2; 
		int c3_=ibm->c2c[elmt].c3; 

                int rigid = ibm->Rigidity[c1_]+ibm->Rigidity[c2_]+ibm->Rigidity[c3_];

		if (rigid) {
			ibm->Delz[ elmt ] = 0.0;
		}

	} 


	return 0; 
}




PetscErrorCode	outlet_sediment_flux_contra( UserCtx * user, IBMNodes * ibm )
{
	DM	 da  = user->da, 
		 fda  = user->fda;

	Cmpnts		 * Bvel;
	PetscReal	 * F_flux_12, 
				 * F_flux_13, 
				 * F_flux_23;
	PetscReal	 * SCont;
	int	   n_elmt  = ibm->n_elmt, 
					n_v  = ibm->n_v;
	PetscReal	   sb, 
					sc;
	IBMInfo  	 * ibminfo;
	IBMListNode  * current;
	int   elmt,nbelmt,	n1e,n2e,n3e;
	PetscReal	   riter;
	PetscReal	   nfx, 
					nfy, 
					nfz, 
					dx12, 
					dy12, 
					dz12, 
					dx13, 
					dy13, 
					dz13, 
					dx23, 
					dy23, 
					dz23, 
					dr, 
					ds;
       	PetscReal	   tmp;
	PetscReal	        	nfx12, 
					nfy12, 
					nfx13, 
					nfy13, 
					nfx23, 
					nfy23; 
	PetscReal                       x_outlet = 10.0; //total osl//25.; // Jhoke Crossvane Rockvane 50.;//49.71;
	PetscReal                       x_inlet = 0.0; //total osl//25.; // Jhoke Crossvane Rockvane 50.;//49.71; // xiaolei add
        PetscReal                       xdif1,xdif2,xdif3;
	PetscReal                       Ave_face12_u,
                                        Ave_face13_u,
                                        Ave_face23_u,
                                        Ave_face12_v, 
                                        Ave_face13_v, 
                                        Ave_face23_v,
                                        nor_vel_to_face12,
                                        nor_vel_to_face13,
                                        nor_vel_to_face23,
                                     	le12x,
                                        le12y,
                                        le13x,
                                        le13y,
                                        le23x,
                                        le23y,
                                        le_dot_n12,
                                        le_dot_n13,
                                        le_dot_n23;

// computing fluxes through oulet faces which could be 1->2,3 or 2->3
// the outlet charachteristics is that the y is 3.41 there, later I must find a better characteristics for this, 18 Nov. 2009, ali
     
	// outlet neumann  
	for( elmt = 0; elmt < n_elmt; elmt++ )
	   {
	         	n1e  = ibm->nv1[ elmt ];
			n2e  = ibm->nv2[ elmt ];
			n3e  = ibm->nv3[ elmt ];
                                                                   
                        xdif1=fabs(x_outlet-ibm->x_bp[n1e]);
			xdif2=fabs(x_outlet-ibm->x_bp[n2e]); 
	                xdif3=fabs(x_outlet-ibm->x_bp[n3e]);

	      if((xdif1<1.e-4 && xdif2<1.e-4)||(xdif1<1.e-4 && xdif3<1.e-4)||(xdif2<1.e-4 && xdif3<1.e-4))  
	      {


		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3)
		{
			ibm->F_flux_12[ elmt ] = 0.;
			ibm->F_flux_13[ elmt ] = 0.;
			ibm->F_flux_23[ elmt ] = 0.;
		}
		else
		{
		        dx12 = ibm->x_bp[ n2e ] - ibm->x_bp[ n1e ];
			dy12 = ibm->y_bp[ n2e ] - ibm->y_bp[ n1e ];
			dz12 = ibm->z_bp[ n2e ] - ibm->z_bp[ n1e ];

			dx13 = ibm->x_bp[ n3e ] - ibm->x_bp[ n1e ];
			dy13 = ibm->y_bp[ n3e ] - ibm->y_bp[ n1e ];
			dz13 = ibm->z_bp[ n3e ] - ibm->z_bp[ n1e ];

			dx23 = ibm->x_bp[ n3e ] - ibm->x_bp[ n2e ];
			dy23 = ibm->y_bp[ n3e ] - ibm->y_bp[ n2e ];
			dz23 = ibm->z_bp[ n3e ] - ibm->z_bp[ n2e ];

// finding neighbor cells and related inflow and outflow sediment flux for current elmt 
			// for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ ) // xiaolei deactivate 
			{
				//					1---->2 edge flux
				/* // xiaolei deactivate 
				if( nbelmt != elmt
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
				   )
				*/

				nbelmt=ibm->c2c[elmt].c1; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
	      	                        if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)
      					{

						ibm->F_flux_12[ elmt ] = -ibm->F_flux_13[ elmt ]-ibm->F_flux_23[ elmt ];  


						// xiaolei deactivate 
						/*
					       	Ave_face12_u = ibm->Bvel[ elmt ].x;

						Ave_face12_v = ibm->Bvel[ elmt ].y;

						nfx12 = dy12;
						nfy12 = -dx12;
						//ds    = sqrt( dx12 * dx12 + dy12 * dy12 + dz12 * dz12 );
						dr    = sqrt( dx12 * dx12 + dy12 * dy12 );
                                                ds = dr;
						nfx12 /= dr;
						nfy12 /= dr;

						nor_vel_to_face12  = nfx12 * Ave_face12_u + nfy12 * Ave_face12_v;

						le12x = 0.5 * ( ibm->x_bp[ n2e ] + ibm->x_bp[ n1e ] ) - ibm->cent_x[ elmt ];
						le12y = 0.5 * ( ibm->y_bp[ n2e ] + ibm->y_bp[ n1e ] ) - ibm->cent_y[ elmt ];

						le_dot_n12 = nfx12 * le12x + nfy12 * le12y;


						if( le_dot_n12 > 0. )	//  the current cell "elmt" is on the L side or U/S of neighbor cell "nbelmt"
                                                {
							if( nor_vel_to_face12 > 0. )	//the vel is in positive direction    
							{
								tmp = ibm->Bvel[ elmt ].x * nfx12 + ibm->Bvel[ elmt ].y * nfy12;
								ibm->F_flux_12[ elmt ] = -tmp * ds * ibm->SCont[ elmt ];
              						}
							else	//the velocity is in negative direction					    
							{
								tmp = ibm->Bvel[ nbelmt ].x * nfx12 + ibm->Bvel[ nbelmt ].y * nfy12;
								ibm->F_flux_12[ elmt ]
								= -tmp * ds * ibm->SCont[ nbelmt ];
							}
						}
						else    	// the current cell "elmt" is on the R side or D/S of neighbor cell "nbelmt"		
						{
							if( nor_vel_to_face12 > 0. )	//the vel is in positive direction    
                                                      	{
								tmp = ibm->Bvel[ nbelmt ].x * nfx12 + ibm->Bvel[ nbelmt ].y * nfy12;
								ibm->F_flux_12[ elmt ] = tmp * ds * ibm->SCont[ nbelmt ];
							}
							else	//the velocity is in negative direction		
                                                        {
								tmp = ibm->Bvel[ elmt ].x * nfx12 + ibm->Bvel[ elmt ].y * nfy12;
								ibm->F_flux_12[ elmt ] = tmp * ds * ibm->SCont[ elmt ];
							}
						}
                                            	*/ 
					}
					else
					{		
					}
				}

				//					1---->3 edge flux 

				/*   // xiaolei deactivate 
				if( nbelmt != elmt
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c2; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)
					{

						ibm->F_flux_13[ elmt ] = -ibm->F_flux_12[ elmt ]-ibm->F_flux_23[ elmt ];  

						// xiaolei deactivate
						/*
                                                Ave_face13_u = ibm->Bvel[ elmt ].x;

						Ave_face13_v = ibm->Bvel[ elmt ].y;

						nfx13 = dy13;
						nfy13 = -dx13;
						//ds    = sqrt( dx13 * dx13 + dy13 * dy13 + dz13 * dz13 );
						dr    = sqrt( dx13 * dx13 + dy13 * dy13 );
                                                ds = dr;
						nfx13 /= dr;
						nfy13 /= dr;

						nor_vel_to_face13  = nfx13 * Ave_face13_u + nfy13 * Ave_face13_v;

						le13x = 0.5 * ( ibm->x_bp[ n3e ] + ibm->x_bp[ n1e ] ) - ibm->cent_x[ elmt ];
						le13y = 0.5 * ( ibm->y_bp[ n3e ] + ibm->y_bp[ n1e ] ) - ibm->cent_y[ elmt ];

						le_dot_n13 = nfx13 * le13x + nfy13 * le13y;


						if( le_dot_n13 > 0. )	// the current cell "elmt" is on the L side or U/S of neighbor cell "nbelmt"
                                                {
							if( nor_vel_to_face13 > 0. )	//the vel is in positive direction    
							{
								tmp = ibm->Bvel[ elmt ].x * nfx13 + ibm->Bvel[ elmt ].y * nfy13;
								ibm->F_flux_13[ elmt ] = -tmp * ds * ibm->SCont[ elmt ];
							}
							else	//the velocity is in negative direction		
							{
								tmp = ibm->Bvel[ nbelmt ].x * nfx13 + ibm->Bvel[ nbelmt ].y * nfy13;
								ibm->F_flux_13[ elmt ]
								= -tmp * ds * ibm->SCont[ nbelmt ];
							}
						}
						else	// the current cell "elmt" is on the R side or D/S of neighbor cell  "nbelmt"
                                                {
							if( nor_vel_to_face13 > 0. )	//the vel is in positive direction    
                                                        {
								tmp = ibm->Bvel[ nbelmt ].x * nfx13 + ibm->Bvel[ nbelmt ].y * nfy13;
								ibm->F_flux_13[ elmt ] = tmp * ds * ibm->SCont[ nbelmt ];
							}
							else	//the velocity is in negative direction	
                                 			{
								tmp = ibm->Bvel[ elmt ].x * nfx13 + ibm->Bvel[ elmt ].y * nfy13;
								ibm->F_flux_13[ elmt ] = tmp * ds * ibm->SCont[ elmt ];
							}
						}
						*/
					}
					else
               				{
					}
				}

				//					2---->3 edge  flux
				/*  xiaolei deactivate 
				if( nbelmt != elmt
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c3; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
	                            	if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)
					{


						ibm->F_flux_23[ elmt ] = -ibm->F_flux_13[ elmt ]-ibm->F_flux_12[ elmt ];  

						/*
                                              	Ave_face23_u = ibm->Bvel[ elmt ].x;

						Ave_face23_v = ibm->Bvel[ elmt ].y;

						nfx23 = dy23;
						nfy23 = -dx23;
						//ds    = sqrt( dx23 * dx23 + dy23 * dy23 + dz23 * dz23 );
						dr    = sqrt( dx23 * dx23 + dy23 * dy23 );
                                                ds = dr;
						nfx23 /= dr;
						nfy23 /= dr;

						nor_vel_to_face23  = nfx23 * Ave_face23_u + nfy23 * Ave_face23_v;

						le23x = 0.5 * ( ibm->x_bp[ n3e ] + ibm->x_bp[ n2e ] ) - ibm->cent_x[ elmt ];
						le23y = 0.5 * ( ibm->y_bp[ n3e ] + ibm->y_bp[ n2e ] ) - ibm->cent_y[ elmt ];

						le_dot_n23 = nfx23 * le23x + nfy23 * le23y;


						if( le_dot_n23 > 0. )	// the current cell "elmt" is on the L side or U/S of neighbor cell "nbelmt"
                                                {
							if( nor_vel_to_face23 > 0. )	//the vel is in positive direction    
							{
								tmp = ibm->Bvel[ elmt ].x * nfx23 + ibm->Bvel[ elmt ].y * nfy23;
								ibm->F_flux_23[ elmt ] = -tmp * ds * ibm->SCont[ elmt ];
							}
							else	//the velocity is in negative direction				
							{
								tmp = ibm->Bvel[ nbelmt ].x * nfx23 + ibm->Bvel[ nbelmt ].y * nfy23;
								ibm->F_flux_23[ elmt ]
								= -tmp * ds * ibm->SCont[ nbelmt ];
							}
						}
						else	// the current cell "elmt" is on the R side or D/S of neighbor cell "nbelmt"
                                                {
							if( nor_vel_to_face23 > 0. )	//the vel is in positive direction    
							{
								tmp = ibm->Bvel[ nbelmt ].x * nfx23 + ibm->Bvel[ nbelmt ].y * nfy23;
								ibm->F_flux_23[ elmt ]
								= tmp * ds * ibm->SCont[ nbelmt ];
							}
							else	//the velocity is in negative direction					    
							{
								tmp = ibm->Bvel[ elmt ].x * nfx23 + ibm->Bvel[ elmt ].y * nfy23;
								ibm->F_flux_23[ elmt ] = tmp * ds * ibm->SCont[ elmt ];
							}
						}
						*/						
					}
					else
					{
					}
				}
			} // for loop, nbelmt cycling
		}  // if,to check if elmt is on bed  
	 } // if,to check the x_bp value if at outlet
   }  // for loop, elmt cycling



 
	// inlet neumann  // xiaolei
for( elmt = 0; elmt < n_elmt; elmt++ ) {
	n1e  = ibm->nv1[ elmt ];
	n2e  = ibm->nv2[ elmt ];
	n3e  = ibm->nv3[ elmt ];
                                                                   
       	xdif1=fabs(x_inlet-ibm->x_bp[n1e]);
	xdif2=fabs(x_inlet-ibm->x_bp[n2e]); 
	xdif3=fabs(x_inlet-ibm->x_bp[n3e]);

	if((xdif1<1.e-4 && xdif2<1.e-4)||(xdif1<1.e-4 && xdif3<1.e-4)||(xdif2<1.e-4 && xdif3<1.e-4))   {


	if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3) {
		ibm->F_flux_12[ elmt ] = 0.;
		ibm->F_flux_13[ elmt ] = 0.;
		ibm->F_flux_23[ elmt ] = 0.;
	}
	else
	{
	       	dx12 = ibm->x_bp[ n2e ] - ibm->x_bp[ n1e ];
		dy12 = ibm->y_bp[ n2e ] - ibm->y_bp[ n1e ];
		dz12 = ibm->z_bp[ n2e ] - ibm->z_bp[ n1e ];

		dx13 = ibm->x_bp[ n3e ] - ibm->x_bp[ n1e ];
		dy13 = ibm->y_bp[ n3e ] - ibm->y_bp[ n1e ];
		dz13 = ibm->z_bp[ n3e ] - ibm->z_bp[ n1e ];

		dx23 = ibm->x_bp[ n3e ] - ibm->x_bp[ n2e ];
		dy23 = ibm->y_bp[ n3e ] - ibm->y_bp[ n2e ];
		dz23 = ibm->z_bp[ n3e ] - ibm->z_bp[ n2e ];

		nbelmt=ibm->c2c[elmt].c1; // xiaolei add
		if(nbelmt>0) {
	            	if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3) {

				ibm->F_flux_12[ elmt ] = -ibm->F_flux_13[ elmt ]-ibm->F_flux_23[ elmt ];  

			}
			else
			{		
			}
		}

		nbelmt=ibm->c2c[elmt].c2; // xiaolei add
		if(nbelmt>0)	{
			if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3){

				ibm->F_flux_13[ elmt ] = -ibm->F_flux_12[ elmt ]-ibm->F_flux_23[ elmt ];  

			}
			else
               		{
			}
		}

		nbelmt=ibm->c2c[elmt].c3; // xiaolei add
		if(nbelmt>0) {
	              	if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3){

				ibm->F_flux_23[ elmt ] = -ibm->F_flux_13[ elmt ]-ibm->F_flux_12[ elmt ];  

			}
			else
			{
			}
		}
	}  // if,to check if elmt is on bed  
	} // if,to check the x_bp value if at outlet
}  // for

	return ( 0 );		//	ali completed on 4 Dec. 2009				    
}



PetscErrorCode	inlet_sediment_flux_bend( UserCtx * user, IBMNodes * ibm, double Q_total)
{
	DM	 da  = user->da, 
		 fda  = user->fda;

	int	   n_elmt  = ibm->n_elmt, 
					n_v  = ibm->n_v;
	IBMInfo  	 * ibminfo;
	IBMListNode  * current;
	int   elmt,nbelmt,	n1e,n2e,n3e;
	PetscReal	   riter;
	PetscReal	   dx12, 
					dy12, 
					dz12, 
					dx13, 
					dy13, 
					dz13, 
					dx23, 
					dy23, 
					dz23, 
					ds;
        PetscReal     channle_width = 7.7118;   
	PetscReal                       //y_outlet = 3.41,
                                        //y_plus_x_outlet=29.11,                                            
                                        //someX_minus_Y_outlet = 33.07459,                                            
                                        //someX_plus_Y_inlet = 100.05944;                                            
                                        someX_plus_Y_inlet = 90.16268;                                            
        PetscReal                       ssxydif1,ssxydif2,ssxydif3;
        PetscReal                       xc,yc,zc;
                

// prescribing inlet fluxes at inlet faces which could be 1->2,3 or 2->3
// the outlet charachteristics is that the y is 3.41 there, later I must find a better characteristics for this, 18 Nov. 2009, ali
      
	for( elmt = 0; elmt < n_elmt; elmt++ )
	   {
	         	n1e  = ibm->nv1[ elmt ];
			n2e  = ibm->nv2[ elmt ];
			n3e  = ibm->nv3[ elmt ];
                        
                        xc	 = ibm->cent_x[ elmt ];
			yc	 = ibm->cent_y[ elmt ];
			zc	 = ibm->cent_z[ elmt ];
                        
                        //ssxydif1=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n1e]+ibm->y_bp[n1e]));
                        //ssxydif2=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n2e]+ibm->y_bp[n2e]));
                        //ssxydif3=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n3e]+ibm->y_bp[n3e]));
                        ssxydif1=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n1e]+ibm->y_bp[n1e]));
                        ssxydif2=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n2e]+ibm->y_bp[n2e]));
                        ssxydif3=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n3e]+ibm->y_bp[n3e]));
//    Bend 90 degree                                       
	     // if((ydif1<1.e-6 && ydif2<1.e-6)||(ydif1<1.e-6 && ydif3<1.e-6)||(ydif2<1.e-6 && ydif3<1.e-6)) 


 //   Bend 135 degree
	     // if((xydif1<1.e-6 && xydif2<1.e-6)||(xydif1<1.e-6 && xydif3<1.e-6)||(xydif2<1.e-6 && xydif3<1.e-6))  
 

//   OSL inlet
                if((ssxydif1<1.e-4 && ssxydif2<1.e-4)||(ssxydif1<1.e-4 && ssxydif3<1.e-4)||(ssxydif2<1.e-4 && ssxydif3<1.e-4))  
	      {


		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt])
		{
			ibm->F_flux_12[ elmt ] = 0.;
			ibm->F_flux_13[ elmt ] = 0.;
			ibm->F_flux_23[ elmt ] = 0.;
		}
		else
		{
		        dx12 = ibm->x_bp[ n2e ] - ibm->x_bp[ n1e ];
			dy12 = ibm->y_bp[ n2e ] - ibm->y_bp[ n1e ];
			dz12 = ibm->z_bp[ n2e ] - ibm->z_bp[ n1e ];

			dx13 = ibm->x_bp[ n3e ] - ibm->x_bp[ n1e ];
			dy13 = ibm->y_bp[ n3e ] - ibm->y_bp[ n1e ];
			dz13 = ibm->z_bp[ n3e ] - ibm->z_bp[ n1e ];

			dx23 = ibm->x_bp[ n3e ] - ibm->x_bp[ n2e ];
			dy23 = ibm->y_bp[ n3e ] - ibm->y_bp[ n2e ];
			dz23 = ibm->z_bp[ n3e ] - ibm->z_bp[ n2e ];

// finding neighbor cells and related inflow and outflow sediment flux for current elmt 
			// for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ )  xiaolei deactivate 
			{
				//					1---->2 edge flux
				/*   // xiaolei deactivate 
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c1; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])&& ibm->nf_z[nbelmt]>0.8)
      					{
						ds    = sqrt( dx12 * dx12 + dy12 * dy12 );
						ibm->F_flux_12[ elmt ] += (ds/channle_width) * Q_total;
					}
					else
					{		
					}
				}

				//					1---->3 edge flux 

				/*  // xiaolei deactivate 
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/

				nbelmt=ibm->c2c[elmt].c2; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])&& ibm->nf_z[nbelmt]>0.8)
					{

						ds    = sqrt( dx13 * dx13 + dy13 * dy13 );
						ibm->F_flux_13[ elmt ] += (ds/channle_width) * Q_total;
					}
					else
               				{
					}
				}

				//					2---->3 edge  flux

				/* // xiaolei deactivate 
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/

				nbelmt=ibm->c2c[elmt].c3; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])&& ibm->nf_z[nbelmt]>0.8)
					{
						ds    = sqrt( dx23 * dx23 + dy23 * dy23 );
						ibm->F_flux_23[ elmt ] += (ds/channle_width) * Q_total;
					}
					else
					{
					}
				}
			} // for loop, nbelmt cycling
		}  // if,to check if elmt is on bed  
	 } // if,to check the cell face if at outlet
   }  // for loop, elmt cycling

	return ( 0 );		//	ali completed on 19 nov. 2009				    
}


PetscErrorCode	inlet_sediment_flux_contra( UserCtx * user, IBMNodes * ibm, double Q_total)
{
	DM	 da  = user->da, 
		 fda  = user->fda;

	int	   n_elmt  = ibm->n_elmt, 
					n_v  = ibm->n_v;
	IBMInfo  	 * ibminfo;
	IBMListNode  * current;
	int   elmt,nbelmt,	n1e,n2e,n3e;
	PetscReal	   riter;
	PetscReal	   dx12, 
					dy12, 
					dz12, 
					dx13, 
					dy13, 
					dz13, 
					dx23, 
					dy23, 
					dz23, 
					ds;
        PetscReal     channle_width = 2.75;   
	PetscReal                       //y_outlet = 3.41,
                                        //y_plus_x_outlet=29.11,                                            
                                        //someX_minus_Y_outlet = 33.07459,                                            
                                        //someX_plus_Y_inlet = 100.05944;                                            
                                        someX_plus_Y_inlet = 90.16268;                                            
        PetscReal                       ssxydif1,ssxydif2,ssxydif3;
        PetscReal                       xc,yc,zc;
                

// prescribing inlet fluxes at inlet faces which could be 1->2,3 or 2->3
// the outlet charachteristics is that the y is 3.41 there, later I must find a better characteristics for this, 18 Nov. 2009, ali
      
	for( elmt = 0; elmt < n_elmt; elmt++ )
	   {
	         	n1e  = ibm->nv1[ elmt ];
			n2e  = ibm->nv2[ elmt ];
			n3e  = ibm->nv3[ elmt ];
                        
                        xc	 = ibm->cent_x[ elmt ];
			yc	 = ibm->cent_y[ elmt ];
			zc	 = ibm->cent_z[ elmt ];
                        
                        //ssxydif1=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n1e]+ibm->y_bp[n1e]));
                        //ssxydif2=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n2e]+ibm->y_bp[n2e]));
                        //ssxydif3=fabs(someX_plus_Y_inlet-(0.95754*ibm->x_bp[n3e]+ibm->y_bp[n3e]));
                        ssxydif1=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n1e]+ibm->y_bp[n1e]));
                        ssxydif2=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n2e]+ibm->y_bp[n2e]));
                        ssxydif3=fabs(someX_plus_Y_inlet-(0.91932*ibm->x_bp[n3e]+ibm->y_bp[n3e]));
//    Bend 90 degree                                       
	     // if((ydif1<1.e-6 && ydif2<1.e-6)||(ydif1<1.e-6 && ydif3<1.e-6)||(ydif2<1.e-6 && ydif3<1.e-6)) 


 //   Bend 135 degree
	     // if((xydif1<1.e-6 && xydif2<1.e-6)||(xydif1<1.e-6 && xydif3<1.e-6)||(xydif2<1.e-6 && xydif3<1.e-6))  
 

//   OSL inlet
                if((ssxydif1<1.e-4 && ssxydif2<1.e-4)||(ssxydif1<1.e-4 && ssxydif3<1.e-4)||(ssxydif2<1.e-4 && ssxydif3<1.e-4))  
	      {


		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt])
		{
			ibm->F_flux_12[ elmt ] = 0.;
			ibm->F_flux_13[ elmt ] = 0.;
			ibm->F_flux_23[ elmt ] = 0.;
		}
		else
		{
		        dx12 = ibm->x_bp[ n2e ] - ibm->x_bp[ n1e ];
			dy12 = ibm->y_bp[ n2e ] - ibm->y_bp[ n1e ];
			dz12 = ibm->z_bp[ n2e ] - ibm->z_bp[ n1e ];

			dx13 = ibm->x_bp[ n3e ] - ibm->x_bp[ n1e ];
			dy13 = ibm->y_bp[ n3e ] - ibm->y_bp[ n1e ];
			dz13 = ibm->z_bp[ n3e ] - ibm->z_bp[ n1e ];

			dx23 = ibm->x_bp[ n3e ] - ibm->x_bp[ n2e ];
			dy23 = ibm->y_bp[ n3e ] - ibm->y_bp[ n2e ];
			dz23 = ibm->z_bp[ n3e ] - ibm->z_bp[ n2e ];

        //PetscPrintf(PETSC_COMM_WORLD, "entering the loop  \n");

// finding neighbor cells and related inflow and outflow sediment flux for current elmt 
	//		for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ )  // xiaolei deactivate 
			{
				//					1---->2 edge flux
				//
				/*  // xiaolei deactivate 
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c1; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
	      	                        if((ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])&& ibm->nf_z[nbelmt]>0.8)
      					{
						ds    = sqrt( dx12 * dx12 + dy12 * dy12 );
						ibm->F_flux_12[ elmt ] += (ds/channle_width) * Q_total;
					}
					else
					{		
					}
				}

				//					1---->3 edge flux 

				/* // xiaolei deactivate  
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c2; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])&& ibm->nf_z[nbelmt]>0.8)
					{

						ds    = sqrt( dx13 * dx13 + dy13 * dy13 );
						ibm->F_flux_13[ elmt ] += (ds/channle_width) * Q_total;
					}
					else
               				{
					}
				}

				//					2---->3 edge  flux

				/*  /// xiaolei deactivate 
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c3; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
	                            	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt])&& ibm->nf_z[nbelmt]>0.8)
					{
						ds    = sqrt( dx23 * dx23 + dy23 * dy23 );
						ibm->F_flux_23[ elmt ] += (ds/channle_width) * Q_total;
					}
					else
					{
					}
				}
			} // for loop, nbelmt cycling
		}  // if,to check if elmt is on bed  
	 } // if,to check the cell face if at outlet
   }  // for loop, elmt cycling

	return ( 0 );		//	ali completed on 19 nov. 2009				    
}



PetscErrorCode	PeriodicSedimentFlux( UserCtx * user, IBMNodes * ibm)
{
	DM	 da  = user->da, 
		 fda  = user->fda;

	int	   n_elmt  = ibm->n_elmt, 
					n_v  = ibm->n_v;
	int   elmt,nbelmt,	n1e,n2e,n3e;
        PetscReal     channle_width = 30.0;   
	PetscReal                       Xin = 0.,
                                        Xout = 739.9;                                            
        PetscReal                       sxdif1,sxdif2,sxdif3;

// finding the outlet Fluxes (of each outlet cell) to be fed at the corresponding inlet cell, April 4, 2012, ali
	for( elmt = 0; elmt < n_elmt; elmt++ )
	   {
	         	n1e  = ibm->nv1[ elmt ];
			n2e  = ibm->nv2[ elmt ];
			n3e  = ibm->nv3[ elmt ];
                        
                        sxdif1=fabs(Xout - ibm->x_bp[n1e]);
                        sxdif2=fabs(Xout - ibm->x_bp[n2e]);
                        sxdif3=fabs(Xout - ibm->x_bp[n3e]);
// Outlet Cells Only 
                if((sxdif1<1.e-4 && sxdif2<1.e-4)||(sxdif1<1.e-4 && sxdif3<1.e-4)||(sxdif2<1.e-4 && sxdif3<1.e-4))  
	      {


		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt])
		{
		}
		else
		{
// finding neighbor cells and related inflow and outflow sediment flux for current elmt 
			// for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ )  // xiaolei deactivate 
			{

				//					1---->2 edge flux
				/* // xiaolei deactivate 
				
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c1; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt]) && fabs(ibm->cent_x[nbelmt]- Xout)<1.e-4)
      					{
						int Nbr = int (0.5 * (ibm->y_bp[n1e]+ibm->y_bp[n2e]) / (channle_width/Nseg));
						ibm->SedFlux[Nbr] = ibm->F_flux_12[ elmt ];
					}
					else
					{		
					}
				}

				//					1---->3 edge flux 
				/*
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c2; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt]) && fabs(ibm->cent_x[nbelmt]- Xout)<1.e-4)
					{

						int Nbr = int (0.5 * (ibm->y_bp[n1e]+ibm->y_bp[n3e]) / (channle_width/Nseg));
						ibm->SedFlux[Nbr] = ibm->F_flux_13[ elmt ];
					}
					else
               				{
					}
				}

				//					2---->3 edge  flux
				/*  // xiaolei deactivate 
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c3; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt]) && fabs(ibm->cent_x[nbelmt]- Xout)<1.e-4)
					{
						int Nbr = int (0.5 * (ibm->y_bp[n2e]+ibm->y_bp[n3e]) / (channle_width/Nseg));
						ibm->SedFlux[Nbr] = ibm->F_flux_23[ elmt ];
					}
					else
					{
					}
				}
			} // for loop, nbelmt cycling
		}  // if,to check if elmt is on bed  
	 } // if,to check the cell face if at outlet
   }  // for loop, elmt cycling


// finding the inlet cell and assigning the flux from outlet cells, April 4, 2012, ali
	for( elmt = 0; elmt < n_elmt; elmt++ )
	   {
	         	n1e  = ibm->nv1[ elmt ];
			n2e  = ibm->nv2[ elmt ];
			n3e  = ibm->nv3[ elmt ];
                        
                        sxdif1=fabs(Xin - ibm->x_bp[n1e]);
                        sxdif2=fabs(Xin - ibm->x_bp[n2e]);
                        sxdif3=fabs(Xin - ibm->x_bp[n3e]);
// Outlet Cells Only 
                if((sxdif1<1.e-4 && sxdif2<1.e-4)||(sxdif1<1.e-4 && sxdif3<1.e-4)||(sxdif2<1.e-4 && sxdif3<1.e-4))  
	      {


		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt])
		{
		}
		else
		{
// finding neighbor cells and related inflow and outflow sediment flux for current elmt 
			// for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ )  // xiaolei deactivate 
			{

				//					1---->2 edge flux
				/* // xiaolei deactivate
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c1; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt]) && fabs(ibm->cent_x[nbelmt]- Xin)<1.e-4)
      					{
						int Nbr = int (0.5 * (ibm->y_bp[n1e]+ibm->y_bp[n2e]) / (channle_width/Nseg));
						ibm->F_flux_12[ elmt ] = -ibm->SedFlux[Nbr];
					}
					else
					{		
					}
				}

				//					1---->3 edge flux 
				/* xiaolei deactivate
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c2; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt]) && fabs(ibm->cent_x[nbelmt]- Xin)<1.e-5)
					{

						int Nbr = int (0.5 * (ibm->y_bp[n1e]+ibm->y_bp[n3e]) / (channle_width/Nseg));
						ibm->F_flux_13[ elmt ] = -ibm->SedFlux[Nbr];
					}
					else
               				{
					}
				}

				//					2---->3 edge  flux
				/*  // xiaolei deactivate
				if( nbelmt != elmt 
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c3; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
				       	if(( ibm->elmt_depth[nbelmt] > 0.3 || ibm->Rigidity[nbelmt]) && fabs(ibm->cent_x[nbelmt]- Xin)<1.e-5)
					{
						int Nbr = int (0.5 * (ibm->y_bp[n2e]+ibm->y_bp[n3e]) / (channle_width/Nseg));
						ibm->F_flux_23[ elmt ] = -ibm->SedFlux[Nbr];
					}
					else
					{
					}
				}
			} // for loop, nbelmt cycling
		}  // if,to check if elmt is on bed  
	 } // if,to check the cell face if at outlet
   }  // for loop, elmt cycling

	return ( 0 );						    
}


PetscErrorCode	check_correct_new_elev( UserCtx * user, IBMNodes * ibm )
{
	int		n_elmt	= ibm->n_elmt, 
					 n_v  = ibm->n_v;
	IBMInfo  	 *	ibminfo;
	IBMListNode  *	current;
	int		elmt, nbelmt, n1e, n2e, n3e;
	PetscReal		riter;
	PetscReal		nfx, nfy, nfz, dx12, dy12, dz12, dx13, dy13, dz13, dx23, dy23, dz23, dr, angle;
	PetscReal		xc, yc, zc;
	PetscReal		tmp;
	PetscReal		rtinyn	= 1.e-7, sign;
    
// check and correct z-direction angle between element's centerpoint

	for( elmt = 0; elmt < n_elmt; elmt++ )
	{
	//	if( ibm->nf_z[ elmt ] < 1.e-6 )
		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3)
		{
		}
		else
		{
	        	//xc	= ibm->cent_x[ elmt ];
		        //yc	= ibm->cent_y[ elmt ];
	                //zc      = ( ibm->z_bp[ n1e ] + ibm->z_bp[ n2e ] + ibm->z_bp[ n3e ] ) / 3.;
			//zc	= ibm->cent_z[ elmt ];

			n1e = ibm->nv1[ elmt ];
			n2e = ibm->nv2[ elmt ];
			n3e = ibm->nv3[ elmt ];

// finding neighbor cells  
			// for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ ) // xiaolei deactivate

			{
				/* xiaolei deactivate
				if( nbelmt != elmt
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
				            )
                                  )
				*/
				nbelmt=ibm->c2c[elmt].c1; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
					//				  abgle between elmt and  nbelmt beyond 1---->2 edge
				//	if( ibm->nf_z[ nbelmt ] < 1.e-6 )
		if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)
					{
					}
					else
					{
						dx12 = ibm->cent_x[ nbelmt ] - ibm->cent_x[ elmt ];
						dy12 = ibm->cent_y[ nbelmt ] - ibm->cent_y[ elmt ];
						dz12 = ibm->cent_z[ nbelmt ] - ibm->cent_z[ elmt ];
						dr	  = sqrt( dx12 * dx12 + dy12 * dy12 + dz12*dz12 );

						angle = asin( dz12 / dr );

                                                if(angle<0.){sign=-1.;}else{sign=1.;}
                                                
						if( fabs( angle )
							 >= ( 0.9 * Angle_repose ) )
						{
							ibm->cent_z[ nbelmt ]
							= ibm->cent_z[ elmt ]
							   + dr * sin( 0.9 * Angle_repose ) *sign;
						}
						else
						{
						}

					}
				}

				//				  abgle between elmt and  nbelmt beyond 1---->3 edge  
				/* // xiaolei deactivate
				if( nbelmt != elmt
					 && ( ibm->nv1[ nbelmt ] == n1e
						   || ibm->nv2[ nbelmt ] == n1e
						   || ibm->nv3[ nbelmt ] == n1e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c2; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
			//		if( ibm->nf_z[ nbelmt ] < 1.e-6 )
		if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)
					{
					}
					else
					{
						dx13 = ibm->cent_x[ nbelmt ] - ibm->cent_x[ elmt ];
						dy13 = ibm->cent_y[ nbelmt ] - ibm->cent_y[ elmt ];
						dz13 = ibm->cent_z[ nbelmt ] - ibm->cent_z[ elmt ];

						dr	  = sqrt( dx13 * dx13 + dy13 * dy13+dz13*dz13 );

						angle = asin( dz13 / dr );

                                                if(angle<0.){sign=-1.;}else{sign=1.;}

						if( fabs( angle )
							 >= ( 0.9 * Angle_repose ) )
						{
							ibm->cent_z[ nbelmt ]
							= ibm->cent_z[ elmt ]
							   + dr * sin( 0.9 * Angle_repose ) * sign;
						}
						else
						{
						}
					}
				}

				//				  abgle between elmt and  nbelmt beyond 2---->3 edge 
				//
				/* // xiaolei deactivate
				if( nbelmt != elmt
					 && ( ibm->nv1[ nbelmt ] == n2e
						   || ibm->nv2[ nbelmt ] == n2e
						   || ibm->nv3[ nbelmt ] == n2e
						 )
					 && ( ibm->nv1[ nbelmt ] == n3e
						   || ibm->nv2[ nbelmt ] == n3e
						   || ibm->nv3[ nbelmt ] == n3e
						 )
				   )
				*/
				nbelmt=ibm->c2c[elmt].c3; // xiaolei add
				if(nbelmt>0) // xiaolei add 
				{
			//		if( ibm->nf_z[ nbelmt ] < 1.e-6 )
		if( ibm->nf_z[nbelmt] < 1.e-6 || ibm->elmt_depth[nbelmt] > 0.3)
					{
					}
					else
					{
						dx23 = ibm->cent_x[ nbelmt ] - ibm->cent_x[ elmt ];
						dy23 = ibm->cent_y[ nbelmt ] - ibm->cent_y[ elmt ];
						dz23 = ibm->cent_z[ nbelmt ] - ibm->cent_z[ elmt ];

						dr	  = sqrt( dx23 * dx23 + dy23 * dy23+dz23*dz23 );

						angle = asin( dz23 / dr );

                                                if(angle<0.){sign=-1.;}else{sign=1.;}

						if( fabs( angle )
							 >= ( 0.9 * Angle_repose ) )
						{
							ibm->cent_z[ nbelmt ]
							= ibm->cent_z[ elmt ]
							   + dr * sin( 0.9 * Angle_repose ) * sign;
						}
						else
						{
						}

					}
				}
			}
		}
	}

	return ( 0 );												//	ali completed on 3 nov. 2009				    
}


PetscErrorCode	avalanche_first_sweep( UserCtx * user, IBMNodes * ibm )
{
	int		n_elmt	= ibm->n_elmt, n_v  = ibm->n_v;
	int		elmt, nbelmt, n1e, n2e, n3e;
	PetscReal		riter;
	PetscReal		nfx, nfy, nfz, dx, dy, dz, dr, angle;
	PetscReal		xc, yc, zc;
	PetscReal		tmp;
	PetscReal		rtinyn	= 1.e-7, sign;
    
// check and correct z-direction angle between element's centerpoint

	for( elmt = 0; elmt < n_elmt; elmt++ )
	{
         nfx = ibm->nf_x[elmt];
         nfy = ibm->nf_y[elmt];
         nfz = ibm->nf_z[elmt];

         ibm->deltz_p_us[elmt] = 0.;
         ibm->deltz_p_ds[elmt] = 0.;
         ibm->A_us[elmt] = 0.;
         ibm->A_ds[elmt] = 0.;
         
         if(nfz != 0. && ibm->elmt_depth[elmt]<0.3 && ibm->Rigidity[elmt]==0) ibm->max_bed_angle[elmt] = atan(sqrt( ( nfy / nfz ) * ( nfy / nfz ) + ( nfx / nfz ) * ( nfx / nfz ) ));
         if(nfz  = 0. || ibm->elmt_depth[elmt]>0.3 || ibm->Rigidity[elmt]==1) ibm->max_bed_angle[elmt] = 0.;   
            

		if( ibm->nf_z[elmt] > 1.e-7 && ibm->elmt_depth[elmt] < 0.3 && ibm->Rigidity[elmt] == 0 && fabs(ibm->max_bed_angle[elmt]) > Angle_repose)
		{

       
PetscPrintf(PETSC_COMM_WORLD, "avalanch model activated_first_sweep: elmt & Maxangle &  repose: %d %le %le\n",elmt, ibm->max_bed_angle[elmt]*180./PI, Angle_repose*180./PI);
			// for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ )  // xiaolei deactivate
			///  xiaolei add
//				int _nv[3];
//				_nv[0]=ibm->nv1[elmt];
//				_nv[1]=ibm->nv2[elmt];
//				_nv[2]=ibm->nv3[elmt];
	
				int _c2c[3];
				int ii,jj;
				_c2c[0]=ibm->c2c[elmt].c1;
				_c2c[1]=ibm->c2c[elmt].c2;
				_c2c[2]=ibm->c2c[elmt].c3;
				for(jj=0;jj<3;jj++)
//				for(ii=0;ii<100;ii++)
			// end add

			{

					nbelmt=_c2c[jj];	// xiaolei add
                 	 if(nbelmt>=0 && nbelmt != elmt && ibm->nf_z[nbelmt] > 1.e-7 && ibm->elmt_depth[nbelmt] < 0.3 && ibm->Rigidity[nbelmt] ==0
                            /*&& (ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] // xiaolei deactivate
                            || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ]
                            || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ]== ibm->nv3[ nbelmt ]
                            || ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ]== ibm->nv2[ nbelmt ]
                            || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ])*/   )	
//	&& (((ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ])
//             &&(ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv3[ nbelmt ]))
//
//	    ||((ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ])
 //             &&(ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ]))
//
//	    ||((ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv3[ nbelmt ])
 //             &&(ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ]))))
			       { 
 			        dx = ibm->cent_x[ nbelmt ] - ibm->cent_x[ elmt ];
				dy = ibm->cent_y[ nbelmt ] - ibm->cent_y[ elmt ];
				dz = ibm->cent_z[ nbelmt ] - ibm->cent_z[ elmt ];
				dr   = sqrt( dx * dx + dy * dy );

				angle = atan( dz / dr );

                                if(angle<0.){sign=-1.;}else{sign=1.;}
                                                
			//	if( fabs( angle )> Angle_repose || sign < 0.)
				  {
        PetscPrintf(PETSC_COMM_WORLD, "neighbour-elmts with angle larger than PHI, angle: %d %le\n",nbelmt, angle*180./PI);
                                   if(sign < 0.) {
                                      ibm->deltz_p_us[elmt] += ibm->dA[nbelmt]*(ibm->cent_z[nbelmt] - ibm->cent_z[elmt] + dr * tan(Angle_repose));
                                      ibm->A_us[elmt] +=ibm->dA[nbelmt];}

                                   if(sign > 0.){
                                      ibm->deltz_p_ds[elmt] += ibm->dA[nbelmt]*(ibm->cent_z[nbelmt] - ibm->cent_z[elmt] - dr * tan(Angle_repose));
                                      ibm->A_ds[elmt] +=ibm->dA[nbelmt];}

				  } // neighbour cells with big angle of connecting line -IF
			       } // neighbour cell -IF
			 }  // neighbour cell  -FOR
                    
 
                    ibm->deltz_p_us[elmt] /= (ibm->dA[elmt]+ibm->A_us[elmt]); 
                    ibm->deltz_p_ds[elmt] /= (ibm->dA[elmt]+ibm->A_ds[elmt]); 
                 
                         
			// for( nbelmt = 0; nbelmt < n_elmt; nbelmt++ )  // xiaolei deactivate
			///  xiaolei add
	
	//			int ii,jj;
				for(jj=0;jj<3;jj++)
//				for(ii=0;ii<100;ii++)
			// end add


			{

					nbelmt=_c2c[jj];	// xiaolei add
                 	 if(nbelmt>=0 && nbelmt != elmt && ibm->nf_z[nbelmt] > 1.e-7 && ibm->elmt_depth[nbelmt] < 0.3 && ibm->Rigidity[nbelmt]==0
                            /*&& (ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ]   // xiaolei deactivate
                            || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ]
                            || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ]== ibm->nv3[ nbelmt ]
                            || ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ]== ibm->nv2[ nbelmt ]
                            || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ])*/ )	
                        
//	&& (((ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ])
//             &&(ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv3[ nbelmt ]))
//
//	    ||((ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ])
//              &&(ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ]))
//
//	    ||((ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv3[ nbelmt ])
//              &&(ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ]))))

			       { 
 			        dx = ibm->cent_x[ nbelmt ] - ibm->cent_x[ elmt ];
				dy = ibm->cent_y[ nbelmt ] - ibm->cent_y[ elmt ];
				dz = ibm->cent_z[ nbelmt ] - ibm->cent_z[ elmt ];
				dr   = sqrt( dx * dx + dy * dy );

				angle = atan( dz / dr );

                                if(angle<0.){sign=-1.;}else{sign=1.;}
                                                
			//	if( fabs( angle )> Angle_repose || sign < 0.)
				  {
                                   if(sign < 0.) {
                                      ibm->deltz_p_us[nbelmt] = ibm->deltz_p_us[elmt] + ibm->cent_z[elmt] - ibm->cent_z[nbelmt] - dr * tan(Angle_repose);
                                      ibm->cent_z[nbelmt] += ibm->deltz_p_us[nbelmt];}

                                   if(sign > 0.){
                                      ibm->deltz_p_ds[nbelmt] = ibm->deltz_p_ds[elmt] + ibm->cent_z[elmt] - ibm->cent_z[nbelmt] + dr * tan(Angle_repose);
                                      ibm->cent_z[nbelmt] += ibm->deltz_p_ds[nbelmt];}

				  } // neighbour cells with big angle of connecting line -IF
			       } // neighbour cell -IF
			 }  // neighbour cell  -FOR

                   ibm->cent_z[elmt] += ibm->deltz_p_us[elmt] + ibm->deltz_p_ds[elmt];

		  } // bed elements with big bed-slope  -IF
	} // bed elements -FOR

	return ( 0 ); // July 2, 2010															    
}


PetscErrorCode	avalanche_second_sweep( UserCtx * user, IBMNodes * ibm )
{
	int		n_elmt	= ibm->n_elmt, n_v  = ibm->n_v;
	int		elmt, nbelmt, n1e, n2e, n3e;
	PetscReal		riter;
	PetscReal		nfx, nfy, nfz, dx, dy, dz, dr, angle;
	PetscReal		xc, yc, zc;
	PetscReal		tmp;
	PetscReal		rtinyn	= 1.e-7, sign;
    
// check and correct z-direction angle between element's centerpoint

	for( elmt = n_elmt; elmt >= 0; elmt-- )
	{
         nfx = ibm->nf_x[elmt];
         nfy = ibm->nf_y[elmt];
         nfz = ibm->nf_z[elmt];

         ibm->deltz_p_us[elmt] = 0.;
         ibm->deltz_p_ds[elmt] = 0.;
         ibm->A_us[elmt] = 0.;
         ibm->A_ds[elmt] = 0.;
         
         if(nfz != 0. && ibm->elmt_depth[elmt]<0.3 && ibm->Rigidity[elmt]==0) ibm->max_bed_angle[elmt] = atan(sqrt( ( nfy / nfz ) * ( nfy / nfz ) + ( nfx / nfz ) * ( nfx / nfz ) ));
         if(nfz  = 0. || ibm->elmt_depth[elmt]>0.3 || ibm->Rigidity[elmt]==1) ibm->max_bed_angle[elmt] = 0.;   
            

		if( ibm->nf_z[elmt] > 1.e-7 && ibm->elmt_depth[elmt] < 0.3 && ibm->Rigidity[elmt]==0 && fabs(ibm->max_bed_angle[elmt]) > Angle_repose)
		{

       
        PetscPrintf(PETSC_COMM_WORLD, "avalanch model activated_second_sweep: elmt & Maxangle &  repose: %d %le %le\n",elmt, ibm->max_bed_angle[elmt]*180./PI, Angle_repose*180./PI);
			// for( nbelmt = n_elmt; nbelmt >= 0; nbelmt-- ) // xiaolei deactivate
		///  xiaolei add

				int _c2c[3];
				int ii,jj;
				_c2c[0]=ibm->c2c[elmt].c1;
				_c2c[1]=ibm->c2c[elmt].c2;
				_c2c[2]=ibm->c2c[elmt].c3;

	
				for(jj=0;jj<3;jj++)
//				for(ii=0;ii<100;ii++)



			// end add


			{

					nbelmt=_c2c[jj];	// xiaolei add
                 	 if(nbelmt>=0 && nbelmt != elmt && ibm->nf_z[nbelmt] > 1.e-7 && ibm->elmt_depth[nbelmt] < 0.3 && ibm->Rigidity[nbelmt]==0
                            /*&& (ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] // xiaolei deactivate 
                            || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ]
                            || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ]== ibm->nv3[ nbelmt ]
                            || ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ]== ibm->nv2[ nbelmt ]
                            || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ])*/ )	
//	&& (((ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ])
//             &&(ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv3[ nbelmt ]))
//
//	    ||((ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ])
//             &&(ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ]))
//
//	    ||((ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv3[ nbelmt ])
//             &&(ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ]))))
			       { 
 			        dx = ibm->cent_x[ nbelmt ] - ibm->cent_x[ elmt ];
				dy = ibm->cent_y[ nbelmt ] - ibm->cent_y[ elmt ];
				dz = ibm->cent_z[ nbelmt ] - ibm->cent_z[ elmt ];
				dr   = sqrt( dx * dx + dy * dy );

				angle = atan( dz / dr );

                                if(angle<0.){sign=-1.;}else{sign=1.;}
                                                
			//	if( fabs( angle )> Angle_repose || sign < 0.)
				  {
        PetscPrintf(PETSC_COMM_WORLD, "neighbour-elmts with angle larger than PHI, anlgle: %d %le\n",nbelmt,angle*180./PI);
                                   if(sign < 0.) {
                                      ibm->deltz_p_us[elmt] += ibm->dA[nbelmt]*(ibm->cent_z[nbelmt] - ibm->cent_z[elmt] + dr * tan(Angle_repose));
                                      ibm->A_us[elmt] +=ibm->dA[nbelmt];}

                                   if(sign > 0.){
                                      ibm->deltz_p_ds[elmt] += ibm->dA[nbelmt]*(ibm->cent_z[nbelmt] - ibm->cent_z[elmt] - dr * tan(Angle_repose));
                                      ibm->A_ds[elmt] +=ibm->dA[nbelmt];}

				  } // neighbour cells with big angle of connecting line -IF
			       } // neighbour cell -IF
			 }  // neighbour cell  -FOR
                    
 
                    ibm->deltz_p_us[elmt] /= (ibm->dA[elmt]+ibm->A_us[elmt]); 
                    ibm->deltz_p_ds[elmt] /= (ibm->dA[elmt]+ibm->A_ds[elmt]); 
                 
                         
			// for( nbelmt = n_elmt; nbelmt >=0; nbelmt-- )  // xiaolei deactivate 
		///  xiaolei add
//				int ii,jj;
				for(jj=0;jj<3;jj++)
			// end add


			{

					nbelmt=_c2c[jj];	// xiaolei add
                 	 if(nbelmt>=0 && nbelmt != elmt && ibm->nf_z[nbelmt] > 1.e-7 && ibm->elmt_depth[nbelmt] < 0.3 && ibm->Rigidity[nbelmt]==0
                            /*&& (ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] // xiaolei deactivate 
                            || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ]
                            || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ]== ibm->nv3[ nbelmt ]
                            || ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ]== ibm->nv2[ nbelmt ]
                            || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ])*/ )	
//	&& (((ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ])
//             &&(ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv3[ nbelmt ]))
//
//	    ||((ibm->nv1[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv1[ elmt ] == ibm->nv3[ nbelmt ])
//              &&(ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ]))
//
//	    ||((ibm->nv2[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv2[ elmt ] == ibm->nv3[ nbelmt ])
//              &&(ibm->nv3[ elmt ] == ibm->nv1[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv2[ nbelmt ] || ibm->nv3[ elmt ] == ibm->nv3[ nbelmt ]))))
			       { 
 			        dx = ibm->cent_x[ nbelmt ] - ibm->cent_x[ elmt ];
				dy = ibm->cent_y[ nbelmt ] - ibm->cent_y[ elmt ];
				dz = ibm->cent_z[ nbelmt ] - ibm->cent_z[ elmt ];
				dr   = sqrt( dx * dx + dy * dy );

				angle = atan( dz / dr );

                                if(angle<0.){sign=-1.;}else{sign=1.;}
                                                
			//	if( fabs( angle )> Angle_repose || sign < 0.)
				  {
                                   if(sign < 0.) {
                                      ibm->deltz_p_us[nbelmt] = ibm->deltz_p_us[elmt] + ibm->cent_z[elmt] - ibm->cent_z[nbelmt] - dr * tan(Angle_repose);
                                      ibm->cent_z[nbelmt] += ibm->deltz_p_us[nbelmt];}

                                   if(sign > 0.){
                                      ibm->deltz_p_ds[nbelmt] = ibm->deltz_p_ds[elmt] + ibm->cent_z[elmt] - ibm->cent_z[nbelmt] + dr * tan(Angle_repose);
                                      ibm->cent_z[nbelmt] += ibm->deltz_p_ds[nbelmt];}

				  } // neighbour cells with big angle of connecting line -IF
			       } // neighbour cell -IF
			 }  // neighbour cell  -FOR

                   ibm->cent_z[elmt] += ibm->deltz_p_us[elmt] + ibm->deltz_p_ds[elmt];

		  } // bed elements with big bed-slope  -IF
	} // bed elements -FOR

	return ( 0 ); // July 2, 2010															    
}


PetscErrorCode recomputing_geometry(UserCtx * user, IBMNodes * ibm, PetscInt tistart, int ti, int itr_sc, int avalanche_check_number)
{

	int	         n_elmt  = ibm->n_elmt, n_v  = ibm->n_v;
	int		 iter, elmt, vert,n1e,n2e,n3e;
	PetscReal		 xc, yc, zc, dx12, dx13, dy12, dy13, dz12, dz13, dr;
        int itt=0;        
        
        for( elmt = 0; elmt < n_elmt; elmt++ )
	{
                itt++;
		n1e  			  = ibm->nv1[ elmt ];
		n2e  			  = ibm->nv2[ elmt ];
		n3e  			  = ibm->nv3[ elmt ];
		dx12			  = ibm->x_bp[ n2e ] - ibm->x_bp[ n1e ];
		dy12			  = ibm->y_bp[ n2e ] - ibm->y_bp[ n1e ];
		dz12			  = ibm->z_bp[ n2e ] - ibm->z_bp[ n1e ];

		dx13			  = ibm->x_bp[ n3e ] - ibm->x_bp[ n1e ];
		dy13			  = ibm->y_bp[ n3e ] - ibm->y_bp[ n1e ];
		dz13			  = ibm->z_bp[ n3e ] - ibm->z_bp[ n1e ];

		ibm->nf_x[ elmt ] = dy12 * dz13 - dz12 * dy13;
		ibm->nf_y[ elmt ] = -dx12 * dz13 + dz12 * dx13;
		ibm->nf_z[ elmt ] = dx12 * dy13 - dy12 * dx13;

		dr = sqrt(ibm->nf_x[ elmt ] * ibm->nf_x[ elmt ] + ibm->nf_y[ elmt ] * ibm->nf_y[ elmt ]
				+ ibm->nf_z[ elmt ] * ibm->nf_z[ elmt ] );

		ibm->nf_x[ elmt ] /= dr;
		ibm->nf_y[ elmt ] /= dr;
		ibm->nf_z[ elmt ] /= dr;

		ibm->dA[ elmt ]    = dr / 2.;
                PetscInt steps=100;
                if((ti==tistart || ti==0) && itt==1 && avalanche_check_number==0) PetscPrintf(PETSC_COMM_WORLD,"Averaging Cz onto z_bp every: %d \n",smooth_bed);
                if(((ti-tistart)/smooth_bed)*smooth_bed== (ti-tistart))  // xiaolei deactivate
                {
			//ibm->cent_z[ elmt ] = (ibm->z_bp[ n1e ] + ibm->z_bp[ n2e ] + ibm->z_bp[ n3e ] ) / 3.;   // xiaolei deactivate  RTD 
          	                              
                }
         }
	return (0);
}



PetscErrorCode Scour(UserCtx * user, IBMNodes * ibm, PetscInt tistart, int ti, int itr_sc)
{
        PetscBool SMOOTHING;
	DM                da  = user->da, fda  = user->fda;
	DMDALocalInfo  	 info  = user->info;
//---------------------------
	Cmpnts		 *** ucat;
	Cmpnts		 *   Bvel;  
//	PetscReal	 *** ustar;       
	int	     n_elmt  = ibm->n_elmt, n_v  = ibm->n_v;
	PetscReal	 *	 F_flux_12, 
			 *	 F_flux_13, 
			 *	 F_flux_23, *elmt_depth,*netflux_old;
	PetscReal	 *	 SCont;
	//---------------------------
//	IBMListNode      *	 current;
//	PetscReal	 *** nvert, *** nvert_o;
//	IBMInfo  	 *	 ibminfo;
//--------------------------
        int rank;
	int		 i,iter, iteration, elmt, vert,n1e,n2e,n3e;
	PetscReal		 nfx, nfy, nfz;
	PetscReal		 ucx, ucy, ucz;
	PetscReal		 xc, yc, zc, dx12, dx13, dy12, dy13, dz12, dz13, dr, riter;
	PetscReal		 netflux, dzdt, z_minimum,z_maximum;
        PetscReal                maxz=0.0, minz=100.0;
        int                 bend = 0;
        int                 contra = 1;
        PetscReal                checoo;
        PetscReal                ts,te,cputime;
        PetscReal                Total_Area, infinity_norm, smoothing_residual;
        int                 aval_loop = 4;
        //PetscReal                dZ[n_elmt], dZ_old[n_elmt], atke[n_elmt], under_relax[n_elmt];
        extern PetscInt projection_method;
        int                 AngleSkewness_compute = 0;
//---------------------------
	// for least squares
	double a[6][6], a1[6][6], b[6], c[6], d[6], c_[6];
	double xi[100], yi[100], zi[100], dcoef[100];
	double flux_x[100], flux_y[100], flux_z[100];
	int label[100];
	double _xc, _yc, _zc;
	double bed_angle;

	double wgt[100];;
	int N_ac;	
//--------------------------------------
// compute new normal vec to elmts, elmt area, elmt center point coordinate

  PetscReal ts_1, te_1;

  PetscGetTime(&ts_1);

 if(ti==tistart || ti==0)
 {
        ibm->dtime=0.;
        ibm->time_bedchange=0.;
        int itt=0;

        for( elmt = 0; elmt < n_elmt; elmt++ )
	{
                itt++;
		n1e  			  = ibm->nv1[ elmt ];
		n2e  			  = ibm->nv2[ elmt ];
		n3e  			  = ibm->nv3[ elmt ];
		dx12			  = ibm->x_bp[ n2e ] - ibm->x_bp[ n1e ];
		dy12			  = ibm->y_bp[ n2e ] - ibm->y_bp[ n1e ];
		dz12			  = ibm->z_bp[ n2e ] - ibm->z_bp[ n1e ];


		dx13			  = ibm->x_bp[ n3e ] - ibm->x_bp[ n1e ];
		dy13			  = ibm->y_bp[ n3e ] - ibm->y_bp[ n1e ];
		dz13			  = ibm->z_bp[ n3e ] - ibm->z_bp[ n1e ];

		ibm->nf_x[ elmt ] = dy12 * dz13 - dz12 * dy13;
		ibm->nf_y[ elmt ] = -dx12 * dz13 + dz12 * dx13;
		ibm->nf_z[ elmt ] = dx12 * dy13 - dy12 * dx13;

		dr = sqrt(ibm->nf_x[ elmt ] * ibm->nf_x[ elmt ] + ibm->nf_y[ elmt ] * ibm->nf_y[ elmt ]
				+ ibm->nf_z[ elmt ] * ibm->nf_z[ elmt ] );

		ibm->nf_x[ elmt ] /= dr;
		ibm->nf_y[ elmt ] /= dr;
		ibm->nf_z[ elmt ] /= dr;

		ibm->dA[ elmt ]    = dr / 2.;

		ibm->cent_x[ elmt ] = ( ibm->x_bp[ n1e ] + ibm->x_bp[ n2e ] + ibm->x_bp[ n3e ] ) / 3.;
		ibm->cent_y[ elmt ] = ( ibm->y_bp[ n1e ] + ibm->y_bp[ n2e ] + ibm->y_bp[ n3e ] ) / 3.;
		ibm->cent_z[ elmt ] = ( ibm->z_bp[ n1e ] + ibm->z_bp[ n2e ] + ibm->z_bp[ n3e ] ) / 3.;
		
		if(itr_sc==1)
                {
                ibm->cent_z_AVE[ elmt ] = ibm->cent_z[elmt];
		ibm->cent_z_old[ elmt ] = ibm->cent_z[elmt];
		ibm->cent_zl[ elmt ] = ibm->cent_z[elmt];
                if(itt==1){
                PetscPrintf(PETSC_COMM_WORLD, "ti start: %d \n",ti);
                PetscPrintf(PETSC_COMM_WORLD, "Angle of Repose: %e\n",Angle_repose*180./PI);
                PetscPrintf(PETSC_COMM_WORLD, "Prosity:  %e\n",bed_porosity);
                PetscPrintf(PETSC_COMM_WORLD, "Avalanche correction loop numbers:  %d\n",aval_loop);
                PetscPrintf(PETSC_COMM_WORLD, "projection_method:  %d\n",projection_method);
                }
	        }

                /*
                double Dis1 = PetscMax(sqrt((ibm->cent_x[elmt]-ibm->x_bp[n1e])*(ibm->cent_x[elmt]-ibm->x_bp[n1e])
                                 +(ibm->cent_y[elmt]-ibm->y_bp[n1e])*(ibm->cent_y[elmt]-ibm->y_bp[n1e])), 1.e-10);
                            
                double Dis2 = PetscMax(sqrt((ibm->cent_x[elmt]-ibm->x_bp[n2e])*(ibm->cent_x[elmt]-ibm->x_bp[n2e])
                                 +(ibm->cent_y[elmt]-ibm->y_bp[n2e])*(ibm->cent_y[elmt]-ibm->y_bp[n2e])),1.e-10);

                double Dis3 = PetscMax(sqrt((ibm->cent_x[elmt]-ibm->x_bp[n3e])*(ibm->cent_x[elmt]-ibm->x_bp[n3e])
                                 +(ibm->cent_y[elmt]-ibm->y_bp[n3e])*(ibm->cent_y[elmt]-ibm->y_bp[n3e])),1.e-10);
                            
          	ibm->cent_z[elmt] =  (ibm->z_bp[n1e]/Dis1+ibm->z_bp[n2e]/Dis2+ibm->z_bp[n3e]/Dis3)/(1./Dis1+1./Dis2+1./Dis3); */

	        z_minimum=PetscMin(ibm->z_bp[ibm->nv1[elmt]],ibm->z_bp[ibm->nv2[elmt]]);
                z_minimum=PetscMin(z_minimum,ibm->z_bp[ibm->nv3[elmt]]); 
                z_maximum=PetscMax(ibm->z_bp[ibm->nv1[elmt]],ibm->z_bp[ibm->nv2[elmt]]);
                z_maximum=PetscMax(z_maximum,ibm->z_bp[ibm->nv3[elmt]]); 
                
                if(!input_ib_depth)
                                  {
                 ibm->elmt_depth[elmt]=fabs(z_maximum-z_minimum);
                if(ibm->elmt_depth[elmt]>0.02)ibm->elmt_depth[elmt]=5.;
                //if(ibm->elmt_depth[elmt]>1.5)ibm->elmt_depth[elmt]=5.;
                // added just for Indoor_Rock_vane case for z over 1.0 :
                // added just for Cross_vane case for z over 0.0 :
                // added just for Jhook case for z over 0.5 :
                if(ibm->cent_z[elmt]<1.32)ibm->elmt_depth[elmt]=5.;
                if(ibm->cent_z[elmt]>1.45)ibm->elmt_depth[elmt]=5.;
                //if(ibm->cent_z[elmt]>=1.3 && ibm->cent_x[elmt]>275.3 && ibm->cent_x[elmt]<294.6)ibm->elmt_depth[elmt]=5.;
                if(ibm->cent_x[elmt]<30.96769199)ibm->elmt_depth[elmt]=5.;
                if(ibm->nf_z[elmt]<1.e-6)ibm->elmt_depth[elmt]=5.;
                                   

                //if(ibm->cent_x[elmt]<2.0)ibm->elmt_depth[elmt]=5.;
                //PetscReal line_1 = 100.05944;
                //PetscReal line_1 = 90.16268;
                //PetscReal line_2 = 33.07459;
                //PetscReal elmt_coord_1 = 0.95754 * ibm->cent_x[elmt] + ibm->cent_y[elmt]; 
                //PetscReal elmt_coord_1 = 0.91932 * ibm->cent_x[elmt] + ibm->cent_y[elmt]; 
                //PetscReal elmt_coord_2 = 1.13087 * ibm->cent_x[elmt] - ibm->cent_y[elmt]; 
                //if(elmt_coord_1 > line_1 && elmt_coord_2 < line_2)
                //if(elmt_coord_1 > line_1)
                // {
                //  ibm->Rigidity[elmt] = 0;
                // } else { 
                //  ibm->Rigidity[elmt] = 1;
                // }

                ibm->Mobility[elmt] = 1;
                ibm->Rigidity[elmt] = 0;
                if(ibm->elmt_depth[elmt] > 0.3) ibm->Rigidity[elmt] = 1;
                if(ibm->elmt_depth[elmt] > 0.3) ibm->Mobility[elmt] = 0;
                                   }
                ibm->netflux_old[elmt]=0.;
              }
  }

  PetscGetTime(&te_1);

  PetscPrintf(PETSC_COMM_WORLD, "#### IB_Depth  cputime %le\n", te_1-ts_1);



  PetscReal ts_2, te_2;

  PetscGetTime(&ts_2);

        
if(STRONG_COUPLING)
{
  for(vert = 0; vert < ibm->n_v; vert++)
     {       
      ibm->z_bp_l[vert]=ibm->z_bp[vert];
     }
}          

//Computing Angle of Skewness
//if(AngleSkewness_compute) AngleSkewness(ibm);


// Read and/or Write the "ibm->elmt_depth[elmt]"
    if(input_ib_depth && (ti==tistart || ti==0))
    {  
    read_bed_cell_depth(ibm, tistart);
    PetscBarrier( PETSC_NULL );  								//	stop till all procc to the jobs  			   
    } 
     else
    { 
    /*if (ti==tistart || ti==0)*/ {  
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if( (ti==tistart) && (!input_ib_depth) && (!rank) ){
    FILE *fd;
    PetscPrintf(PETSC_COMM_SELF, "WRITE bed cell depth data\n");
    char string[128];
    char filen[80];
    sprintf(filen, "ib_cell_depth.dat");
    fd = fopen(filen, "w");
   
    if (fd) {
     
    for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, fd, "%le %d\n", ibm->elmt_depth[i], ibm->Rigidity[i]);
                                  }
            } fclose(fd);
            }
            }
      
     PetscBarrier( PETSC_NULL );}  								//	stop till all procc to the jobs  			   
// End of Read and/or Write "ibm->elmt_depth[elmt]"

/*    if (ti==tistart) {  
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if(!rank) { // root processor read in the data
    FILE *fd;
    PetscPrintf(PETSC_COMM_SELF, "READ bed cell concentration data\n");
    char filen[90];  
    sprintf(filen,"bed_elmt_concntrtn_%5.5d" , ti-1);
 
    fd = fopen(filen, "r"); if (!fd) SETERRQ(1, "Cannot open IBM node file\n")
    for (i=0; i<ibm->n_elmt; i++) {
	fscanf(fd, "%e\n", ibm->SCont[i]);
	                          }
    MPI_Bcast(ibm->SCont, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
              }
    else  
                                  {
    MPI_Bcast(ibm->SCont, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
                                  }
                                      }
	PetscBarrier( PETSC_NULL );  								//	stop till all procc to the jobs  			   
*/


  PetscGetTime(&te_2);

  PetscPrintf(PETSC_COMM_WORLD, "#### Write/Read IB_Depth  cputime %le\n", te_2-ts_2);


  PetscReal ts_3, te_3;

  PetscGetTime(&ts_3);


	/* 
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	if (rank==1) {
    char filen[80];  
    FILE *f;
    
    sprintf(filen, "surface_elmt_depth_test1.dat");
    f = fopen(filen, "w");
    PetscFPrintf(PETSC_COMM_SELF, f, "Variables=x,y,z,elmt_depth,rigidity\n");
    PetscFPrintf(PETSC_COMM_SELF, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-5]=CELLCENTERED)\n",ibm->n_v, ibm->n_elmt);
    for (i=0; i<ibm->n_v; i++) {
      PetscFPrintf(PETSC_COMM_SELF, f, "%e\n", ibm->x_bp[i]);
    }
    for (i=0; i<ibm->n_v; i++) {
      PetscFPrintf(PETSC_COMM_SELF, f, "%e\n", ibm->y_bp[i]);
    }
    for (i=0; i<ibm->n_v; i++) {	
      PetscFPrintf(PETSC_COMM_SELF, f, "%e\n", ibm->z_bp[i]);
    }
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_SELF, f, "%le\n", ibm->elmt_depth[i]);
    }
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_SELF, f, "%d\n", ibm->Rigidity[i]);
    }

      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_SELF, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
      }

    fclose(f);

	}

    PetscBarrier( PETSC_NULL );  								//	stop till all procc to the jobs  			   
	exit(0);
	*/
if(projection_method == 1)  {
// call for computing flow at reference level
        PetscGetTime(&ts);
        flow_variables_ref_level (user, ibm, ti, tistart);
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "EXNER_WALL-Function_ref_level  cputime %d %le\n", ti,cputime);

	// xiaolei deactivate

// call for computing Bvel, and BShS on the bed surface
	/*
        PetscGetTime(&ts);
	sediment_variables_projection( user, ibm );
	PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "sediment_variable Projection  cputime %d %le\n", ti,cputime);
        PetscGetTime(&ts);
	Projecting( user, ibm );
	PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "Projecting  cputime %d %le\n", ti,cputime);
	*/

	// xiaolei deactivate
	//
	/*
        PetscGetTime(&ts);
	Finalizing_Projecting( user, ibm, ti, tistart );
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "Finalizing_Projecting  cputime %d %le\n", ti,cputime);
	*/
        //PetscPrintf(PETSC_COMM_WORLD, "EXNER_COMM  cputime %d %le\n", ti,cputime);
}
	
/*
if(projection_method == 2)  {
// call for computing flow to a point above bed load layer normal from centroid of bed cell 
        PetscGetTime(&ts);
        ibm_intp_pj_centroid (user, ibm, ti, tistart);
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "EXNER_ibm_intp_pj_centroid  cputime %d %le\n", ti,cputime);
    
// call for comunicating velocities above bed load layer 
        PetscGetTime(&ts);
	Projecting_new( user, ibm );
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "EXNER_COMM through Projecting_new  cputime %d %le\n", ti,cputime);


// call for wallfunction to construct the velociies and shear stress on the ref level 
        PetscGetTime(&ts);
        flow_variables_ref_level_new (user, ibm, ti, tistart);
	PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "EXNER_wallfunction  cputime %d %le\n", ti,cputime);
                          }

*/

  PetscGetTime(&te_3);

  PetscPrintf(PETSC_COMM_WORLD, "#### Total Projecting cputime %le\n", te_3-ts_3);



  PetscReal ts_4, te_4;

  PetscGetTime(&ts_4);

if (ti!=tistart && ti == (ti/bed_avg)*bed_avg) {  // xiaolei add 

	PetscGetTime(&ts);
	Finalizing_Projecting( user, ibm, ti, tistart );
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "Finalizing_Projecting  cputime %d %le\n", ti,cputime);

iteration=0;
SMOOTHING = PETSC_TRUE;
while(SMOOTHING)
{           
        iteration++;
// call for Critical bed shear stress computation
        PetscGetTime(&ts);
        bed_concentration( user, ibm, ti, tistart );
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "EXNER_BED-Concentration  cputime %d %le\n", ti,cputime);


	if (particlevel_model) {
        PetscPrintf(PETSC_COMM_WORLD, "bed velocity model \n");
	bedvel_model( user, ibm );
	}


// computing sediment fluxes for every elemnts on bed surface
        PetscGetTime(&ts);
	sediment_flux( user, ibm, ti, tistart );
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "EXNER_SED-Flux-inner-domain  cputime %d %le\n", ti,cputime);

// outlet flux  must be adjusted for each case study, ali 18 Nov. 2009
	// xiaolei deactivate 
	/*
        PetscGetTime(&ts);
        if(bend) outlet_sediment_flux_bend( user, ibm );
        if(contra) outlet_sediment_flux_contra( user, ibm); // xiaolei deactivate  RTD
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "EXNER_SED-Flux-Boundary  cputime %d %le\n", ti,cputime);
	*/
// inlet flux  must be adjusted for each case study, ali Jan 4, 2012
//
	// xiaolei deactivate 
	/*
       if(inlet_sediment_flux > 0.)
       { 
       //double inlet_sediment_flux = 4.0 /(1650. * 60. * 0.32 * 0.3 * 0.3);
        PetscGetTime(&ts);
        if(bend) inlet_sediment_flux_bend( user, ibm, inlet_sediment_flux );
        // if(contra) inlet_sediment_flux_contra( user, ibm, inlet_sediment_flux); // xiaolei deactivate  RTD 
        PetscGetTime(&te);
        cputime=te-ts;
       }
        PetscPrintf(PETSC_COMM_WORLD, "EXNER_SED-Flux-Boundary  cputime %d %le\n", ti,cputime);
	*/

// Inlet_sediment_flux to be set for Periodic B.C. Apr 4, 2012
       if(periodic_morpho)
       { 
        PetscGetTime(&ts);
         PeriodicSedimentFlux(user, ibm);
        PetscGetTime(&te);
        cputime=te-ts;
        PetscPrintf(PETSC_COMM_WORLD, "Periodicity B.C. for Bed-Load  cputime %d %le\n", ti,cputime);
       }

// defineing the time steps:
        if(itr_sc==1 && iteration ==1)
        {
	/*double Max_dzdt=0.0;
        for (elmt=0;elmt<n_elmt;elmt++)
            {
	      if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.1)
		{
		}
	      else
		{
	         netflux = ibm->F_flux_12[ elmt ] + ibm->F_flux_13[ elmt ] + ibm->F_flux_23[ elmt ];
                 dzdt = netflux * ( 1. / ibm->dA[ elmt ] )  / ( 1. - bed_porosity );
                 Max_dzdt=PetscMax(Max_dzdt, fabs(dzdt));           
                }
	     }*/
 
        ibm->dtime = dt_bed; //dt_bed*user->dt;//3.0; //0.01/Max_dzdt;
        ibm->time_bedchange+=ibm->dtime;
        PetscPrintf(PETSC_COMM_WORLD, "dt and time :  %e %e\n",ibm->dtime, ibm->time_bedchange);
        }
        

	// xiaolei deactivate 
        smoothing_residual=0.0;
        for( elmt = 0; elmt < n_elmt; elmt++ )
	{

		double diffuse = 0.0;
		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt]==1)
		{
                        ibm->Delz[elmt]=0.; ibm->netflux_old[elmt]=0.; ibm->cent_z[elmt]=ibm->cent_z_old[elmt];
		}
	      	else
		{

			N_ac=0;
			_xc=0.0;
			_yc=0.0;
			_zc=0.0;
			int ii,jj, nii;
			for(jj=0;jj<6;jj++) {
				for(ii=0;ii<6;ii++) {
					a[jj][ii]=0.0;			
				}
				b[jj]=0.0;
				c[jj]=0.0;
				c_[jj]=0.0;
				d[jj]=0.0;
			}

			nfx = ibm->nf_x[elmt];
		        nfy = ibm->nf_y[elmt];
        		nfz = ibm->nf_z[elmt];

			double ang_bed=atan(sqrt( ( nfy / nfz ) * ( nfy / nfz ) + ( nfx / nfz ) * ( nfx / nfz ) ));


			xi[0]=ibm->cent_x[elmt];
			yi[0]=ibm->cent_y[elmt];
			zi[0]=ibm->cent_z[elmt];
			dcoef[0]=0.0001*PetscMax((ang_bed-Angle_repose)/Angle_repose, 0.0);
			_xc=ibm->cent_x[elmt];
			_yc=ibm->cent_y[elmt];
			_zc=ibm->cent_z[elmt];
			N_ac=1;
			label[0]=1;

	
			for(ii=1;ii<100;ii++) {
				xi[ii]=0.0;			
				yi[ii]=0.0;			
				zi[ii]=0.0;			
				dcoef[ii]=0.0;			
				label[ii]=0.0;			
				nii=ibm->c2ac[elmt].c[ii];	// xiaolei add
				if (nii>=0) {    // xiaolei add 
                			int rigid = ibm->Rigidity[nii];
					if(ibm->nf_z[nii]<1.e-6 || ibm->elmt_depth[nii]>0.3 || rigid ) 
                	                { 
                        	        }
	                                else
        	                        {

						nfx = ibm->nf_x[nii];
					        nfy = ibm->nf_y[nii];
			        		nfz = ibm->nf_z[nii];

						double ang_bed=atan(sqrt( ( nfy / nfz ) * ( nfy / nfz ) + ( nfx / nfz ) * ( nfx / nfz ) ));


						xi[ii]=ibm->cent_x[nii];
						yi[ii]=ibm->cent_y[nii];
						zi[ii]=ibm->cent_z[nii];
						dcoef[ii]=0.0001*PetscMax((ang_bed-Angle_repose)/Angle_repose, 0.0);
						_xc+=ibm->cent_x[nii];
						_yc+=ibm->cent_y[nii];
						_zc+=ibm->cent_z[nii];
						N_ac+=1;
						label[ii]=1;
					}
				}
			}

			_xc/=double(N_ac);
			_yc/=double(N_ac);
			_zc/=double(N_ac);

			for(ii=0;ii<100;ii++) {
				if (label[ii]) {
					//xi[ii]-=_xc;
					//yi[ii]-=_yc;
				}
			}



			if (N_ac<6) {
			} else {
				for(ii=0;ii<100;ii++) {
					if (label[ii]) {
						b[0]+=zi[ii];
						b[1]+=zi[ii]*xi[ii];
						b[2]+=zi[ii]*yi[ii];
						b[3]+=zi[ii]*xi[ii]*yi[ii];
						b[4]+=zi[ii]*xi[ii]*xi[ii];
						b[5]+=zi[ii]*yi[ii]*yi[ii];

						d[0]+=dcoef[ii];
						d[1]+=dcoef[ii]*xi[ii];
						d[2]+=dcoef[ii]*yi[ii];
						d[3]+=dcoef[ii]*xi[ii]*yi[ii];
						d[4]+=dcoef[ii]*xi[ii]*xi[ii];
						d[5]+=dcoef[ii]*yi[ii]*yi[ii];

						double _a0=1, _a1=xi[ii], _a2=yi[ii], _a3=xi[ii]*yi[ii], _a4=xi[ii]*xi[ii], _a5=yi[ii]*yi[ii];
						a[0][0]+=_a0; a[0][1]+=_a1; a[0][2]+=_a2; a[0][3]+=_a3; a[0][4]+=_a4; a[0][5]+=_a5;
						a[1][0]+=_a0*xi[ii]; a[1][1]+=_a1*xi[ii]; a[1][2]+=_a2*xi[ii]; a[1][3]+=_a3*xi[ii]; a[1][4]+=_a4*xi[ii]; a[1][5]+=_a5*xi[ii];
						a[2][0]+=_a0*yi[ii]; a[2][1]+=_a1*yi[ii]; a[2][2]+=_a2*yi[ii]; a[2][3]+=_a3*yi[ii]; a[2][4]+=_a4*yi[ii]; a[2][5]+=_a5*yi[ii];
						a[3][0]+=_a0*xi[ii]*yi[ii]; a[3][1]+=_a1*xi[ii]*yi[ii]; a[3][2]+=_a2*xi[ii]*yi[ii]; a[3][3]+=_a3*xi[ii]*yi[ii]; a[3][4]+=_a4*xi[ii]*yi[ii]; a[3][5]+=_a5*xi[ii]*yi[ii];
						a[4][0]+=_a0*xi[ii]*xi[ii]; a[4][1]+=_a1*xi[ii]*xi[ii]; a[4][2]+=_a2*xi[ii]*xi[ii]; a[4][3]+=_a3*xi[ii]*xi[ii]; a[4][4]+=_a4*xi[ii]*xi[ii]; a[4][5]+=_a5*xi[ii]*xi[ii];
						a[5][0]+=_a0*yi[ii]*yi[ii]; a[5][1]+=_a1*yi[ii]*yi[ii]; a[5][2]+=_a2*yi[ii]*yi[ii]; a[5][3]+=_a3*yi[ii]*yi[ii]; a[5][4]+=_a4*yi[ii]*yi[ii]; a[5][5]+=_a5*yi[ii]*yi[ii];
					}
				}			

				int ia, ja;
				for(ia=0;ia<6;ia++)
				for(ja=0;ja<6;ja++) {
					a1[ja][ia]=a[ja][ia];
				}


				AGAUS (a, b, c);

				double _dcoef=0.0001*PetscMax((ang_bed-Angle_repose)/Angle_repose, 0.0);
				//double _dcoef=0.000001;
				diffuse=2.0*_dcoef*(c[4]+c[5]);

				for(ia=0;ia<6;ia++)
				for(ja=0;ja<6;ja++) {
					a[ja][ia]=a1[ja][ia];
				}

				AGAUS (a, d, c_);

				diffuse+=(c[1]+c[3]*ibm->cent_y[elmt]+2*c[4]*ibm->cent_x[elmt])*(c_[1]+c_[3]*ibm->cent_y[elmt]+2*c_[4]*ibm->cent_x[elmt]);
				diffuse+=(c[2]+c[3]*ibm->cent_x[elmt]+2*c[5]*ibm->cent_y[elmt])*(c_[2]+c_[3]*ibm->cent_x[elmt]+2*c_[5]*ibm->cent_y[elmt]);
			}

		        netflux = (ibm->F_flux_12[ elmt ] + ibm->F_flux_13[ elmt ] + ibm->F_flux_23[ elmt ])*deltab;
                         
                        if(LiveBed) netflux += w_s * (ibm->C[elmt] - ibm->SCont[elmt]) * ibm->dA[elmt]; 
                    
		        ibm->netflux_old[elmt] = netflux;


                     	dzdt = netflux * ( 1. / ibm->dA[ elmt ] )  / ( 1. - bed_porosity ) + diffuse / (1.0-bed_porosity);
	

			ibm->Delz[ elmt ] = ibm->dtime * dzdt;  

//        		PetscPrintf(PETSC_COMM_WORLD, "dzdt :  %e \n",dzdt);
		       	ibm->cent_z[elmt] += ibm->Delz[elmt];          
                        smoothing_residual= PetscMax(smoothing_residual, fabs(ibm->cent_z[elmt]-ibm->cent_zl[elmt]));

                }
          
	}
	

	/*
	// xiaolei deactivate 
        smoothing_residual=0.0;
        for( elmt = 0; elmt < n_elmt; elmt++ )
	{

		
		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || ibm->Rigidity[elmt]==1)
		{
                        ibm->Delz[elmt]=0.; ibm->netflux_old[elmt]=0.; ibm->cent_z[elmt]=ibm->cent_z_old[elmt];
		}
	      	else
		{
		        netflux = (ibm->F_flux_12[ elmt ] + ibm->F_flux_13[ elmt ] + ibm->F_flux_23[ elmt ])*deltab;
                         
                        if(LiveBed) netflux += w_s * (ibm->C[elmt] - ibm->SCont[elmt]) * ibm->dA[elmt]; 
                    
		        ibm->netflux_old[elmt] = netflux;


                     	dzdt = netflux * ( 1. / ibm->dA[ elmt ] )  / ( 1. - bed_porosity );// + diffuse / (1.0-bed_porosity);
                     	//dzdt = netflux / ( 1. - bed_porosity );// + diffuse / (1.0-bed_porosity);
	

			ibm->Delz[ elmt ] = ibm->dtime * dzdt;

		       	ibm->cent_z[elmt] += ibm->Delz[elmt];          
                        smoothing_residual= PetscMax(smoothing_residual, fabs(ibm->cent_z[elmt]-ibm->cent_zl[elmt]));

                }
          
	}
	*/


 
	// boundary conditions on elevation change // xiaolei add 
	//BC_delz( user, ibm );
 
	// check and correct the new bed elevation based on the Avalanche model

     	int avalanche_check_number = 0;
     	//correct_cell_slope: avalanche_first_sweep( user, ibm );  //xiaolei deactivate
     	//avalanche_second_sweep( user, ibm ); // xiaolei deactivate 

	//PetscReal *cent_dz, *dz;
	
      	//PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(cent_dz));
      	//PetscMalloc(ibm->n_v*sizeof(PetscReal), &(dz));
    
        //for( elmt = 0; elmt < n_elmt; elmt++ ) {
	//	cent_dz[elmt]=ibm->Delz[elmt];
 	//}


	// transfer the bed change from center of elmt to vertices and assign the new z_bp[] valuse
	// xiaolei deactivate 
	//

	if (!LS_bedConstr) {
	for(vert = 0; vert < ibm->n_v; vert++)
	{       
             	Total_Area = 0.;
             	double zb = 0.;
             	riter = 0.;

		// xiaole add
		int ii,jj;
		for(ii=0;ii<100;ii++)
		{
			elmt=ibm->n2c[vert].c[ii];	// xiaolei add
			if (elmt>=0) {
                      		int rigid = ibm->Rigidity[elmt];
		     		if( ibm->nf_z[elmt] < 1.e-6 || ibm->elmt_depth[elmt] > 0.3 || rigid )
				{
                        	}
				else
			      	{
			              	//zb +=  cent_dz[elmt] * ibm->dA[elmt];
			              	zb +=  ibm->cent_z[elmt] * ibm->dA[elmt];
                                      	riter++;  
                                      	Total_Area += ibm->dA[elmt];
			      	}
			}
                }
             	if(riter > 0.) {
         		//if( ibm->x_bp[vert]> 0.5 && ibm->x_bp[vert] < 9.5)  ibm->z_bp[vert] += dz[vert];
         		//if( ibm->x_bp[vert]> 1.0 )  ibm->z_bp[vert] = zb / Total_Area;
         		ibm->z_bp[vert] = zb / Total_Area;
		}
        }
	}

	// least square

	//double a[6][6], b[6], c[6];
	//double xi[100], yi[100], zi[100];
	//int label[100];
	//double _xc, _yc, _zc;

	if (LS_bedConstr) {
	double Good, Number; 
	double Good_max=0.0; 
	Good=0.0;
	Number=0.0;
	//double a[6][6], b[6], c[6];
        for( elmt = 0; elmt < n_elmt; elmt++ ) {
                int rigid = ibm->Rigidity[elmt];
		if(ibm->nf_z[elmt]<1.e-6 || ibm->elmt_depth[elmt]>0.3 || rigid) 
		{
		}
		else {
			n1e = ibm->nv1[ elmt ];
			n2e = ibm->nv2[ elmt ];
			n3e = ibm->nv3[ elmt ];

			N_ac=0;
			_xc=0.0;
			_yc=0.0;
			_zc=0.0;
			int ii,jj, nii;
			for(jj=0;jj<6;jj++) {
				for(ii=0;ii<6;ii++) {
					a[jj][ii]=0.0;			
				}
				b[jj]=0.0;
				c[jj]=0.0;
			}


			xi[0]=ibm->cent_x[elmt];
			yi[0]=ibm->cent_y[elmt];
			zi[0]=ibm->cent_z[elmt];
			_xc=ibm->cent_x[elmt];
			_yc=ibm->cent_y[elmt];
			_zc=ibm->cent_z[elmt];
			N_ac=1;
			label[0]=1;


	
			for(ii=1;ii<100;ii++) {
				xi[ii]=0.0;			
				yi[ii]=0.0;			
				zi[ii]=0.0;			
				label[ii]=0.0;			
				nii=ibm->c2ac[elmt].c[ii];	// xiaolei add
				if (nii>=0) {    // xiaolei add 
                			int rigid = ibm->Rigidity[nii];
					if(ibm->nf_z[nii]<1.e-6 || ibm->elmt_depth[nii]>0.3 || rigid ) 
                	                { 
                        	        }
	                                else
        	                        {
						xi[ii]=ibm->cent_x[nii];
						yi[ii]=ibm->cent_y[nii];
						//zi[ii]=ibm->Delz[nii];
						zi[ii]=ibm->cent_z[nii];
						_xc+=ibm->cent_x[nii];
						_yc+=ibm->cent_y[nii];
						_zc+=ibm->cent_z[nii];
						N_ac+=1;
						label[ii]=1;
					}
				}
			}

			_xc/=double(N_ac);
			_yc/=double(N_ac);
			_zc/=double(N_ac);

			for(ii=0;ii<100;ii++) {
				if (label[ii]) {
					xi[ii]-=_xc;
					yi[ii]-=_yc;
					//zi[ii]-=_zc;
				}
			}


			double rr_max=0;
			for(ii=0;ii<100;ii++) {
				if (label[ii]) {
					wgt[ii]=sqrt(pow(xi[ii]-xi[0],2)+pow(yi[ii]-yi[0],2));
					rr_max=PetscMax(wgt[ii],rr_max);	
				}
			}

			for(ii=0;ii<100;ii++) {
				if (label[ii]) {
					wgt[ii]/=rr_max;
				}
			}

			for(ii=0;ii<100;ii++) {
				if (label[ii]) {
					double rr=wgt[ii];
					wgt[ii]=1.0; //exp(-rr);
				}
			}




	         	/*if( ibm->cent_x[elmt]> 2)*/ {
				if (N_ac<6) {
		         		//ibm->z_bp[n1e] += ibm->Delz[elmt];
         				//ibm->z_bp[n2e] += ibm->Delz[elmt];
		         		//ibm->z_bp[n3e] += ibm->Delz[elmt];
		         		ibm->z_bp[n1e] = _zc;
         				ibm->z_bp[n2e] = _zc;
		         		ibm->z_bp[n3e] = _zc;
					ibm->cent_z[elmt]=_zc;

				} else {
					for(ii=0;ii<100;ii++) {
						if (label[ii]) {
							b[0]+=wgt[ii]*zi[ii];
							b[1]+=wgt[ii]*zi[ii]*xi[ii];
							b[2]+=wgt[ii]*zi[ii]*yi[ii];
							b[3]+=wgt[ii]*zi[ii]*xi[ii]*yi[ii];
							b[4]+=wgt[ii]*zi[ii]*xi[ii]*xi[ii];
							b[5]+=wgt[ii]*zi[ii]*yi[ii]*yi[ii];
							double _a0=wgt[ii]*1, _a1=wgt[ii]*xi[ii], _a2=wgt[ii]*yi[ii], _a3=wgt[ii]*xi[ii]*yi[ii], _a4=wgt[ii]*xi[ii]*xi[ii], _a5=wgt[ii]*yi[ii]*yi[ii];
							a[0][0]+=_a0; a[0][1]+=_a1; a[0][2]+=_a2; a[0][3]+=_a3; a[0][4]+=_a4; a[0][5]+=_a5;
							a[1][0]+=_a0*xi[ii]; a[1][1]+=_a1*xi[ii]; a[1][2]+=_a2*xi[ii]; a[1][3]+=_a3*xi[ii]; a[1][4]+=_a4*xi[ii]; a[1][5]+=_a5*xi[ii];
							a[2][0]+=_a0*yi[ii]; a[2][1]+=_a1*yi[ii]; a[2][2]+=_a2*yi[ii]; a[2][3]+=_a3*yi[ii]; a[2][4]+=_a4*yi[ii]; a[2][5]+=_a5*yi[ii];
							a[3][0]+=_a0*xi[ii]*yi[ii]; a[3][1]+=_a1*xi[ii]*yi[ii]; a[3][2]+=_a2*xi[ii]*yi[ii]; a[3][3]+=_a3*xi[ii]*yi[ii]; a[3][4]+=_a4*xi[ii]*yi[ii]; a[3][5]+=_a5*xi[ii]*yi[ii];
							a[4][0]+=_a0*xi[ii]*xi[ii]; a[4][1]+=_a1*xi[ii]*xi[ii]; a[4][2]+=_a2*xi[ii]*xi[ii]; a[4][3]+=_a3*xi[ii]*xi[ii]; a[4][4]+=_a4*xi[ii]*xi[ii]; a[4][5]+=_a5*xi[ii]*xi[ii];
							a[5][0]+=_a0*yi[ii]*yi[ii]; a[5][1]+=_a1*yi[ii]*yi[ii]; a[5][2]+=_a2*yi[ii]*yi[ii]; a[5][3]+=_a3*yi[ii]*yi[ii]; a[5][4]+=_a4*yi[ii]*yi[ii]; a[5][5]+=_a5*yi[ii]*yi[ii];
						}
					}			

					AGAUS (a, b, c);
		
					double xx, yy;

					xx=ibm->x_bp[n1e]-_xc; yy=ibm->y_bp[n1e]-_yc; 
					ibm->z_bp[n1e]=c[0]+c[1]*xx+c[2]*yy+c[3]*xx*yy+c[4]*xx*xx+c[5]*yy*yy;

					xx=ibm->x_bp[n2e]-_xc; yy=ibm->y_bp[n2e]-_yc; 
					ibm->z_bp[n2e]=c[0]+c[1]*xx+c[2]*yy+c[3]*xx*yy+c[4]*xx*xx+c[5]*yy*yy;

					xx=ibm->x_bp[n3e]-_xc; yy=ibm->y_bp[n3e]-_yc; 
					ibm->z_bp[n3e]=c[0]+c[1]*xx+c[2]*yy+c[3]*xx*yy+c[4]*xx*xx+c[5]*yy*yy;
	

					ibm->cent_z[elmt]=c[0]+c[1]*xi[0]+c[2]*yi[0]+c[3]*xi[0]*yi[0]+c[4]*xi[0]*xi[0]+c[5]*yi[0]*yi[0];

					double zzz=c[0]+c[1]*xi[0]+c[2]*yi[0]+c[3]*xi[0]*yi[0]+c[4]*xi[0]*xi[0]+c[5]*yi[0]*yi[0];
					Good+=pow(zi[0]-zzz,2);
					Good_max=PetscMax(pow(zi[0]-zzz,2),Good_max);
					Number+=1;	
				}


			}

		}
	}

	Good=Good/(Number+1.e-9);
        PetscPrintf(PETSC_COMM_WORLD, "Averaged Goodness of least squares :  %e \n",Good);
        PetscPrintf(PETSC_COMM_WORLD, "Maximum Goodness of least squares :  %e \n",sqrt(Good_max));
	}

        //recomputing_geometry( user, ibm, tistart,ti,itr_sc, avalanche_check_number );

	BC_elevation( user, ibm );

         		
        for( vert= 0; vert < ibm->n_v; vert++ ) {
	//	if( ibm->x_bp[vert]< 1.0 )  ibm->z_bp[vert] = ibm->z_bp_o[vert]  ;  //  xiaolei RTD
	}

        for( elmt= 0; elmt < ibm->n_elmt; elmt++ ) {

	//	if( ibm->cent_x[elmt]< 1.0 )  ibm->cent_z[elmt] = ibm->cent_z_old[elmt]  ;  //  xiaolei RTD
	}

        recomputing_geometry( user, ibm, tistart,ti,itr_sc, avalanche_check_number );
       
        double max_slope = 0.;
        int iel;

        for(iel=0; iel<ibm->n_elmt; iel++)
        {
         max_slope = PetscMax(max_slope, ibm->max_bed_angle[iel]);
        }
        
        avalanche_check_number ++; 


//        if(max_slope > Angle_repose && avalanche_check_number<aval_loop) goto correct_cell_slope; // xiaolei deactivate

        PetscPrintf(PETSC_COMM_WORLD, "smoothing_residual ti itr_sc iteration_number %d %d %d %le\n",ti, itr_sc, iteration, smoothing_residual);
        if(smoothing_residual<1.e-6) SMOOTHING = PETSC_FALSE;
        if(iteration>0) SMOOTHING = PETSC_FALSE;


   }   // end of while SMOOTHIMG
	
  PetscGetTime(&te_4);

  PetscPrintf(PETSC_COMM_WORLD, "#### Smoothing cputime %le\n", te_4-ts_4);


        // computing bed change speed
	for( vert = 0; vert < ibm->n_v; vert++ )
	{
		//ibm->u[ vert ].z = ( ibm->z_bp[ vert ] - ibm->z_bp_old[ vert ] ) / (user->dt);
		ibm->u[ vert ].z = ( ibm->z_bp[ vert ] - ibm->z_bp_o[ vert ] ) /user->dt; // ibm->dtime;
		//ibm->u[ vert ].z = ( ibm->z_bp[ vert ] - ibm->z_bp_o[ vert ] ) / 1.;
		ibm->u[ vert ].x = 0.;                              // ^ above a new change 18 Dec 09
		ibm->u[ vert ].y = 0.;
                if(fabs(fabs(ibm->z_bp[vert] - ibm->z_bp_o[vert]))>1.e-19)
                  {
                   maxz=PetscMax(ibm->z_bp[vert],maxz);
                   minz=PetscMin(ibm->z_bp[vert],minz);
                  }
	}


	
   PetscPrintf(PETSC_COMM_WORLD, "min and max bed elevation at time_step is: %d %e %e\n",ti,minz,maxz);

	// print out for check and debuging  6 nov 2009
	//---------------------------------------

if(ti==tistart || ti == (ti/tiout) * tiout) {  //print out bed elevations for "ibmdata00" and bed cell concentration for "bed_elmt_concntrtn"
  //print out bed elevations for "ibmdata00" and bed cell concentration for "bed_elmt_concntrtn"
 char   ss[20];
 char string[128];
 int ii=0;    
   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  FILE            *fd;
  char            filen[80];
  if(!rank){
     sprintf(filen, "ibmdata%06d.dat",ti);
    fd = fopen(filen, "w");
   

    if (fd) {
          PetscFPrintf(PETSC_COMM_WORLD, fd, "#      bed change trainle mesh\n");
          PetscFPrintf(PETSC_COMM_WORLD, fd, "#      bed change trainle mesh\n");
          PetscFPrintf(PETSC_COMM_WORLD, fd, "#      bed change trainle mesh\n");
        
          PetscFPrintf(PETSC_COMM_WORLD, fd, "%i %i %i %i %i\n",ibm->n_v,ibm->n_elmt,ii,ii,ii);
           
   for (i=0; i<ibm->n_v; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, fd, "%i %le %le %le\n", ii,ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]);
	          	      }
     
   for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, fd, "%i %i %s %i %i %i\n", ii,ii, "hello",ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);

	//fscanf(fd, "%i %i %s %i %i %i\n", ii,ii, ss, nv1+i, nv2+i, nv3+i);
	//nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
      }
//      ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

  //    i=0;
   //   PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", nv1[i], nv2[i], nv3[i]);
       }
      fclose(fd);
    }
/*
  if(!rank){
     
    sprintf(filen,"bed_elmt_concntrtn_%5.5d" , ti-1);
    fd = fopen(filen, "w");

   if (fd) {
   for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, fd, "%e\n",ibm->SCont[i]);
	          	      }
           }
       
      fclose(fd);
    }  */
}

//  PetscPrintf(PETSC_COMM_WORLD, "ti=%d, tistart= %d\n",ti,tistart);
//int tttt=(int)dt_bed;
int tttt=tiout;
if(ti == tistart || ti == (ti/tttt) * tttt) {  //print out bed elevations
  //print out bed elevations
//if((ti==40661) || (ti== 40662)){//|| (ti== 14892) || (ti == 14893)) {  //print out bed elevations
  //PetscPrintf(PETSC_COMM_WORLD, "n_elmt n_v================== %d %d\n",ti,tistart);

 int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  FILE            *f;
  char            filen[80];
  if(rank==1){
     sprintf(filen, "bed_change_%06d.dat",ti);
    f = fopen(filen, "w");
    PetscFPrintf(PETSC_COMM_SELF, f, "Variables=x,y,z,u,v,w,UU,tau0,tau_cr,SCont,flux1,flux2,flux3,delz,cent_z,C,rigidity\n");
    PetscFPrintf(PETSC_COMM_SELF, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-17]=CELLCENTERED)\n", ibm->n_v, ibm->n_elmt);
  //  PetscPrintf(PETSC_COMM_WORLD, "new_bed level\n");
    // x - component
    for (i=0; i<ibm->n_v; i++) {
    PetscFPrintf(PETSC_COMM_SELF, f, "%e \n", ibm->x_bp[i]);
    }
    //PetscFPrintf(PETSC_COMM_SELF, f, "\n");
    // y - component                                                                 
    for (i=0; i<ibm->n_v; i++) {
      PetscFPrintf(PETSC_COMM_SELF, f, "%e \n", ibm->y_bp[i]);
    }
    //PetscFPrintf(PETSC_COMM_SELF, f, "\n");
    // z - component
    for (i=0; i<ibm->n_v; i++) {
      PetscFPrintf(PETSC_COMM_SELF, f, "%e \n", ibm->z_bp[i]);
    }
    
    // PetscFPrintf(PETSC_COMM_SELF, f, "\n");
    
    
    //Write out the Delta_z
    for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->Bvel[i].x);
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->Bvel_u[i]);
      }
    for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->Bvel[i].y);
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->Bvel_v[i]);
      } 
     for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->Bvel[i].z);
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->Bvel_w[i]);
      }
     for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",sqrt(ibm->Bvel[i].x*ibm->Bvel[i].x+ibm->Bvel[i].y*ibm->Bvel[i].y+ibm->Bvel[i].z*ibm->Bvel[i].z));
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",sqrt(ibm->Bvel_u[i]*ibm->Bvel_u[i]+ibm->Bvel_v[i]*ibm->Bvel_v[i]+ibm->Bvel_w[i]*ibm->Bvel_w[i]));
      } 
     for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->BShS[i]);
      } 
     for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->BCShS[i]);
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->c_grad_x[i]);
      } 
      for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->SCont[i]);
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->c_grad_y[i]);
      }
      for (i=0; i<ibm->n_elmt; i++) 
      {
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->u_grad_x[i]);
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->F_flux_12[i]);
      }  
     for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->F_flux_13[i]);
      }  
     for (i=0; i<ibm->n_elmt; i++) 
      {
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->cent_z_AVE[i]);
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->F_flux_23[i]);
      }

     for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->Delz[i]);
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->BCShS[i]);
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->cent_z_AVE[i]);
      }  
     for (i=0; i<ibm->n_elmt; i++) 
      {
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->C[i]);
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->cent_z[i]);
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->max_bed_angle[i]*180./3.1415);
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->elmt_depth[i]);
      }
     for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->C[i]);
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->cent_z[i]);
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->max_bed_angle[i]*180./3.1415);
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->elmt_depth[i]);
      }
     for (i=0; i<ibm->n_elmt; i++) 
      {
	PetscFPrintf(PETSC_COMM_SELF, f, "%d\n",ibm->Rigidity[i]);
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->cent_z[i]);
//	PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->max_bed_angle[i]*180./3.1415);
	//PetscFPrintf(PETSC_COMM_SELF, f, "%e\n",ibm->elmt_depth[i]);
      }



 
    //Write out the link nodes
    for (i=0; i<ibm->n_elmt; i++) 
      {
	
	PetscFPrintf(PETSC_COMM_SELF, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
      }   
    
   
    fclose(f);
  } //end of Comm-wolrd procces
  
} //end if iteration


	for (i=0;i<ibm->n_elmt;i++) {
//
		ibm[0].Bvel[i].x=0.0;
		ibm[0].Bvel[i].y=0.0;
		ibm[0].Bvel[i].z=0.0;
		ibm[0].Shvel[i]=0.0;
		if(LiveBed) ibm[0].C[i]=0.0;

	}

}  // if bed_avg // xiaolei add 

return ( 0 );
}

// xiaolei add
PetscErrorCode bedvel_model( UserCtx * user, IBMNodes * ibm ) 
{

	int i;

	double particle_sg, gi=9.806;
	double fluid_density=1000.0;
	
	particle_sg=particle_dens/fluid_density;

	double sgd_tmp   = ( particle_sg - 1. ) * gi * particle_D50;

	double Rh=0.626; // Hysraulic radius
	double Sf=0.0008;  // friction slope

	double Tks=Rh*Sf/((particle_sg-1)*roughness_size);

	for (i=0;i<ibm->n_elmt;i++) {
		double nfx = ibm->nf_x[ i ];
		double nfy = ibm->nf_y[ i ];
		double nfz = ibm->nf_z[ i ];

		if( nfz < 1.e-6 || ibm->elmt_depth[i] > 0.3 || ibm->Rigidity[i])
		{

		}
		else {
			double vel_mag=sqrt(ibm[0].Bvel[i].x*ibm[0].Bvel[i].x+ibm[0].Bvel[i].y*ibm[0].Bvel[i].y+ibm[0].Bvel[i].z*ibm[0].Bvel[i].z);

			double nx=ibm[0].Bvel[i].x/(vel_mag+1.e-19);
			double ny=ibm[0].Bvel[i].y/(vel_mag+1.e-19);
			double nz=ibm[0].Bvel[i].z/(vel_mag+1.e-19);

			// model 1
			//
			double Vel;
			if (particlevel_model==1) {
				double coef1=sqrt(PetscMax((ibm->BShS[i]-ibm->BCShS[i]),0.0)/(ibm->BCShS[i]+1.e-19));
				double coef2=ibm->BShS[i]/(ibm->BCShS[i]+1.e-19);
				Vel=(4.2*coef1/(coef2+1.e-19)+2.4)*ibm[0].Shvel[i];
				if (ibm->BShS[i]-ibm->BCShS[i]<0.0) Vel=0.0;			
			} else if (particlevel_model==2) {
			// model 2
				double BCvel=sqrt(ibm->BCShS[i]*sgd_tmp);
				Vel=PetscMax(12.0*(ibm[0].Shvel[i]-BCvel),0.0);
			} else if (particlevel_model==3) {
			// model 3
				Vel=ibm[0].Shvel[i]*(3.3*log(Tks)+17.7);
			}
	
			ibm[0].Bvel[i].x=Vel*nx;
			ibm[0].Bvel[i].y=Vel*ny;
			ibm[0].Bvel[i].z=0.0; //Vel*nz;

		}
	}

        return false;
}


// xiaolei add SEDI
PetscErrorCode Connectivity_ib( UserCtx * user, IBMNodes * ibm ) 
{
	int i,j;
	int n1e,n2e,n3e;
	int _n1e,_n2e,_n3e;

	PetscPrintf(PETSC_COMM_WORLD, "c2c connectivity \n");
	for(i=0;i<ibm->n_elmt;i++) {
		n1e=ibm->nv1[i];	
		n2e=ibm->nv2[i];	
		n3e=ibm->nv3[i];	
		
		ibm->c2c[i].c1=-100;
		ibm->c2c[i].c2=-100;
		ibm->c2c[i].c3=-100;


		for(j=0;j<ibm->n_elmt;j++) {
			_n1e=ibm->nv1[j];
			_n2e=ibm->nv2[j];
			_n3e=ibm->nv3[j];

			if (j!=i && (_n1e==n1e || _n2e==n1e || _n3e==n1e) &&  (_n1e==n2e || _n2e==n2e || _n3e==n2e) ) ibm->c2c[i].c1=j; 
			if (j!=i && (_n1e==n1e || _n2e==n1e || _n3e==n1e) &&  (_n1e==n3e || _n2e==n3e || _n3e==n3e) ) ibm->c2c[i].c2=j; 
			if (j!=i && (_n1e==n2e || _n2e==n2e || _n3e==n2e) &&  (_n1e==n3e || _n2e==n3e || _n3e==n3e) ) ibm->c2c[i].c3=j; 

		}

	}

	PetscPrintf(PETSC_COMM_WORLD, "n2c connectivity \n");

	for(i=0;i<ibm->n_v;i++) {

		int iii=0;
		int ii;
		
		for(ii=0;ii<100;ii++) {
			ibm->n2c[i].c[ii]=-100;
		}

		for(j=0;j<ibm->n_elmt;j++) {
			_n1e=ibm->nv1[j];
			_n2e=ibm->nv2[j];
			_n3e=ibm->nv3[j];

			if (_n1e==i || _n2e==i || _n3e==i) {
				ibm->n2c[i].c[iii]=j;
				iii++; 
				if (iii>99) {
					PetscPrintf(PETSC_COMM_WORLD, "The numbder of cells sharing the node %d is larger than 100\n",i);
					exit(1);
				}
			}

		}

	}


	PetscPrintf(PETSC_COMM_WORLD, "c2ac connectivity \n");

	for(i=0;i<ibm->n_elmt;i++) {

		n1e=ibm->nv1[i];	
		n2e=ibm->nv2[i];	
		n3e=ibm->nv3[i];	
	
		int iii=0;
		int ii;
		
		for(ii=0;ii<100;ii++) {
			ibm->c2ac[i].c[ii]=-100;
		}

		for(j=0;j<ibm->n_elmt;j++) {
			_n1e=ibm->nv1[j];
			_n2e=ibm->nv2[j];
			_n3e=ibm->nv3[j];

			if (j!=i &&  (_n1e==n1e || _n1e==n2e || _n1e==n3e || _n2e==n1e || _n2e==n2e || _n2e==n3e || _n3e==n1e || _n3e==n2e || _n3e==n3e) ) {
				ibm->c2ac[i].c[iii]=j;
				iii++; 
				if (iii>99) { 
					PetscPrintf(PETSC_COMM_WORLD, "The numbder of cells sharing the node of cell %d is larger than 100\n",i);
					exit(1);
				}
			}

		}

	}


	PetscPrintf(PETSC_COMM_WORLD, "finish connectivity \n");

        return false;
}






