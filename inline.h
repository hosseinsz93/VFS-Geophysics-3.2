//#include <variables.h>

#ifndef INLINE_FUNCTION
#define INLINE_FUNCTION

inline double M(double a, double b)
{
  if(fabs(a)<fabs(b)) return a;
  else return b;
}

inline void get_weight ( int i, int j, int k, int mx, int my, int mz, PetscReal ***aj, PetscReal ***nvert, PetscReal nv, double weight[3][3][3])
{
		for(int p=-1; p<=1; p++)
		for(int q=-1; q<=1; q++)
		for(int r=-1; r<=1; r++) {
			int R=r+1, Q=q+1, P=p+1;
			int K=k+r, J=j+q, I=i+p;

			if( nvert[K][J][I]>nv ) weight[R][Q][P]=0;
			else weight[R][Q][P] = 1./aj[K][J][I];
		}
}

inline double minmod(double m1, double m2)	// for UNO
{
	return 0.5 * ( sign(m1)+sign(m2) ) * std::min ( fabs(m1), fabs(m2) );
}

inline double eno2(double f0, double f1, double f2, double f3, double a)
{
	if( a > 0 ) return ( f1 + 0.5*M(f2-f1, f1-f0) );
	else  return ( f2 - 0.5*M(f3-f2, f2-f1) );
};


inline double weno3(double f0, double f1, double f2, double f3, double wavespeed)
{
	double fL, fC, fR;
	
	if(wavespeed>0)  {
		fL = f0; 
		fC = f1; 
		fR = f2;
	}
	else {
		// mirror
		fL = f3; 
		fC = f2; 
		fR = f1; 
	}
	
	double d0=2./3., d1=1./3.;	// weno3
	//if(wavespeed<=0) d0=1./3., d1=2./3.;
	
	const double eps=1.e-6;
		
	double beta0 = pow( fC - fR,  2. );
	double beta1 = pow( fL - fC,  2. );
	
	double alpha0 = d0 / pow( eps + beta0, 2. );
	double alpha1 = d1 / pow( eps + beta1, 2. );

	double sumalpha = alpha0 + alpha1;
	
	double w0 = alpha0 / sumalpha;
	double w1 = alpha1 / sumalpha;
	
	double u0 = ( fC*0.5  + fR*0.5 );
	double u1 = ( -fL*0.5 + fC*1.5 );
	
	return w0*u0 + w1*u1;
};

/*
High Order Finite Difference and Finite Volume
WENO Schemes and Discontinuous Galerkin Methods
for CFD, by Shu, ICASE
*/
inline double weno5(double f0, double f1, double f2, double f3, double f4, double f5, double wavespeed)
{
	double A, B, C, D, E;
	
	if(wavespeed>0) {
		A = f0;
		B = f1;
		C = f2;
		D = f3;
		E = f4;
	}
	else {
		// mirror
		A = f5;
		B = f4;
		C = f3;
		D = f2;
		E = f1;
	}
	
	double eps = 1.e-6;// * std::max ( A*A, std::max ( B*B, std::max ( C*C, std::max ( D*D, E*E ) ) ) ) + 1.e-99;
	double d0, d1, d2;
	
	//if(wavespeed<=0) d0=0.1, d1=0.6, d2=0.3;
	//else 
	  d0=0.3, d1=0.6, d2=0.1;
	
	double beta0 = 13./12. * pow( A - 2. * B + C, 2. ) + 1./4. * pow ( A - 4. * B  + 3. * C, 2. );
	double beta1 = 13./12. * pow( B - 2. * C + D, 2. ) + 1./4. * pow ( B - D, 2. );
	double beta2 = 13./12. * pow( C - 2. * D + E, 2. ) + 1./4. * pow ( 3. * C - 4. * D  + E, 2. );
	
	double alpha0 = d0 / pow( eps + beta0, 2.);
	double alpha1 = d1 / pow( eps + beta1, 2.);
	double alpha2 = d2 / pow( eps + beta2, 2.);
	
	double sumalpha = alpha0 + alpha1 + alpha2;
		
	double w0 = alpha0 / sumalpha;
	double w1 = alpha1 / sumalpha;
	double w2 = alpha2 / sumalpha;
	
	double u0 = 2./6. * A - 7./6. * B + 11./6. * C;
	double u1 = -1./6. * B + 5./6. * C + 2./6. * D;
	double u2 = 2./6. * C + 5./6. * E - 1./6. * E;
	
	return w0*u0 + w1*u1 + w2*u2;
	
};

inline double levelset_thickness(double aj, Cmpnts csi, Cmpnts eta, Cmpnts zet)
{
	double dx = 1./aj/sqrt( eta.x*eta.x + eta.y*eta.y + eta.z*eta.z );
	
	if(dthick_set) dx*=dthick;
	if(dthick_const>0) dx = dthick_const;
	
	return dx;
}

// xiaolei add for distributing the force at zero level 
inline double DDelta(double p, double dx)
{
	double eps;
	eps = dx;

	double r=p/dx;

  	if (fabs(r) <= 1.0) {
    		return 1.0;
  	} 
	else if (fabs(r) >=1.0 && fabs(r) <= 2){
		return (2-fabs(r));
	} 
	else {
    		return 0.0;
  	}

	/*
  	if (fabs(r) <= 1.0) {
    		return 17.0/48.0 + sqrt(3.0)*3.14159265/108.0 + fabs(r)/4.0 - r*r/4.0 + (1.0-2.0*fabs(r))*sqrt(-12.0*r*r+12.0*fabs(r)+1.0)/16.0 - sqrt(3.0)*asin(sqrt(3.0)*(2.0*fabs(r)-1.0)/2.0)/12.0;
  	} else if (fabs(r) >= 1.0 && fabs(r) <= 2.0) {
    		return 55.0/48.0 - sqrt(3.0)*3.14159265/108.0 - 13.0*fabs(r)/12.0 + r*r/4.0 + (2.0*fabs(r)-3.0)*sqrt(-12.0*r*r+36.0*fabs(r)-23.0)/48.0 + sqrt(3.0)*asin(sqrt(3.0)*(2.0*fabs(r)-3.0)/2.0)/36.0;
  	} else { 
    		return 0.0;
  	}
	*/
	
}


inline double H_1 (double p, double dx)
{
	double eps;
	
	eps = dx;
 
	if ( p < 0.0 ) return 0;
	else return 1;
};




inline double H (double p, double dx)
{
	double eps;
	
	eps = dx;
 
	if ( p < -eps ) return 0;  
	else if ( fabs(p) <= eps ) return 0.5 * ( 1.0 + p/eps + 1./M_PI * sin ( M_PI * p / eps ) ); 
	else return 1;
};

inline double dH (double p, double dx)
{
	double eps;
	
	eps = dx;
	
	if ( p < -eps ) return 0;
	else if ( fabs(p) <= eps ) return 0.5 / eps * ( 1.0 + cos ( M_PI * p / eps ) );
	else return 0;
};

// xiaolei add
inline double dH_1 (double p, double dx)
{
	double eps;
	
	eps = dx;
	
	if ( p < -eps ) return 0;
	else if ( fabs(p) <= eps ) return 1.0;
	else return 0;
};


// Puckett
inline double mean ( double A, double B )
{
	return 2.0 * A * B / ( A + B );
	//return 0.5 * ( A + B );
}

inline double sign1(double a, double dx)
{
  //return sign(a);
  	return 2.0 * ( H (a, dx) - 0.5 );
};

// Yue & Patel
inline double mod_sign(double d0, double grad0, double e)
{
	return d0 / sqrt ( d0*d0 + pow(grad0*e, 2.0) );
}

inline void Compute_du_center (int i, int j, int k,  int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
{
	if ((nvert[k][j][i+1])> 0.1 || (!i_periodic &&  !ii_periodic && i==mx-2) ) {
		*dudc = ( ucat[k][j][i].x - ucat[k][j][i-1].x );
		*dvdc = ( ucat[k][j][i].y - ucat[k][j][i-1].y );
		*dwdc = ( ucat[k][j][i].z - ucat[k][j][i-1].z );
	}
	else if ((nvert[k][j][i-1])> 0.1 || (!i_periodic &&  !ii_periodic && i==1) ) {
		*dudc = ( ucat[k][j][i+1].x - ucat[k][j][i].x );
		*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][i].y );
		*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][i].z );
	}
	else {
		/*if(i_periodic && i==1) {
			*dudc = ( ucat[k][j][i+1].x - ucat[k][j][mx-2].x ) * 0.5;
			*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][mx-2].y ) * 0.5;
			*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][mx-2].z ) * 0.5;
		}
		else if(i_periodic && i==mx-2) {
			*dudc = ( ucat[k][j][1].x - ucat[k][j][i-1].x ) * 0.5;
			*dvdc = ( ucat[k][j][1].y - ucat[k][j][i-1].y ) * 0.5;
			*dwdc = ( ucat[k][j][1].z - ucat[k][j][i-1].z ) * 0.5;
		}
		else if(ii_periodic && i==1) {
			*dudc = ( ucat[k][j][i+1].x - ucat[k][j][-2].x ) * 0.5;
			*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][-2].y ) * 0.5;
			*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][-2].z ) * 0.5;
		}
		else if(ii_periodic && i==mx-2) {
			*dudc = ( ucat[k][j][mx+1].x - ucat[k][j][i-1].x ) * 0.5;
			*dvdc = ( ucat[k][j][mx+1].y - ucat[k][j][i-1].y ) * 0.5;
			*dwdc = ( ucat[k][j][mx+1].z - ucat[k][j][i-1].z ) * 0.5;
		}
		else*/ {
			*dudc = ( ucat[k][j][i+1].x - ucat[k][j][i-1].x ) * 0.5;
			*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][i-1].y ) * 0.5;
			*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][i-1].z ) * 0.5;
		}
	}

	if ((nvert[k][j+1][i])> 0.1 || (!j_periodic &&  !jj_periodic && j==my-2) ) {
		*dude = ( ucat[k][j][i].x - ucat[k][j-1][i].x );
		*dvde = ( ucat[k][j][i].y - ucat[k][j-1][i].y );
		*dwde = ( ucat[k][j][i].z - ucat[k][j-1][i].z );
	}
	else if ((nvert[k][j-1][i])> 0.1 || (!j_periodic &&  !jj_periodic && j==1) ) {
		*dude = ( ucat[k][j+1][i].x - ucat[k][j][i].x );
		*dvde = ( ucat[k][j+1][i].y - ucat[k][j][i].y );
		*dwde = ( ucat[k][j+1][i].z - ucat[k][j][i].z );
	}
	else {
		/*if(j_periodic && j==1) {
			*dude = ( ucat[k][j+1][i].x - ucat[k][my-2][i].x ) * 0.5;
			*dvde = ( ucat[k][j+1][i].y - ucat[k][my-2][i].y ) * 0.5;
			*dwde = ( ucat[k][j+1][i].z - ucat[k][my-2][i].z ) * 0.5;
		}
		else if(j_periodic && j==my-2) {
			*dude = ( ucat[k][1][i].x - ucat[k][j-1][i].x ) * 0.5;
			*dvde = ( ucat[k][1][i].y - ucat[k][j-1][i].y ) * 0.5;
			*dwde = ( ucat[k][1][i].z - ucat[k][j-1][i].z ) * 0.5;
		}
		else if(jj_periodic && j==1) {
			*dude = ( ucat[k][j+1][i].x - ucat[k][-2][i].x ) * 0.5;
			*dvde = ( ucat[k][j+1][i].y - ucat[k][-2][i].y ) * 0.5;
			*dwde = ( ucat[k][j+1][i].z - ucat[k][-2][i].z ) * 0.5;
		}
		else if(jj_periodic && j==my-2) {
			*dude = ( ucat[k][my+1][i].x - ucat[k][j-1][i].x ) * 0.5;
			*dvde = ( ucat[k][my+1][i].y - ucat[k][j-1][i].y ) * 0.5;
			*dwde = ( ucat[k][my+1][i].z - ucat[k][j-1][i].z ) * 0.5;
		}
		else */{
			*dude = ( ucat[k][j+1][i].x - ucat[k][j-1][i].x ) * 0.5;
			*dvde = ( ucat[k][j+1][i].y - ucat[k][j-1][i].y ) * 0.5;
			*dwde = ( ucat[k][j+1][i].z - ucat[k][j-1][i].z ) * 0.5;
		}
	}

	if ((nvert[k+1][j][i])> 0.1 || ( !k_periodic &&  !kk_periodic && k==mz-2) ) {
		*dudz = ( ucat[k][j][i].x - ucat[k-1][j][i].x );
		*dvdz = ( ucat[k][j][i].y - ucat[k-1][j][i].y );
		*dwdz = ( ucat[k][j][i].z - ucat[k-1][j][i].z );
	}
	else if ((nvert[k-1][j][i])> 0.1 || (!k_periodic &&  !kk_periodic && k==0) ) {
		*dudz = ( ucat[k+1][j][i].x - ucat[k][j][i].x );
		*dvdz = ( ucat[k+1][j][i].y - ucat[k][j][i].y );
		*dwdz = ( ucat[k+1][j][i].z - ucat[k][j][i].z );
	}
	else {
		/*if(k_periodic && k==1) {
			*dudz = ( ucat[k+1][j][i].x - ucat[mz-2][j][i].x ) * 0.5;
			*dvdz = ( ucat[k+1][j][i].y - ucat[mz-2][j][i].y ) * 0.5;
			*dwdz = ( ucat[k+1][j][i].z - ucat[mz-2][j][i].z ) * 0.5;
		}
		else if(k_periodic && k==mz-2) {
			*dudz = ( ucat[1][j][i].x - ucat[k-1][j][i].x ) * 0.5;
			*dvdz = ( ucat[1][j][i].y - ucat[k-1][j][i].y ) * 0.5;
			*dwdz = ( ucat[1][j][i].z - ucat[k-1][j][i].z ) * 0.5;
		}
		else if(kk_periodic && k==1) {
			*dudz = ( ucat[k+1][j][i].x - ucat[-2][j][i].x ) * 0.5;
			*dvdz = ( ucat[k+1][j][i].y - ucat[-2][j][i].y ) * 0.5;
			*dwdz = ( ucat[k+1][j][i].z - ucat[-2][j][i].z ) * 0.5;
		}
		else if(kk_periodic && k==mz-2) {
			*dudz = ( ucat[mz+1][j][i].x - ucat[k-1][j][i].x ) * 0.5;
			*dvdz = ( ucat[mz+1][j][i].y - ucat[k-1][j][i].y ) * 0.5;
			*dwdz = ( ucat[mz+1][j][i].z - ucat[k-1][j][i].z ) * 0.5;
		}
		else*/ {
			*dudz = ( ucat[k+1][j][i].x - ucat[k-1][j][i].x ) * 0.5;
			*dvdz = ( ucat[k+1][j][i].y - ucat[k-1][j][i].y ) * 0.5;
			*dwdz = ( ucat[k+1][j][i].z - ucat[k-1][j][i].z ) * 0.5;
		}
	}
}

inline void Compute_du_dxyz (	double csi0, double csi1, double csi2, double eta0, double eta1, double eta2, double zet0, double zet1, double zet2, double ajc,
					double dudc, double dvdc, double dwdc, double dude, double dvde, double dwde, double dudz, double dvdz, double dwdz,
					double *du_dx, double *dv_dx, double *dw_dx, double *du_dy, double *dv_dy, double *dw_dy, double *du_dz, double *dv_dz, double *dw_dz )
{
	*du_dx = (dudc * csi0 + dude * eta0 + dudz * zet0) * ajc;
	*du_dy = (dudc * csi1 + dude * eta1 + dudz * zet1) * ajc;
	*du_dz = (dudc * csi2 + dude * eta2 + dudz * zet2) * ajc;
	*dv_dx = (dvdc * csi0 + dvde * eta0 + dvdz * zet0) * ajc;
	*dv_dy = (dvdc * csi1 + dvde * eta1 + dvdz * zet1) * ajc;
	*dv_dz = (dvdc * csi2 + dvde * eta2 + dvdz * zet2) * ajc;
	*dw_dx = (dwdc * csi0 + dwde * eta0 + dwdz * zet0) * ajc;
	*dw_dy = (dwdc * csi1 + dwde * eta1 + dwdz * zet1) * ajc;	
	*dw_dz = (dwdc * csi2 + dwde * eta2 + dwdz * zet2) * ajc;
};

inline void Compute_dkdo_dxyz (	double csi0, double csi1, double csi2, double eta0, double eta1, double eta2, double zet0, double zet1, double zet2, double ajc,
					double dkdc, double dodc, double dkde, double dode, double dkdz, double dodz,
					double *dk_dx, double *do_dx, double *dk_dy, double *do_dy, double *dk_dz, double *do_dz)
{
	*dk_dx = (dkdc * csi0 + dkde * eta0 + dkdz * zet0) * ajc;
	*dk_dy = (dkdc * csi1 + dkde * eta1 + dkdz * zet1) * ajc;
	*dk_dz = (dkdc * csi2 + dkde * eta2 + dkdz * zet2) * ajc;
	*do_dx = (dodc * csi0 + dode * eta0 + dodz * zet0) * ajc;
	*do_dy = (dodc * csi1 + dode * eta1 + dodz * zet1) * ajc;
	*do_dz = (dodc * csi2 + dode * eta2 + dodz * zet2) * ajc;
};

inline void Compute_dscalar_dxyz ( double csi0, double csi1, double csi2, double eta0, double eta1, double eta2, double zet0, double zet1, double zet2, double ajc,
							double dkdc, double dkde, double dkdz, double *dk_dx, double *dk_dy, double *dk_dz)
{
	*dk_dx = (dkdc * csi0 + dkde * eta0 + dkdz * zet0) * ajc;
	*dk_dy = (dkdc * csi1 + dkde * eta1 + dkdz * zet1) * ajc;
	*dk_dz = (dkdc * csi2 + dkde * eta2 + dkdz * zet2) * ajc;
};

inline void Compute_du_dxyz ( int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, Cmpnts ***csi, Cmpnts ***eta, Cmpnts ***zet, PetscReal ***aj, 
		       double *du_dx, double *du_dy, double *du_dz, double *dv_dx, double *dv_dy, double *dv_dz, double *dw_dx, double *dw_dy, double *dw_dz)
{
  double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
  double eta0 = eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
  double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
  double ajc = aj[k][j][i];
  
  double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
  
  Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
  Compute_du_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz );
}

inline void Compute_du_i (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
{
	*dudc = ucat[k][j][i+1].x - ucat[k][j][i].x;
	*dvdc = ucat[k][j][i+1].y - ucat[k][j][i].y;
	*dwdc = ucat[k][j][i+1].z - ucat[k][j][i].z;
	
	if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j+1][i+1])> 0.1) {
		*dude = (ucat[k][j  ][i+1].x + ucat[k][j  ][i].x - ucat[k][j-1][i+1].x - ucat[k][j-1][i].x) * 0.5;
		*dvde = (ucat[k][j  ][i+1].y + ucat[k][j  ][i].y - ucat[k][j-1][i+1].y - ucat[k][j-1][i].y) * 0.5;
		*dwde = (ucat[k][j  ][i+1].z + ucat[k][j  ][i].z - ucat[k][j-1][i+1].z - ucat[k][j-1][i].z) * 0.5;
	}
	else if  ((nvert[k][j-1][i])> 0.1 || (nvert[k][j-1][i+1])> 0.1) {
		*dude = (ucat[k][j+1][i+1].x + ucat[k][j+1][i].x - ucat[k][j  ][i+1].x - ucat[k][j  ][i].x) * 0.5;
		*dvde = (ucat[k][j+1][i+1].y + ucat[k][j+1][i].y - ucat[k][j  ][i+1].y - ucat[k][j  ][i].y) * 0.5;
		*dwde = (ucat[k][j+1][i+1].z + ucat[k][j+1][i].z - ucat[k][j  ][i+1].z - ucat[k][j  ][i].z) * 0.5;
	}
	else {
		*dude = (ucat[k][j+1][i+1].x + ucat[k][j+1][i].x - ucat[k][j-1][i+1].x - ucat[k][j-1][i].x) * 0.25;
		*dvde = (ucat[k][j+1][i+1].y + ucat[k][j+1][i].y - ucat[k][j-1][i+1].y - ucat[k][j-1][i].y) * 0.25;
		*dwde = (ucat[k][j+1][i+1].z + ucat[k][j+1][i].z - ucat[k][j-1][i+1].z - ucat[k][j-1][i].z) * 0.25;
	}

	if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j][i+1])> 0.1) {
		*dudz = (ucat[k  ][j][i+1].x + ucat[k  ][j][i].x - ucat[k-1][j][i+1].x - ucat[k-1][j][i].x) * 0.5;
		*dvdz = (ucat[k  ][j][i+1].y + ucat[k  ][j][i].y - ucat[k-1][j][i+1].y - ucat[k-1][j][i].y) * 0.5;
		*dwdz = (ucat[k  ][j][i+1].z + ucat[k  ][j][i].z - ucat[k-1][j][i+1].z - ucat[k-1][j][i].z) * 0.5;
	}
	else if ((nvert[k-1][j][i])> 0.1 || (nvert[k-1][j][i+1])> 0.1) {
		*dudz = (ucat[k+1][j][i+1].x + ucat[k+1][j][i].x - ucat[k  ][j][i+1].x - ucat[k  ][j][i].x) * 0.5;
		*dvdz = (ucat[k+1][j][i+1].y + ucat[k+1][j][i].y - ucat[k  ][j][i+1].y - ucat[k  ][j][i].y) * 0.5;
		*dwdz = (ucat[k+1][j][i+1].z + ucat[k+1][j][i].z - ucat[k  ][j][i+1].z - ucat[k  ][j][i].z) * 0.5;
	}
	else {
		*dudz = (ucat[k+1][j][i+1].x + ucat[k+1][j][i].x - ucat[k-1][j][i+1].x - ucat[k-1][j][i].x) * 0.25;
		*dvdz = (ucat[k+1][j][i+1].y + ucat[k+1][j][i].y - ucat[k-1][j][i+1].y - ucat[k-1][j][i].y) * 0.25;
		*dwdz = (ucat[k+1][j][i+1].z + ucat[k+1][j][i].z - ucat[k-1][j][i+1].z - ucat[k-1][j][i].z) * 0.25;
	}
}

inline void Compute_dscalar_i (int i, int j, int k, int mx, int my, int mz, PetscReal ***K, PetscReal ***nvert, double *dkdc, double *dkde, double *dkdz )
{
	
	*dkdc = K[k][j][i+1] - K[k][j][i];
	
	
	if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j+1][i+1])> 0.1) {
		*dkde = (K[k][j  ][i+1] + K[k][j  ][i] - K[k][j-1][i+1] - K[k][j-1][i]) * 0.5;
	}
	else if  ((nvert[k][j-1][i])> 0.1 || (nvert[k][j-1][i+1])> 0.1) {
		*dkde = (K[k][j+1][i+1] + K[k][j+1][i] - K[k][j  ][i+1] - K[k][j  ][i]) * 0.5;
	}
	else {
		*dkde = (K[k][j+1][i+1] + K[k][j+1][i] - K[k][j-1][i+1] - K[k][j-1][i]) * 0.25;
	}

	if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j][i+1])> 0.1) {
		*dkdz = (K[k  ][j][i+1] + K[k  ][j][i] - K[k-1][j][i+1] - K[k-1][j][i]) * 0.5;
	}
	else if ((nvert[k-1][j][i])> 0.1 || (nvert[k-1][j][i+1])> 0.1) {
		*dkdz = (K[k+1][j][i+1] + K[k+1][j][i] - K[k  ][j][i+1] - K[k  ][j][i]) * 0.5;
	}
	else {
		*dkdz = (K[k+1][j][i+1] + K[k+1][j][i] - K[k-1][j][i+1] - K[k-1][j][i]) * 0.25;
	}
}

inline void Compute_du_j (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
				
{
	if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j+1][i+1])> 0.1) {
		*dudc = (ucat[k][j+1][i  ].x + ucat[k][j][i  ].x - ucat[k][j+1][i-1].x - ucat[k][j][i-1].x) * 0.5;
		*dvdc = (ucat[k][j+1][i  ].y + ucat[k][j][i  ].y - ucat[k][j+1][i-1].y - ucat[k][j][i-1].y) * 0.5;
		*dwdc = (ucat[k][j+1][i  ].z + ucat[k][j][i  ].z - ucat[k][j+1][i-1].z - ucat[k][j][i-1].z) * 0.5;
	}
	else if ((nvert[k][j][i-1])> 0.1 || (nvert[k][j+1][i-1])> 0.1) {
		*dudc = (ucat[k][j+1][i+1].x + ucat[k][j][i+1].x - ucat[k][j+1][i  ].x - ucat[k][j][i  ].x) * 0.5;
		*dvdc = (ucat[k][j+1][i+1].y + ucat[k][j][i+1].y - ucat[k][j+1][i  ].y - ucat[k][j][i  ].y) * 0.5;
		*dwdc = (ucat[k][j+1][i+1].z + ucat[k][j][i+1].z - ucat[k][j+1][i  ].z - ucat[k][j][i  ].z) * 0.5;
	}
	else {
		*dudc = (ucat[k][j+1][i+1].x + ucat[k][j][i+1].x - ucat[k][j+1][i-1].x - ucat[k][j][i-1].x) * 0.25;
		*dvdc = (ucat[k][j+1][i+1].y + ucat[k][j][i+1].y - ucat[k][j+1][i-1].y - ucat[k][j][i-1].y) * 0.25;
		*dwdc = (ucat[k][j+1][i+1].z + ucat[k][j][i+1].z - ucat[k][j+1][i-1].z - ucat[k][j][i-1].z) * 0.25;
	}
	
	*dude = ucat[k][j+1][i].x - ucat[k][j][i].x;
	*dvde = ucat[k][j+1][i].y - ucat[k][j][i].y;
	*dwde = ucat[k][j+1][i].z - ucat[k][j][i].z;
	
	if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j+1][i])> 0.1) {
		*dudz = (ucat[k  ][j+1][i].x + ucat[k  ][j][i].x - ucat[k-1][j+1][i].x - ucat[k-1][j][i].x) * 0.5;
		*dvdz = (ucat[k  ][j+1][i].y + ucat[k  ][j][i].y - ucat[k-1][j+1][i].y - ucat[k-1][j][i].y) * 0.5;
		*dwdz = (ucat[k  ][j+1][i].z + ucat[k  ][j][i].z - ucat[k-1][j+1][i].z - ucat[k-1][j][i].z) * 0.5;
	}
	else if ((nvert[k-1][j][i])> 0.1 || (nvert[k-1][j+1][i])> 0.1) {
		*dudz = (ucat[k+1][j+1][i].x + ucat[k+1][j][i].x - ucat[k  ][j+1][i].x - ucat[k  ][j][i].x) * 0.5;
		*dvdz = (ucat[k+1][j+1][i].y + ucat[k+1][j][i].y - ucat[k  ][j+1][i].y - ucat[k  ][j][i].y) * 0.5;
		*dwdz = (ucat[k+1][j+1][i].z + ucat[k+1][j][i].z - ucat[k  ][j+1][i].z - ucat[k  ][j][i].z) * 0.5;
	}
	else {
		*dudz = (ucat[k+1][j+1][i].x + ucat[k+1][j][i].x - ucat[k-1][j+1][i].x - ucat[k-1][j][i].x) * 0.25;
		*dvdz = (ucat[k+1][j+1][i].y + ucat[k+1][j][i].y - ucat[k-1][j+1][i].y - ucat[k-1][j][i].y) * 0.25;
		*dwdz = (ucat[k+1][j+1][i].z + ucat[k+1][j][i].z - ucat[k-1][j+1][i].z - ucat[k-1][j][i].z) * 0.25;
	}
};

inline void Compute_dscalar_j (int i, int j, int k, int mx, int my, int mz, PetscReal ***K, PetscReal ***nvert, double *dkdc, double *dkde, double *dkdz )
				
{
	if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j+1][i+1])> 0.1) {
		*dkdc = (K[k][j+1][i  ] + K[k][j][i  ] - K[k][j+1][i-1] - K[k][j][i-1]) * 0.5;
	}
	else if ((nvert[k][j][i-1])> 0.1 || (nvert[k][j+1][i-1])> 0.1) {
		*dkdc = (K[k][j+1][i+1] + K[k][j][i+1] - K[k][j+1][i  ] - K[k][j][i  ]) * 0.5;
	}
	else {
		*dkdc = (K[k][j+1][i+1] + K[k][j][i+1] - K[k][j+1][i-1] - K[k][j][i-1]) * 0.25;
	}

	*dkde = K[k][j+1][i] - K[k][j][i];
	
	if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j+1][i])> 0.1) {
		*dkdz = (K[k  ][j+1][i] + K[k  ][j][i] - K[k-1][j+1][i] - K[k-1][j][i]) * 0.5;
	}
	else if ((nvert[k-1][j][i])> 0.1 || (nvert[k-1][j+1][i])> 0.1) {
		*dkdz = (K[k+1][j+1][i] + K[k+1][j][i] - K[k  ][j+1][i] - K[k  ][j][i]) * 0.5;
	}
	else {
		*dkdz = (K[k+1][j+1][i] + K[k+1][j][i] - K[k-1][j+1][i] - K[k-1][j][i]) * 0.25;
	}
};

inline void Compute_du_k (int i, int j, int k, int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
				
{
	if ((nvert[k][j][i+1])> 0.1 || (nvert[k+1][j][i+1])> 0.1) {
		*dudc = (ucat[k+1][j][i  ].x + ucat[k][j][i  ].x - ucat[k+1][j][i-1].x - ucat[k][j][i-1].x) * 0.5;
		*dvdc = (ucat[k+1][j][i  ].y + ucat[k][j][i  ].y - ucat[k+1][j][i-1].y - ucat[k][j][i-1].y) * 0.5;
		*dwdc = (ucat[k+1][j][i  ].z + ucat[k][j][i  ].z - ucat[k+1][j][i-1].z - ucat[k][j][i-1].z) * 0.5;
	}
	else if ((nvert[k][j][i-1])> 0.1 || (nvert[k+1][j][i-1])> 0.1) {
		*dudc = (ucat[k+1][j][i+1].x + ucat[k][j][i+1].x - ucat[k+1][j][i  ].x - ucat[k][j][i  ].x) * 0.5;
		*dvdc = (ucat[k+1][j][i+1].y + ucat[k][j][i+1].y - ucat[k+1][j][i  ].y - ucat[k][j][i  ].y) * 0.5;
		*dwdc = (ucat[k+1][j][i+1].z + ucat[k][j][i+1].z - ucat[k+1][j][i  ].z - ucat[k][j][i  ].z) * 0.5;
	}
	else {
		*dudc = (ucat[k+1][j][i+1].x + ucat[k][j][i+1].x - ucat[k+1][j][i-1].x - ucat[k][j][i-1].x) * 0.25;
		*dvdc = (ucat[k+1][j][i+1].y + ucat[k][j][i+1].y - ucat[k+1][j][i-1].y - ucat[k][j][i-1].y) * 0.25;
		*dwdc = (ucat[k+1][j][i+1].z + ucat[k][j][i+1].z - ucat[k+1][j][i-1].z - ucat[k][j][i-1].z) * 0.25;
	}

	if ((nvert[k][j+1][i])> 0.1 || (nvert[k+1][j+1][i])> 0.1) {
		*dude = (ucat[k+1][j  ][i].x + ucat[k][j  ][i].x - ucat[k+1][j-1][i].x - ucat[k][j-1][i].x) * 0.5;
		*dvde = (ucat[k+1][j  ][i].y + ucat[k][j  ][i].y - ucat[k+1][j-1][i].y - ucat[k][j-1][i].y) * 0.5;
		*dwde = (ucat[k+1][j  ][i].z + ucat[k][j  ][i].z - ucat[k+1][j-1][i].z - ucat[k][j-1][i].z) * 0.5;
	}
	else if ((nvert[k][j-1][i])> 0.1 || (nvert[k+1][j-1][i])> 0.1) {
		*dude = (ucat[k+1][j+1][i].x + ucat[k][j+1][i].x - ucat[k+1][j  ][i].x - ucat[k][j  ][i].x) * 0.5;
		*dvde = (ucat[k+1][j+1][i].y + ucat[k][j+1][i].y - ucat[k+1][j  ][i].y - ucat[k][j  ][i].y) * 0.5;
		*dwde = (ucat[k+1][j+1][i].z + ucat[k][j+1][i].z - ucat[k+1][j  ][i].z - ucat[k][j  ][i].z) * 0.5;
	}
	else {
		*dude = (ucat[k+1][j+1][i].x + ucat[k][j+1][i].x - ucat[k+1][j-1][i].x - ucat[k][j-1][i].x) * 0.25;
		*dvde = (ucat[k+1][j+1][i].y + ucat[k][j+1][i].y - ucat[k+1][j-1][i].y - ucat[k][j-1][i].y) * 0.25;
		*dwde = (ucat[k+1][j+1][i].z + ucat[k][j+1][i].z - ucat[k+1][j-1][i].z - ucat[k][j-1][i].z) * 0.25;
	}
	
	*dudz = ucat[k+1][j][i].x - ucat[k][j][i].x;
	*dvdz = ucat[k+1][j][i].y - ucat[k][j][i].y;
	*dwdz = ucat[k+1][j][i].z - ucat[k][j][i].z;
}

inline void Compute_dscalar_k (int i, int j, int k, int mx, int my, int mz, PetscReal ***K, PetscReal ***nvert, double *dkdc, double *dkde, double *dkdz )
{
	if ((nvert[k][j][i+1])> 0.1 || (nvert[k+1][j][i+1])> 0.1) {
		*dkdc = (K[k+1][j][i  ] + K[k][j][i  ] - K[k+1][j][i-1] - K[k][j][i-1]) * 0.5;
	}
	else if ((nvert[k][j][i-1])> 0.1 || (nvert[k+1][j][i-1])> 0.1) {
		*dkdc = (K[k+1][j][i+1] + K[k][j][i+1] - K[k+1][j][i  ] - K[k][j][i  ]) * 0.5;
	}
	else {
		*dkdc = (K[k+1][j][i+1] + K[k][j][i+1] - K[k+1][j][i-1] - K[k][j][i-1]) * 0.25;
	}

	if ((nvert[k][j+1][i])> 0.1 || (nvert[k+1][j+1][i])> 0.1) {
		*dkde = (K[k+1][j  ][i] + K[k][j  ][i] - K[k+1][j-1][i] - K[k][j-1][i]) * 0.5;
	}
	else if ((nvert[k][j-1][i])> 0.1 || (nvert[k+1][j-1][i])> 0.1) {
		*dkde = (K[k+1][j+1][i] + K[k][j+1][i] - K[k+1][j  ][i] - K[k][j  ][i]) * 0.5;
	}
	else {
		*dkde = (K[k+1][j+1][i] + K[k][j+1][i] - K[k+1][j-1][i] - K[k][j-1][i]) * 0.25;
	}

	*dkdz = K[k+1][j][i] - K[k][j][i];
}



inline void Compute_dkdo_center (int i, int j, int k,  int mx, int my, int mz, Cmpnts2 ***komega, PetscReal ***nvert, 
				double *dkdc, double *dodc,  double *dkde, double *dode,  double *dkdz, double *dodz )
{
  /*if ((nvert[k][j][i+1])> 0.1) {
		*dkdc = ( komega[k][j][i].x - komega[k][j][i-1].x );
		*dodc = ( komega[k][j][i].y - komega[k][j][i-1].y );
	}
	else if ((nvert[k][j][i-1])> 0.1) {
		*dkdc = ( komega[k][j][i+1].x - komega[k][j][i].x );
		*dodc = ( komega[k][j][i+1].y - komega[k][j][i].y );
	}
	else*/ {/*
		if(i_periodic && i==1) {
			*dkdc = ( komega[k][j][i+1].x - komega[k][j][mx-2].x ) * 0.5;
			*dodc = ( komega[k][j][i+1].y - komega[k][j][mx-2].y ) * 0.5;
		}
		else if(i_periodic && i==mx-2) {
			*dkdc = ( komega[k][j][1].x - komega[k][j][i-1].x ) * 0.5;
			*dodc = ( komega[k][j][1].y - komega[k][j][i-1].y ) * 0.5;
		}
		else if(ii_periodic && i==1) {
			*dkdc = ( komega[k][j][i+1].x - komega[k][j][-2].x ) * 0.5;
			*dodc = ( komega[k][j][i+1].y - komega[k][j][-2].y ) * 0.5;
		}
		else if(ii_periodic && i==mx-2) {
			*dkdc = ( komega[k][j][mx+1].x - komega[k][j][i-1].x ) * 0.5;
			*dodc = ( komega[k][j][mx+1].y - komega[k][j][i-1].y ) * 0.5;
		}
		else*/ {
			*dkdc = ( komega[k][j][i+1].x - komega[k][j][i-1].x ) * 0.5;
			*dodc = ( komega[k][j][i+1].y - komega[k][j][i-1].y ) * 0.5;
		}
	}

	/*if ((nvert[k][j+1][i])> 0.1) {
		*dkde = ( komega[k][j][i].x - komega[k][j-1][i].x );
		*dode = ( komega[k][j][i].y - komega[k][j-1][i].y );
	}
	else if ((nvert[k][j-1][i])> 0.1) {
		*dkde = ( komega[k][j+1][i].x - komega[k][j][i].x );
		*dode = ( komega[k][j+1][i].y - komega[k][j][i].y );
	}
	else*/ {
	  /*if(j_periodic && j==1) {
			*dkde = ( komega[k][j+1][i].x - komega[k][my-2][i].x ) * 0.5;
			*dode = ( komega[k][j+1][i].y - komega[k][my-2][i].y ) * 0.5;
		}
		else if(j_periodic && j==my-2) {
			*dkde = ( komega[k][1][i].x - komega[k][j-1][i].x ) * 0.5;
			*dode = ( komega[k][1][i].y - komega[k][j-1][i].y ) * 0.5;
		}
		else if(jj_periodic && j==1) {
			*dkde = ( komega[k][j+1][i].x - komega[k][-2][i].x ) * 0.5;
			*dode = ( komega[k][j+1][i].y - komega[k][-2][i].y ) * 0.5;
		}
		else if(jj_periodic && j==my-2) {
			*dkde = ( komega[k][my+1][i].x - komega[k][j-1][i].x ) * 0.5;
			*dode = ( komega[k][my+1][i].y - komega[k][j-1][i].y ) * 0.5;
		}
		else*/ {
			*dkde = ( komega[k][j+1][i].x - komega[k][j-1][i].x ) * 0.5;
			*dode = ( komega[k][j+1][i].y - komega[k][j-1][i].y ) * 0.5;
		}
	}

	/*if ((nvert[k+1][j][i])> 0.1) {
		*dkdz = ( komega[k][j][i].x - komega[k-1][j][i].x );
		*dodz = ( komega[k][j][i].y - komega[k-1][j][i].y );
	}
	else if ((nvert[k-1][j][i])> 0.1) {
		*dkdz = ( komega[k+1][j][i].x - komega[k][j][i].x );
		*dodz = ( komega[k+1][j][i].y - komega[k][j][i].y );
	}
	else*/ {
	  /*if(k_periodic && k==1) {
			*dkdz = ( komega[k+1][j][i].x - komega[mz-2][j][i].x ) * 0.5;
			*dodz = ( komega[k+1][j][i].y - komega[mz-2][j][i].y ) * 0.5;
		}
		else if(k_periodic && k==mz-2) {
			*dkdz = ( komega[1][j][i].x - komega[k-1][j][i].x ) * 0.5;
			*dodz = ( komega[1][j][i].y - komega[k-1][j][i].y ) * 0.5;
		}
		else if(kk_periodic && k==1) {
			*dkdz = ( komega[k+1][j][i].x - komega[-2][j][i].x ) * 0.5;
			*dodz = ( komega[k+1][j][i].y - komega[-2][j][i].y ) * 0.5;
		}
		else if(kk_periodic && k==mz-2) {
			*dkdz = ( komega[mz+1][j][i].x - komega[k-1][j][i].x ) * 0.5;
			*dodz = ( komega[mz+1][j][i].y - komega[k-1][j][i].y ) * 0.5;
		}
		else*/ {
			*dkdz = ( komega[k+1][j][i].x - komega[k-1][j][i].x ) * 0.5;
			*dodz = ( komega[k+1][j][i].y - komega[k-1][j][i].y ) * 0.5;
		}
	}
}

inline void Compute_dscalar_center (int i, int j, int k,  int mx, int my, int mz, PetscReal ***K, PetscReal ***nvert, double *dkdc, double *dkde, double *dkdz)
{

	if (i==mx-1) *dkdc = ( K[k][j][i] - K[k][j][i-1] );
	else if (i==0) *dkdc = ( K[k][j][i+1] - K[k][j][i] );
	else if ((nvert[k][j][i+1])> 0.1) {   // xiaolei add last two conditions for test
		*dkdc = ( K[k][j][i] - K[k][j][i-1] );
	}
	else if ((nvert[k][j][i-1])> 0.1) {
		*dkdc = ( K[k][j][i+1] - K[k][j][i] );
	}
	else {
		/*if(i_periodic && i==1) {
			*dkdc = ( K[k][j][i+1] - K[k][j][mx-2] ) * 0.5;
		}
		else if(i_periodic && i==mx-2) {
			*dkdc = ( K[k][j][1] - K[k][j][i-1] ) * 0.5;
		}
		else if(ii_periodic && i==1) {
			*dkdc = ( K[k][j][i+1] - K[k][j][-2] ) * 0.5;
		}
		else if(ii_periodic && i==mx-2) {
			*dkdc = ( K[k][j][mx+1] - K[k][j][i-1] ) * 0.5;
		}
		else*/ {
			*dkdc = ( K[k][j][i+1] - K[k][j][i-1] ) * 0.5;
		}
	}

	if (j==my-1) *dkde = ( K[k][j][i] - K[k][j-1][i] );
	else if (j==0) *dkde = ( K[k][j+1][i] - K[k][j][i] );
	else if ((nvert[k][j+1][i])> 0.1 ) {
		*dkde = ( K[k][j][i] - K[k][j-1][i] );
	}
	else if ((nvert[k][j-1][i])> 0.1 ) {
		*dkde = ( K[k][j+1][i] - K[k][j][i] );
	}
	else {
		/*if(j_periodic && j==1) {
			*dkde = ( K[k][j+1][i] - K[k][my-2][i] ) * 0.5;
		}
		else if(j_periodic && j==my-2) {
			*dkde = ( K[k][1][i] - K[k][j-1][i] ) * 0.5;
		}
		else if(jj_periodic && j==1) {
			*dkde = ( K[k][j+1][i] - K[k][-2][i] ) * 0.5;
		}
		else if(jj_periodic && j==my-2) {
			*dkde = ( K[k][my+1][i] - K[k][j-1][i] ) * 0.5;
		}
		else*/ {
			*dkde = ( K[k][j+1][i] - K[k][j-1][i] ) * 0.5;
		}
	}

	if (k==mz-1) *dkdz = ( K[k][j][i] - K[k-1][j][i] );
	else if (k==0) *dkdz = ( K[k+1][j][i] - K[k][j][i] );
	else if ((nvert[k+1][j][i])> 0.1 ) {
		*dkdz = ( K[k][j][i] - K[k-1][j][i] );
	}
	else if ((nvert[k-1][j][i])> 0.1 ) {
		*dkdz = ( K[k+1][j][i] - K[k][j][i] );
	}
	else {
		/*if(k_periodic && k==1) {
			*dkdz = ( K[k+1][j][i] - K[mz-2][j][i] ) * 0.5;
		}
		else if(k_periodic && k==mz-2) {
			*dkdz = ( K[1][j][i] - K[k-1][j][i] ) * 0.5;
		}
		else if(kk_periodic && k==1) {
			*dkdz = ( K[k+1][j][i] - K[-2][j][i] ) * 0.5;
		}
		else if(kk_periodic && k==mz-2) {
			*dkdz = ( K[mz+1][j][i] - K[k-1][j][i] ) * 0.5;
		}
		else */{
			*dkdz = ( K[k+1][j][i] - K[k-1][j][i] ) * 0.5;
		}
	}	
}


// xiaolei add

inline void AGAUS (double matrix[6][6], double b[6], double x[6]) {
//inline void AGAUS (double **matrix, double *b, double *x, int N) {
int n,i,count,j;
double ratio,temp;
n=6;
/* Gaussian elimination */
for(i=0;i<(n-1);i++){
        for(j=(i+1);j<n;j++){
                ratio = matrix[j][i] / matrix[i][i];
                for(count=i;count<n;count++) {
                        matrix[j][count] -= (ratio * matrix[i][count]);
                }
                b[j] -= (ratio * b[i]);
        }
}
/* Back substitution */
x[n-1] = b[n-1] / matrix[n-1][n-1];
for(i=(n-2);i>=0;i--){
        temp = b[i];
        for(j=(i+1);j<n;j++){
                temp -= (matrix[i][j] * x[j]);
        }
        x[i] = temp / matrix[i][i];
}
}



#endif
