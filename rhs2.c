#include "variables.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
/*
extern double sign(double a);
extern char path[256];
extern int ucat_plane_allocated;	// bcs.c
extern IBMNodes *ibm_ptr;
*/
int allocated_levelset=0;
int wall_distance=1;

double integrate_gridfilter(double val[3][3][3], double vol[3][3][3]);
double integrate_gridfilter_1d(double val[3], double vol[3]);
void Calculate_Covariant_tensor(double g[3][3], double G[3][3]);
void search_free_surface_position(UserCtx *user);

int fact(int number) {
  int temp;

  if(number <= 1) return 1;

  temp = number * fact(number - 1);
  return temp;
}

int dfact(int n)
{
  int i;
  double res=1.0;
  for(i=n;i>=1;i-=2)
    {
      res *=i;
    }
  return res;
}

//Formulas for Calculating the Error Function of a Complex Variable, H. E. Salzer, 1951, American Mathematical Society, 
// http://www.jstor.org/stable/2002163
std::complex<double> ERF(std::complex<double> z) // a+bi
{
  double X = z.real(), Y = z.imag();
  int N=10;
  std::complex<double> val(0,0);
  double A, B;
  double W = exp(-X*X)/sqrt(M_PI);
  
  A = erf(X) + W * (1.-cos(2.*X*Y))/(4.*X);
  B = W / (4.*X) * sin(2*X*Y);

  for(int n=0; n<N; n++) {
    A += W * pow(n*n+4.*X*X, -1) * ( 2*X - 2*X*cosh(n*Y)*cos(2*X*Y) + n*sinh(n*Y)*sin(2*X*Y) ) * exp(-n*n/4.);
    B += W * pow(n*n+4.*X*X, -1) * exp(-n*n/4.) * ( 2*X*cosh(n*Y)*sin(2*X*Y) + n*sinh(n*Y)*cos(2*X*Y) );
  }

  val = std::complex<double> (A, B);
  return val;
}

void GlobalSum_Root(PetscReal* local,PetscReal* result,MPI_Comm comm)
{
  MPI_Reduce(local,result,1,MPIU_REAL,MPI_SUM,0,comm);
}

void GlobalMax_Root(PetscReal* local,PetscReal* result,MPI_Comm comm)
{
  MPI_Reduce(local,result,1,MPIU_REAL,MPI_MAX,0,comm);
}

void GlobalSum_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm)
{
	MPI_Allreduce(local,result,1,MPIU_SCALAR,MPIU_SUM,comm);
}

void GlobalMax_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm)
{
	MPI_Allreduce(local,result,1,MPIU_REAL,MPI_MAX,comm);
}

void GlobalMin_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm)
{
	MPI_Allreduce(local,result,1,MPIU_REAL,MPI_MIN,comm);
}


double limiter(double r)	// van Albada
{
	return 1;
	//return 2. * r / ( 1. + r*r ); --> gives Nan
};
/*
	Piecewise parabloic method with flux limiting
	<example>
          i-1     i    |   i+1   i+2
         WW   W        E      EE

*/
double PPM(double WW, double W, double E, double EE, double a)
{
	const double kappa=1./3.;
	double du0=W-WW;	// du_i-1/2
	double du1=E-W;		// du_i+1/2
	double du2=EE-E;		// du_i+3/2
	
	double rL = (W-WW)/(E-W+1.e-30);
	double rR = (E-W)/(EE-E+1.e-30);
	double uL = W + 0.25 * limiter(rL) * ( (1.-kappa)*du0 + (1.+kappa)*du1 );
	double uR = E -  0.25 * limiter(rR) * ( (1.-kappa)*du2 + (1.+kappa)*du1 );
	
	if(a>0) return uL;
	else return uR;
};

int nfree(PetscReal ***nvert, int k, int j, int i)
{
	int n=0;
	if( (int)nvert[k][j][i+1]==5 ) n++;
	if( (int)nvert[k][j][i-1]==5 ) n++;
	if( (int)nvert[k][j+1][i]==5 ) n++;
	if( (int)nvert[k][j-1][i]==5 ) n++;
	if( (int)nvert[k+1][j][i]==5 ) n++;
	if( (int)nvert[k-1][j][i]==5 ) n++;
	
	return n;
};

void free_surafe_BC(UserCtx *user)
{
	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	mx, my, mz;
	int i, j, k;
	Cmpnts	***ucont, ***ucat;
	PetscReal ***nvert, ***p, ***iaj, ***jaj, ***kaj;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;
	
	DMDAGetLocalInfo(da, &info);
	mx = info.mx; my = info.my; mz = info.mz;
	
	int xs = info.xs, xe = xs + info.xm;
	int ys = info.ys, ye = ys + info.ym;
	int zs = info.zs, ze = zs + info.zm;

	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	
	
	DMDAVecGetArray(fda, user->lUcont, &ucont);
	DMDAVecGetArray(fda, user->lUcat, &ucat);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lP, &p);
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);
	
	DMDAVecGetArray(fda, user->lICsi, &icsi);
	DMDAVecGetArray(fda, user->lIEta, &ieta);
	DMDAVecGetArray(fda, user->lIZet, &izet);
	DMDAVecGetArray(fda, user->lJCsi, &jcsi);
	DMDAVecGetArray(fda, user->lJEta, &jeta);
	DMDAVecGetArray(fda, user->lJZet, &jzet);
	DMDAVecGetArray(fda, user->lKCsi, &kcsi);
	DMDAVecGetArray(fda, user->lKEta, &keta);
	DMDAVecGetArray(fda, user->lKZet, &kzet);

	/* normal dynamic BC, applies to all surface cell
			
			 ------------------
			 |   Empty   |
			+----V*_i,j------	V*_i,j is unknown
			 |    p_i,j     |
			+---V_i,j-1----
		
			p_i,j	= 2/Re * d(un)/dn 
				= 2/Re * dv / dy     // (assuming normal is +y direction)
				= 2/Re * ( jcsi[k][j][i].x * dvdc + jeta[k][j][i].y * dvde + jzet[k][j][i].z * dvdz ) * jaj
				= 2/Re * ( jeta[k][j][i].y * dvde ) * jaj	// (assuming +y direction is not curved)
				= 2/Re * ( ucont[k][j][i].y - ucont[k][j-1][i].y ) * jaj[k][j][i]
		
			ucont[k][j][i].y = ucont[k][j-1][i].y + p_i,j * Re * 0.5 / jaj[k][j][i]
	*/
	for (k=lzs; k<lze; k++)
	for (j=lys-2; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if( j>1&& (int)nvert[k][j][i]==0 && (int)nvert[k][j+1][i]==5 ) {
			//ucont[k][j][i].y = ucont[k][j-1][i].y + p[k][j][i] * user->ren * 0.5 / jaj[k][j][i];
			ucont[k][j][i].y = 2*ucont[k][j-1][i].y - ucont[k][j-2][i].y;
		}
	}
	
	// localtolocal later here
	
	/* tangential dynamic BC, applies to one side wet cell
		
			 -----------------------------------+
			 |                |		    | 
			 |      i,j    U*_i,j  i+1,j   | 
			 |                |		    | 
			+----V_i,j-1-+-V_i+1,j-1-+--------------Free surface------------
			 |                |		    |
			 |    p_i,j-1 U_i,j-1	    |
			 |                | p_i+1,j-1  |
			+---V_i,j-2--+----------------+
		
			du/dy + dv/dx = 0	(1) -- u is unknown.  v is knwon from normal dynamic BC
			dw/dy + dv/dz = 0	(2) -- w is unkonwn 
	
		(1)	dU/de * jg22 = -  ( ig11 * dV/dc + kg13 * dV/dz )	Eq. (56) Street and Hodges
			dU/de	= - ( ig11 * dV/dc + kg13 * dV/dz ) / jg22
			ucont[k][j][i].x - ucont[k][j-1][i].x = - ( ig11 * ( ucont[k][j-1][i+1].y - ucont[k][j-1][i].y ) + kg13 * ( ucont[k+1][j-1][i].y - ucont[k][j-1][i].y ) ) / jg22
			ucont[k][j][i].x = ucont[k][j-1][i].x - ( ig11 * ( ucont[k][j-1][i+1].y - ucont[k][j-1][i].y ) + kg13 * ( ucont[k+1][j-1][i].y - ucont[k][j-1][i].y ) ) / jg22
	
		(2)	dW/de * jg22 = -  ( kg33 * dV/dz + ig13 * dV/dc )
			dW/de 	= - ( kg33 * dV/dz + ig13 * dV/dc ) / jg22 
			ucont[k][j][i].z - ucont[k][j-1][i].z = - ( kg33 * ( ucont[k+1][j-1][i].y - ucont[k][j-1][i].y ) + ig13 * ( ucont[k][j-1][i+1].y - ucont[k][j-1][i].y ) ) / jg22 
			ucont[k][j][i].z  = ucont[k][j-1][i].z - ( kg33 * ( ucont[k+1][j-1][i].y - ucont[k][j-1][i].y ) + ig13 * ( ucont[k][j-1][i+1].y - ucont[k][j-1][i].y ) ) / jg22 
	*/
	double csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2;
	double ig11, ig13;
	double jg22;
	double kg33, kg13;
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		int n = nfree(nvert,k,j,i);
		
		csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
		zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;
		ig11 = (csi0 * csi0 + csi1 * csi1 + csi2 * csi2) * iaj[k][j][i];
		ig13 = (zet0 * csi0 + zet1 * csi1 + zet2 * csi2) * iaj[k][j][i];
		
		eta0 = jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
		jg22 = (eta0 * eta0 + eta1 * eta1 + eta2 * eta2) * jaj[k][j][i];
		
		csi0 = kcsi[k][j][i].x, csi1 = kcsi[k][j][i].y, csi2 = kcsi[k][j][i].z;
		zet0 = kzet[k][j][i].x, zet1 = kzet[k][j][i].y, zet2 = kzet[k][j][i].z;
		kg33 = (zet0 * zet0 + zet1 * zet1 + zet2 * zet2) * kaj[k][j][i];
		kg13 = (zet0 * csi0 + zet1 * csi1 + zet2 * csi2) * kaj[k][j][i];
		
		
		// ucont.x
		if( (int)(nvert[k][j][i]+nvert[k][j][i+1])==10 ) {
			if( (int)(nvert[k][j-1][i]+nvert[k][j-1][i+1])==0 ) {
				ucont[k][j][i].x = ucont[k][j-1][i].x - ( ig11 * ( ucont[k][j-1][i+1].y - ucont[k][j-1][i].y ) + kg13 * ( ucont[k+1][j-1][i].y - ucont[k][j-1][i].y ) ) / jg22;
			}
		}
		
		// ucont.z
		if( (int)(nvert[k][j][i]+nvert[k+1][j][i])==10 ) {
			if( (int)(nvert[k][j-1][i]+nvert[k+1][j-1][i])==0 ) {
				ucont[k][j][i].z  = ucont[k][j-1][i].z - ( kg33 * ( ucont[k+1][j-1][i].y - ucont[k][j-1][i].y ) + ig13 * ( ucont[k][j-1][i+1].y - ucont[k][j-1][i].y ) ) / jg22 ;
			}
		}
		
		/*
			ucont[k][j][i].x - ucont[k][j][i-1].x + ucont[k][j][i].y - ucont[k][j-1][i].y + ucont[k][j][i].z - ucont[k-1][j][i].z
		*/

		/*
		if( (int)nvert[k][j][i]==0 ) {
			if( (int)nvert[k][j][i+1]==5 && i!=1)  {
				ucont[k][j][i].x = 2*ucont[k][j][i-1].x - ucont[k][j][i-2].x;
			}
			if( (int)nvert[k][j+1][i]==5 && j!=1 )  {
				ucont[k][j][i].y = 2*ucont[k][j-1][i].y - ucont[k][j-2][i].y;
			}
			if( (int)nvert[k+1][j][i]==5 && k!=1 )  {
				ucont[k][j][i].z = 2*ucont[k-1][j][i].z - ucont[k-2][j][i].z;
			}
		}
		
		else if( (int)nvert[k][j][i]==5 ) {
			if( (int)nvert[k][j][i+1]==0 && i!=mx-2)  {
				ucont[k][j][i].x = 2*ucont[k][j][i+1].x - ucont[k][j][i+2].x;
			}
			if( (int)nvert[k][j+1][i]==0 && j!=my-2)  {
				// impossible
			}
			if( (int)nvert[k+1][j][i]==0 && k!=mz-2 )  {
				ucont[k][j][i].z = 2*ucont[k+1][j][i].z - ucont[k+2][j][i].z;
			}
		}
		
		if( (int)nvert[k][j][i]==5 ) {
			if( (int)nvert[k][j-1][i]==0 && j!=1 ) {
				ucat[k][j][i].x = 2*ucat[k][j-1][i].x - ucat[k][j-2][i].x;
				ucat[k][j][i].y = 2*ucat[k][j-1][i].y - ucat[k][j-2][i].y;
				ucat[k][j][i].z = 2*ucat[k][j-1][i].z - ucat[k][j-2][i].z;
			}
		}
		*/
		
		if( (int)nvert[k][j][i]==0 ) {
			if( (int)nvert[k][j][i+1]==5 )  {
				if( n>1 ) ucont[k][j][i].x = ucont[k][j][i-1].x;
			}
			if( (int)nvert[k+1][j][i]==5 )  {
				if( n>1 ) ucont[k][j][i].z = ucont[k-1][j][i].z;
			}
		}
		else if( (int)nvert[k][j][i]==5 ) {
			if( (int)nvert[k][j][i+1]==0 && i!=mx-2)  {
				if(nfree(nvert,k,j,i+1)>1) ucont[k][j][i].x = ucont[k][j][i+1].x;
			}
			if( (int)nvert[k+1][j][i]==0 && k!=mz-2 )  {
				if(nfree(nvert,k+1,j,i)>1) ucont[k][j][i].z = ucont[k+1][j][i].z;
			}
		}
		
	}
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if(j>2 && (int)nvert[k][j-1][i]==5 && (int)nvert[k][j-2][i]==0) ucont[k][j][i].y = ucont[k][j-1][i].y;
	}

	DMDAVecRestoreArray(fda, user->lUcont, &ucont);
	DMDAVecRestoreArray(fda, user->lUcat, &ucat);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lP, &p);
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);
	
	DMDAVecRestoreArray(fda, user->lICsi, &icsi);
	DMDAVecRestoreArray(fda, user->lIEta, &ieta);
	DMDAVecRestoreArray(fda, user->lIZet, &izet);
	DMDAVecRestoreArray(fda, user->lJCsi, &jcsi);
	DMDAVecRestoreArray(fda, user->lJEta, &jeta);
	DMDAVecRestoreArray(fda, user->lJZet, &jzet);
	DMDAVecRestoreArray(fda, user->lKCsi, &kcsi);
	DMDAVecRestoreArray(fda, user->lKEta, &keta);
	DMDAVecRestoreArray(fda, user->lKZet, &kzet);
	
	DMDALocalToLocalBegin(fda, user->lUcont, INSERT_VALUES, user->lUcont);
	DMDALocalToLocalEnd(fda, user->lUcont, INSERT_VALUES, user->lUcont);
	
	DMLocalToGlobalBegin(fda, user->lUcont, INSERT_VALUES, user->Ucont);
	DMLocalToGlobalEnd(fda, user->lUcont, INSERT_VALUES, user->Ucont);
};

void Update_Velocity_by_Gravity(UserCtx *user)
{
	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	mx, my, mz;
	int i, j, k;
	
	PetscScalar ***nvert, ***iaj, ***jaj, ***kaj;
	Cmpnts	***ucont;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;
	
	DMDAGetLocalInfo(da, &info);
	
	mx = info.mx; my = info.my; mz = info.mz;
	int xs = info.xs, xe = xs + info.xm;
	int ys = info.ys, ye = ys + info.ym;
	int zs = info.zs, ze = zs + info.zm;

	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	DMDAVecGetArray(fda, user->lUcont, &ucont);
	
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);
	
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
	
	
		
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
	  // periodic conditions are not considered here
		if( i<mx-2 && j<my-2 && k<mz-2) {
			/* 
				contravariant g vector
				Gi = (gx,gy,gz) dot (csi.x, csi.y, csi.z)
				Gj = (gx,gy,gz) dot (eta.x, eta.y, eta.z)
				Gk = (gx,gy,gz) dot (zet.x, zet.y, zet.z)
			*/
			if((int)(nvert[k][j][i]+nvert[k][j][i+1])==0) ucont[k][j][i].x += ( gravity_x * icsi[k][j][i].x + gravity_y * icsi[k][j][i].y + gravity_z * icsi[k][j][i].z ) * user->dt * user->st / time_coeff();
			if((int)(nvert[k][j][i]+nvert[k][j+1][i])==0) ucont[k][j][i].y += ( gravity_x * jeta[k][j][i].x + gravity_y * jeta[k][j][i].y + gravity_z * jeta[k][j][i].z ) * user->dt * user->st / time_coeff();
			if((int)(nvert[k][j][i]+nvert[k+1][j][i])==0) ucont[k][j][i].z += ( gravity_x * kzet[k][j][i].x + gravity_y * kzet[k][j][i].y + gravity_z * kzet[k][j][i].z ) * user->dt * user->st / time_coeff();
		}
	}
	
	DMDAVecRestoreArray(fda, user->lUcont, &ucont);
	
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);
	
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
	
	DMDALocalToLocalBegin(fda, user->lUcont, INSERT_VALUES, user->lUcont);
	DMDALocalToLocalEnd(fda, user->lUcont, INSERT_VALUES, user->lUcont);
	
	DMLocalToGlobalBegin(fda, user->lUcont, INSERT_VALUES, user->Ucont);
	DMLocalToGlobalEnd(fda, user->lUcont, INSERT_VALUES, user->Ucont);
}

double integrate_testfilter_ik(double val[3][3][3], double vol[3][3][3])
{
	/*
	double v1, v2, v3, v4;
	
	double val_sum=0, w_sum=0;
	double w1, w2, w3, w4, w5, w6, w7, w8;
	
	
	// Trapezoidal 
	v1 = ( val[0][1][0]+val[1][1][0] +val[0][1][1]+val[1][1][1] ) * 0.25;
	v2 = ( val[1][1][0]+val[2][1][0] +val[1][1][1]+val[2][1][1] ) * 0.25;
	v3 = ( val[0][1][1]+val[1][1][1]+val[0][1][2]+val[1][1][2] ) * 0.25;
	v4 = ( val[1][1][1]+val[2][1][1]+val[1][1][2]+val[2][1][2]) * 0.25;
	
	return 0.25 * (v1+v2+v3+v4);
	*/
	// Simpson rule 
	// See Morinish and Vasilyev (2001) Phys Fluids
	// 2d homogeneous test filter :  pow(5.0, 1./3.)
	return  ( (val[0][1][0]+val[2][1][0]+val[0][1][2]+val[2][1][2]) + 4.*(val[0][1][1]+val[1][1][0]+val[2][1][1]+val[1][1][2]) + 16.*val[1][1][1] ) / 36.;
	
};

double integrate_testfilter(double val[3][3][3], double w[3][3][3])
{
	double v1, v2, v3, v4, v5, v6, v7, v8;
	double w1, w2, w3, w4, w5, w6, w7, w8;
	
	if( testfilter_ik ) {
		return integrate_testfilter_ik(val, w);
	}
	/*
	( val[0][0] + val[1][0] + val[0][1] + val[1][1] )
	( val[1][0] + val[2][0] + val[1][1] + val[2][1] )
	( val[0][1] + val[1][1] + val[0][2] + val[1][2] )
	( val[1][1] + val[2][1] + val[1][2] + val[2][2] )
	*/
	/*
	// bottom : j=0, 1
	v1 = 0.125 * ( val[0][0][0] + val[1][0][0] + val[0][0][1] + val[1][0][1] ) + 0.125 * ( val[0][1][0] + val[1][1][0] + val[0][1][1] + val[1][1][1] );
	v2 = 0.125 * ( val[1][0][0] + val[2][0][0] + val[1][0][1] + val[2][0][1] ) + 0.125 * ( val[1][1][0] + val[2][1][0] + val[1][1][1] + val[2][1][1] );
	v3 = 0.125 * ( val[0][0][1] + val[1][0][1] + val[0][0][2] + val[1][0][2] ) + 0.125 * ( val[0][1][1] + val[1][1][1] + val[0][1][2] + val[1][1][2] );
	v4 = 0.125 * ( val[1][0][1] + val[2][0][1] + val[1][0][2] + val[2][0][2] ) + 0.125 * ( val[1][1][1] + val[2][1][1] + val[1][1][2] + val[2][1][2] );
	
	//top : j=1, 2
	v5 = 0.125 * ( val[0][1][0] + val[1][1][0] + val[0][1][1] + val[1][1][1] ) + 0.125 * ( val[0][2][0] + val[1][2][0] + val[0][2][1] + val[1][2][1] );
	v6 = 0.125 * ( val[1][1][0] + val[2][1][0] + val[1][1][1] + val[2][1][1] ) + 0.125 * ( val[1][2][0] + val[2][2][0] + val[1][2][1] + val[2][2][1] );
	v7 = 0.125 * ( val[0][1][1] + val[1][1][1] + val[0][1][2] + val[1][1][2] ) + 0.125 * ( val[0][2][1] + val[1][2][1] + val[0][2][2] + val[1][2][2] );
	v8 = 0.125 * ( val[1][1][1] + val[2][1][1] + val[1][1][2] + val[2][1][2] ) + 0.125 * ( val[1][2][1] + val[2][2][1] + val[1][2][2] + val[2][2][2] );
		
	return 0.125 * ( v1+v2+v3+v4+v5+v6+v7+v8 );
	*/
	v1 = ( val[0][0][0]*w[0][0][0] + val[1][0][0]*w[1][0][0] + val[0][0][1]*w[0][0][1] + val[1][0][1]*w[1][0][1] + val[0][1][0]*w[0][1][0] + val[1][1][0]*w[1][1][0] + val[0][1][1]*w[0][1][1] + val[1][1][1]*w[1][1][1] );
	v2 = ( val[1][0][0]*w[1][0][0] + val[2][0][0]*w[2][0][0] + val[1][0][1]*w[1][0][1] + val[2][0][1]*w[2][0][1] + val[1][1][0]*w[1][1][0] + val[2][1][0]*w[2][1][0] + val[1][1][1]*w[1][1][1] + val[2][1][1]*w[2][1][1] );
	v3 = ( val[0][0][1]*w[0][0][1] + val[1][0][1]*w[1][0][1] + val[0][0][2]*w[0][0][2] + val[1][0][2]*w[1][0][2] + val[0][1][1]*w[0][1][1] + val[1][1][1]*w[1][1][1] + val[0][1][2]*w[0][1][2] + val[1][1][2]*w[1][1][2] );
	v4 = ( val[1][0][1]*w[1][0][1] + val[2][0][1]*w[2][0][1] + val[1][0][2]*w[1][0][2] + val[2][0][2]*w[2][0][2] + val[1][1][1]*w[1][1][1] + val[2][1][1]*w[2][1][1] + val[1][1][2]*w[1][1][2] + val[2][1][2]*w[2][1][2] );
	
	//top : j=1, 2
	v5 = ( val[0][1][0]*w[0][1][0] + val[1][1][0]*w[1][1][0] + val[0][1][1]*w[0][1][1] + val[1][1][1]*w[1][1][1] + val[0][2][0]*w[0][2][0] + val[1][2][0]*w[1][2][0] + val[0][2][1]*w[0][2][1] + val[1][2][1]*w[1][2][1] );
	v6 = ( val[1][1][0]*w[1][1][0] + val[2][1][0]*w[2][1][0] + val[1][1][1]*w[1][1][1] + val[2][1][1]*w[2][1][1] + val[1][2][0]*w[1][2][0] + val[2][2][0]*w[2][2][0] + val[1][2][1]*w[1][2][1] + val[2][2][1]*w[2][2][1] );
	v7 = ( val[0][1][1]*w[0][1][1] + val[1][1][1]*w[1][1][1] + val[0][1][2]*w[0][1][2] + val[1][1][2]*w[1][1][2] + val[0][2][1]*w[0][2][1] + val[1][2][1]*w[1][2][1] + val[0][2][2]*w[0][2][2] + val[1][2][2]*w[1][2][2] );
	v8 = ( val[1][1][1]*w[1][1][1] + val[2][1][1]*w[2][1][1] + val[1][1][2]*w[1][1][2] + val[2][1][2]*w[2][1][2] + val[1][2][1]*w[1][2][1] + val[2][2][1]*w[2][2][1] + val[1][2][2]*w[1][2][2] + val[2][2][2]*w[2][2][2] );
	
	
	// bottom
	w1 = ( w[0][0][0] + w[1][0][0] + w[0][0][1] + w[1][0][1] + w[0][1][0] + w[1][1][0] + w[0][1][1] + w[1][1][1] );
	w2 = ( w[1][0][0] + w[2][0][0] + w[1][0][1] + w[2][0][1] + w[1][1][0] + w[2][1][0] + w[1][1][1] + w[2][1][1] );
	w3 = ( w[0][0][1] + w[1][0][1] + w[0][0][2] + w[1][0][2] + w[0][1][1] + w[1][1][1] + w[0][1][2] + w[1][1][2] );
	w4 = ( w[1][0][1] + w[2][0][1] + w[1][0][2] + w[2][0][2] + w[1][1][1] + w[2][1][1] + w[1][1][2] + w[2][1][2] );
	
	//top : j=1, 2
	w5 = ( w[0][1][0] + w[1][1][0] + w[0][1][1] + w[1][1][1] + w[0][2][0] + w[1][2][0] + w[0][2][1] + w[1][2][1] );
	w6 = ( w[1][1][0] + w[2][1][0] + w[1][1][1] + w[2][1][1] + w[1][2][0] + w[2][2][0] + w[1][2][1] + w[2][2][1] );
	w7 = ( w[0][1][1] + w[1][1][1] + w[0][1][2] + w[1][1][2] + w[0][2][1] + w[1][2][1] + w[0][2][2] + w[1][2][2] );
	w8 = ( w[1][1][1] + w[2][1][1] + w[1][1][2] + w[2][1][2] + w[1][2][1] + w[2][2][1] + w[1][2][2] + w[2][2][2] );
	
	return (v1+v2+v3+v4+v5+v6+v7+v8)/(w1+w2+w3+w4+w5+w6+w7+w8);
};


double integrate_testfilter_simpson(double val[3][3][3], double w[3][3][3])
{
	double v1, v2, v3, v4, v5, v6, v7, v8;
	double w1, w2, w3, w4, w5, w6, w7, w8;
	
	if( testfilter_ik ) {
		return integrate_testfilter_ik(val, w);
	}
	
	double wsum=0, valsum=0;
	
	for(int i=0; i<3; i++)
	for(int j=0; j<3; j++)
	for(int k=0; k<3; k++) {
		double simpson_w = 1.0;
		if(i==1) simpson_w *= 4.;
		if(j==1) simpson_w *= 4.;
		if(k==1) simpson_w *= 4.;
		
		wsum += simpson_w * w[i][j][k];
		valsum += simpson_w * w[i][j][k] * val[i][j][k];
	}
	
	return valsum / wsum;
};


double integrate_xztestfilter_simpson(double val[3][3][3], double w[3][3][3])
{
        double wsum=0, valsum=0;

        int j = 1;
        for(int i=0; i<3; i++)
//              for(int j=0; j<3; j++)
        for(int k=0; k<3; k++) {
                double simpson_w = 1.0;
                if(i==1) simpson_w *= 4.;
                if(j==1) simpson_w *= 4.;
                if(k==1) simpson_w *= 4.;

                wsum += simpson_w * w[i][j][k];
                valsum += simpson_w * w[i][j][k] * val[i][j][k];
        }

       return valsum / wsum;
};


double integrate_gridfilter(double val[3][3][3], double vol[3][3][3])
{
	
	//double val_sum=0;//, w_sum=0;
	
	double v1, v2, v3, v4, v5, v6, v7, v8;
	double w1, w2, w3, w4, w5, w6, w7, w8;
	
	// bottom
	v1 = ( 27.*val[1][1][1] + 9. * (val[2][1][1]+val[1][2][1]+val[1][1][2]) + 3. * ( val[2][2][1] + val[2][1][2] + val[1][2][2] ) + val[2][2][2] ) / 64.;
	v2 = ( 27.*val[1][1][1] + 9. * (val[0][1][1]+val[1][2][1]+val[1][1][2]) + 3. * ( val[0][2][1] + val[0][1][2] + val[1][2][2] ) + val[0][2][2] ) / 64.;
	v3 = ( 27.*val[1][1][1] + 9. * (val[2][1][1]+val[1][0][1]+val[1][1][2]) + 3. * ( val[2][0][1] + val[2][1][2] + val[1][0][2] ) + val[2][0][2] ) / 64.;
	v4 = ( 27.*val[1][1][1] + 9. * (val[0][1][1]+val[1][0][1]+val[1][1][2]) + 3. * ( val[0][0][1] + val[0][1][2] + val[1][0][2] ) + val[0][0][2] ) / 64.;
	
	// top
	v5 = ( 27.*val[1][1][1] + 9. * (val[2][1][1]+val[1][2][1]+val[1][1][0]) + 3. * ( val[2][2][1] + val[2][1][0] + val[1][2][0] ) + val[2][2][0] ) / 64.;
	v6 = ( 27.*val[1][1][1] + 9. * (val[0][1][1]+val[1][2][1]+val[1][1][0]) + 3. * ( val[0][2][1] + val[0][1][0] + val[1][2][0] ) + val[0][2][0] ) / 64.;
	v7 = ( 27.*val[1][1][1] + 9. * (val[2][1][1]+val[1][0][1]+val[1][1][0]) + 3. * ( val[2][0][1] + val[2][1][0] + val[1][0][0] ) + val[2][0][0] ) / 64.;
	v8 = ( 27.*val[1][1][1] + 9. * (val[0][1][1]+val[1][0][1]+val[1][1][0]) + 3. * ( val[0][0][1] + val[0][1][0] + val[1][0][0] ) + val[0][0][0] ) / 64.;

	// bottom
	w1 = ( 27.*vol[1][1][1] + 9. * (vol[2][1][1]+vol[1][2][1]+vol[1][1][2]) + 3. * ( vol[2][2][1] + vol[2][1][2] + vol[1][2][2] ) + vol[2][2][2] ) / 64.;
	w2 = ( 27.*vol[1][1][1] + 9. * (vol[0][1][1]+vol[1][2][1]+vol[1][1][2]) + 3. * ( vol[0][2][1] + vol[0][1][2] + vol[1][2][2] ) + vol[0][2][2] ) / 64.;
	w3 = ( 27.*vol[1][1][1] + 9. * (vol[2][1][1]+vol[1][0][1]+vol[1][1][2]) + 3. * ( vol[2][0][1] + vol[2][1][2] + vol[1][0][2] ) + vol[2][0][2] ) / 64.;
	w4 = ( 27.*vol[1][1][1] + 9. * (vol[0][1][1]+vol[1][0][1]+vol[1][1][2]) + 3. * ( vol[0][0][1] + vol[0][1][2] + vol[1][0][2] ) + vol[0][0][2] ) / 64.;
	
	// top
	w5 = ( 27.*vol[1][1][1] + 9. * (vol[2][1][1]+vol[1][2][1]+vol[1][1][0]) + 3. * ( vol[2][2][1] + vol[2][1][0] + vol[1][2][0] ) + vol[2][2][0] ) / 64.;
	w6 = ( 27.*vol[1][1][1] + 9. * (vol[0][1][1]+vol[1][2][1]+vol[1][1][0]) + 3. * ( vol[0][2][1] + vol[0][1][0] + vol[1][2][0] ) + vol[0][2][0] ) / 64.;
	w7 = ( 27.*vol[1][1][1] + 9. * (vol[2][1][1]+vol[1][0][1]+vol[1][1][0]) + 3. * ( vol[2][0][1] + vol[2][1][0] + vol[1][0][0] ) + vol[2][0][0] ) / 64.;
	w8 = ( 27.*vol[1][1][1] + 9. * (vol[0][1][1]+vol[1][0][1]+vol[1][1][0]) + 3. * ( vol[0][0][1] + vol[0][1][0] + vol[1][0][0] ) + vol[0][0][0] ) / 64.;
	
	return 0.125 * ( v1+v2+v3+v4+v5+v6+v7+v8 );
	//double wsum = w1+w2+w3+w4+w5+w6+w7+w8;
	//return (w1*v1 + w2*v2 + w3*v3 + w4*v4 + w5*v5 + w6*v6 + w7*v7 + w8*v8) / wsum;
}

double integrate_gridfilter_1d(double val[3], double vol[3])
{
	//double val_sum=0;//, w_sum=0;
	double v1, v2, vL, vR;	// vL = v(i-1/4), vR = v(i+1/4)
	double w1, wC, w2;
	
	/*
	                Grid filter
				<--------->
	|_________|__________|__________|
	          o       v1      o        v2      o
	                        vL     vR
	         i-1                i                i+1
	       val_0           val_1          val_2
	*/
	
	
	v1 = (val[0]*vol[0] + val[1]*vol[1]) / ( vol[0]+vol[1] );
	v2 = (val[1]*vol[1] + val[2]*vol[2]) / ( vol[1]+vol[2] );
	
	w1 = 0.25*(vol[0]+vol[1]);		// at v1
	wC = 0.5*vol[1];			// at val[1]
	w2 = 0.25*(vol[1]+vol[2]);		// at v2
	
	
	//0.5*( w1*v1 + wC*val[1] + w2*v2 + wC*val[1] );
	
	vL = 0.5*(vol[0]+vol[1])*v1 + vol[1]* val[1];
	vR = 0.5*(vol[1]+vol[2])*v2 + vol[1] * val[1];
	
	return ( w1*v1 + wC*val[1] + w2*v2 + wC*val[1] ) / ( w1 + wC + w2 + wC );
}

void Calculate_Covariant_metrics(double g[3][3], double G[3][3])
{
	/*
		| csi.x  csi.y csi.z |-1		| x.csi  x.eta x.zet | 
		| eta.x eta.y eta.z |	 =	| y.csi   y.eta  y.zet |
		| zet.x zet.y zet.z |		| z.csi  z.eta z.zet |
	
	*/
	const double a11=g[0][0], a12=g[0][1], a13=g[0][2];
	const double a21=g[1][0], a22=g[1][1], a23=g[1][2];
	const double a31=g[2][0], a32=g[2][1], a33=g[2][2];

	double det= a11*(a33*a22-a32*a23)-a21*(a33*a12-a32*a13)+a31*(a23*a12-a22*a13);
	
	G[0][0] = (a33*a22-a32*a23)/det,	G[0][1] = - (a33*a12-a32*a13)/det, 	G[0][2] = (a23*a12-a22*a13)/det;
	G[1][0] = -(a33*a21-a31*a23)/det, G[1][1] = (a33*a11-a31*a13)/det,	G[1][2] = - (a23*a11-a21*a13)/det;
	G[2][0] = (a32*a21-a31*a22)/det,	G[2][1] = - (a32*a11-a31*a12)/det,	G[2][2] = (a22*a11-a21*a12)/det;
};

void Calculate_normal(Cmpnts csi, Cmpnts eta, Cmpnts zet, double ni[3], double nj[3], double nk[3])
{
	double g[3][3];
	double G[3][3];
	
	g[0][0]=csi.x, g[0][1]=csi.y, g[0][2]=csi.z;
	g[1][0]=eta.x, g[1][1]=eta.y, g[1][2]=eta.z;
	g[2][0]=zet.x, g[2][1]=zet.y, g[2][2]=zet.z;
	
	Calculate_Covariant_metrics(g, G);
	double xcsi=G[0][0], ycsi=G[1][0], zcsi=G[2][0];
	double xeta=G[0][1], yeta=G[1][1], zeta=G[2][1];
	double xzet=G[0][2], yzet=G[1][2], zzet=G[2][2];
	      
	double nx_i = xcsi, ny_i = ycsi, nz_i = zcsi;
	double nx_j = xeta, ny_j = yeta, nz_j = zeta;
	double nx_k = xzet, ny_k = yzet, nz_k = zzet;
	      
	double sum_i=sqrt(nx_i*nx_i+ny_i*ny_i+nz_i*nz_i);
	double sum_j=sqrt(nx_j*nx_j+ny_j*ny_j+nz_j*nz_j);
	double sum_k=sqrt(nx_k*nx_k+ny_k*ny_k+nz_k*nz_k);

//	*Ai = sqrt( g[0][0]*g[0][0] + g[0][1]*g[0][1] + g[0][2]*g[0][2] );	// area
//	*Aj = sqrt( g[1][0]*g[1][0] + g[1][1]*g[1][1] + g[1][2]*g[1][2] );
//	*Ak =sqrt( g[2][0]*g[2][0] + g[2][1]*g[2][1] + g[2][2]*g[2][2] );
		
	nx_i /= sum_i, ny_i /= sum_i, nz_i /= sum_i;
	nx_j /= sum_j, ny_j /= sum_j, nz_j /= sum_j;
	nx_k /= sum_k, ny_k /= sum_k, nz_k /= sum_k;

	ni[0] = nx_i, ni[1] = ny_i, ni[2] = nz_i;
	nj[0] = nx_j, nj[1] = ny_j, nj[2] = nz_j;
	nk[0] = nx_k, nk[1] = ny_k, nk[2] = nz_k;
}

void Calculate_normal_and_area(Cmpnts csi, Cmpnts eta, Cmpnts zet, double ni[3], double nj[3], double nk[3], double *Ai, double *Aj, double *Ak)
{
	double g[3][3];
	double G[3][3];
	
	g[0][0]=csi.x, g[0][1]=csi.y, g[0][2]=csi.z;
	g[1][0]=eta.x, g[1][1]=eta.y, g[1][2]=eta.z;
	g[2][0]=zet.x, g[2][1]=zet.y, g[2][2]=zet.z;
	
	Calculate_Covariant_metrics(g, G);
	double xcsi=G[0][0], ycsi=G[1][0], zcsi=G[2][0];
	double xeta=G[0][1], yeta=G[1][1], zeta=G[2][1];
	double xzet=G[0][2], yzet=G[1][2], zzet=G[2][2];
	      
	double nx_i = xcsi, ny_i = ycsi, nz_i = zcsi;
	double nx_j = xeta, ny_j = yeta, nz_j = zeta;
	double nx_k = xzet, ny_k = yzet, nz_k = zzet;
	      
	double sum_i=sqrt(nx_i*nx_i+ny_i*ny_i+nz_i*nz_i);
	double sum_j=sqrt(nx_j*nx_j+ny_j*ny_j+nz_j*nz_j);
	double sum_k=sqrt(nx_k*nx_k+ny_k*ny_k+nz_k*nz_k);

	*Ai = sqrt( g[0][0]*g[0][0] + g[0][1]*g[0][1] + g[0][2]*g[0][2] );	// area
	*Aj = sqrt( g[1][0]*g[1][0] + g[1][1]*g[1][1] + g[1][2]*g[1][2] );
	*Ak = sqrt( g[2][0]*g[2][0] + g[2][1]*g[2][1] + g[2][2]*g[2][2] );
		
	nx_i /= sum_i, ny_i /= sum_i, nz_i /= sum_i;
	nx_j /= sum_j, ny_j /= sum_j, nz_j /= sum_j;
	nx_k /= sum_k, ny_k /= sum_k, nz_k /= sum_k;

	ni[0] = nx_i, ni[1] = ny_i, ni[2] = nz_i;
	nj[0] = nx_j, nj[1] = ny_j, nj[2] = nz_j;
	nk[0] = nx_k, nk[1] = ny_k, nk[2] = nz_k;
}

void Calculate_dxdydz(PetscReal ajc, Cmpnts csi, Cmpnts eta, Cmpnts zet, double *dx, double *dy, double *dz)
{
	double ni[3], nj[3], nk[3];
	double Li, Lj, Lk;
	double Ai, Aj, Ak;
	double vol = 1./ajc;
	
	Calculate_normal_and_area(csi, eta, zet, ni, nj, nk, &Ai, &Aj, &Ak);
	Li = vol / Ai;
	Lj = vol / Aj;
	Lk = vol / Ak;
	
	// Length scale vector = di * ni_vector + dj * nj_vector + dk * nk_vector
	*dx = fabs( Li * ni[0] + Lj * nj[0] + Lk * nk[0] );
	*dy = fabs( Li * ni[1] + Lj * nj[1] + Lk * nk[1] );
	*dz = fabs( Li * ni[2] + Lj * nj[2] + Lk * nk[2] );
};

void Calculate_Covariant_tensor(double g[3][3], double G[3][3])
{
	/* inversion of contravariant tensor(matrix) */
	const double a11=g[0][0], a12=g[0][1], a13=g[0][2];
	const double a21=g[1][0], a22=g[1][1], a23=g[1][2];
	const double a31=g[2][0], a32=g[2][1], a33=g[2][2];

	double det= a11*(a33*a22-a32*a23)-a21*(a33*a12-a32*a13)+a31*(a23*a12-a22*a13);
	
	G[0][0] = (a33*a22-a32*a23)/det,	G[0][1] = - (a33*a12-a32*a13)/det, 	G[0][2] = (a23*a12-a22*a13)/det;
	G[1][0] = -(a33*a21-a31*a23)/det, 	G[1][1] = (a33*a11-a31*a13)/det,	G[1][2] = - (a23*a11-a21*a13)/det;
	G[2][0] = (a32*a21-a31*a22)/det,	G[2][1] = - (a32*a11-a31*a12)/det,	G[2][2] = (a22*a11-a21*a12)/det;
};

void store_k_section(UserCtx *user, int kplane)
{
	DMDALocalInfo info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int 	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	int i, j, k;
	//int kplane=(mz*6)/10;
	
	if ( !ucat_plane_allocated ) {
		ucat_plane_allocated = 1;
		
		user->ucat_plane = (Cmpnts **)malloc( sizeof(Cmpnts *) * my );
		if(rans) user->komega_plane = (Cmpnts2 **)malloc( sizeof(Cmpnts2 *) * my );
		if(user->temp)
                  user->tmprt_plane = (double **)malloc( sizeof(double *) * my ); // add xiaolei
		if(levelset) user->level_plane = (double **)malloc( sizeof(double *) * my ); // add xiaolei
		
		for(j=0; j<my; j++) {
			user->ucat_plane[j] = (Cmpnts *)malloc( sizeof(Cmpnts) * mx );
			if(rans) user->komega_plane[j] = (Cmpnts2 *)malloc( sizeof(Cmpnts2) * mx );
			if(user->temp)
                          user->tmprt_plane[j] = (double *)malloc( sizeof(double) * mx ); //add xiaolei
			if(levelset) user->level_plane[j] = (double *)malloc( sizeof(double) * mx ); //add xiaolei
		}
	}
	
	Cmpnts ***ucat;
	Cmpnts2 ***komega;
	double ***tmprt; // add xiaolei
	double ***level; // add xiaolei

	
	std::vector< std::vector<Cmpnts> > ucat_plane_tmp (my);
	std::vector< std::vector<Cmpnts2> > komega_plane_tmp (my);
	std::vector< std::vector<double> > tmprt_plane_tmp (my); // add xiaolei
	std::vector< std::vector<double> > level_plane_tmp (my); // add xiaolei
	
	for( j=0; j<my; j++) ucat_plane_tmp[j].resize(mx);
	for( j=0; j<my; j++) komega_plane_tmp[j].resize(mx);
	for( j=0; j<my; j++) tmprt_plane_tmp[j].resize(mx); // add xiaolei
	for( j=0; j<my; j++) level_plane_tmp[j].resize(mx); // add xiaolei
	
	
	for(j=0; j<my; j++)
	for(i=0; i<mx; i++) {
		ucat_plane_tmp[j][i].x = ucat_plane_tmp[j][i].y = ucat_plane_tmp[j][i].z = 0;
		komega_plane_tmp[j][i].x = komega_plane_tmp[j][i].y = 0;
		tmprt_plane_tmp[j][i] = 0;  // add xiaolei
		level_plane_tmp[j][i] = 0;  // add xiaolei
	}
	
	DMDAVecGetArray(user->fda, user->Ucat, &ucat);
	if(rans) DMDAVecGetArray(user->fda2, user->K_Omega, &komega);
	if(user->temp)
          DMDAVecGetArray(user->da, user->gTmprt, &tmprt); // add xiaolei
	if(levelset) DMDAVecGetArray(user->da, user->Levelset, &level); // add xiaolei
	
	for (k=zs; k<ze; k++) {
		if(k==kplane) {
			for (j=ys; j<ye; j++)
			for (i=xs; i<xe; i++) {
				ucat_plane_tmp[j][i] = ucat[k][j][i];
				if(rans) komega_plane_tmp[j][i] = komega[k][j][i];
				if(user->temp) tmprt_plane_tmp[j][i] = tmprt[k][j][i]; // add xiaolei
				if(levelset) level_plane_tmp[j][i] = level[k][j][i]; // add xiaolei
			}
		}
	}
	
	DMDAVecRestoreArray(user->fda, user->Ucat, &ucat);
	if (rans)
          DMDAVecRestoreArray(user->fda2, user->K_Omega, &komega);
	if (user->temp)
          DMDAVecRestoreArray(user->da, user->gTmprt, &tmprt); // add xiaolei
	
	for(j=0; j<my; j++) {
		/*
		MPI_Allreduce( &ucat_plane_tmp[j][0], &user->ucat_plane[j][0], mx*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		if(rans) MPI_Allreduce( &komega_plane_tmp[j][0], &user->komega_plane[j][0], mx*2, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		*/
		MPI_Reduce( &ucat_plane_tmp[j][0], &user->ucat_plane[j][0], mx*3, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		if(rans) MPI_Reduce( &komega_plane_tmp[j][0], &user->komega_plane[j][0], mx*2, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		if(user->temp) MPI_Reduce( &tmprt_plane_tmp[j][0], &user->tmprt_plane[j][0], mx, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD); // add xiaolei
		if(levelset) MPI_Reduce( &level_plane_tmp[j][0], &user->level_plane[j][0], mx, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD); // add xiaolei
	}

};

void save_inflow_section(UserCtx *user)
{
	DMDALocalInfo info = user->info;
	int mx = info.mx, my = info.my, mz = info.mz;
	int j, rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	int ti0=ti-save_inflow_minus;
	
	if(nsave_ksection==0) {
		nsave_ksection=1;
		save_ksection[0] = mz/2;
	}
	
	for(int k=0; k<nsave_ksection; k++) {
		int kplane = save_ksection[k];
		if (kplane<1 || kplane>mz-2) continue;
		
		store_k_section(user, kplane);

		if(!rank) {
                        // WRO - debug
			// static int output = 0;
                        // if (!output)
                        // {
                        //   output = 1;
                        //   printf("WRO - mx,my,mz: %d,%d,%d\n", mx, my, mz);
                        // }
			char fname[256];
			int ti_name = ( (ti0-1) / save_inflow_period ) * save_inflow_period + 1;

			// sprintf(fname, "%s/k%04d_inflow_%06d_dt=%g.dat", path, kplane, ti_name, /*user->dt*/dt_inflow);
			sprintf(fname, "%s/inflow_%06d_dt=%g.dat", path_inflow, ti_name, /*user->dt*/dt_inflow);
			if(ti0==tistart || ti0%save_inflow_period==1) unlink(fname);	// delete existing file

                        // WRO - debug
                        // printf("WRO - Saving inflow ts=%d, k=%d\n", ti0, kplane);

			FILE *fp=fopen(fname, "ab");
			if(!fp) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(1);
			
			for(j=0; j<my; j++)
                        {
                          fwrite(&user->ucat_plane[j][0], sizeof(Cmpnts), mx, fp);
                          // WRO - debug
                          // printf("WRO - byte written %d\n", sizeof(Cmpnts) * mx);
                        }
                        
			fclose(fp);
		
			// rans part
			if(rans) {
				char fname[256];
				int ti_name = ( (ti0-1) / save_inflow_period ) * save_inflow_period + 1;
				
				// sprintf(fname, "%s/k%04d_inflow_ko_%06d_dt=%g.dat", path, kplane, ti_name, /*user->dt*/dt_inflow);
				sprintf(fname, "%s/inflow_ko_%06d_dt=%g.dat", path_inflow, ti_name, /*user->dt*/dt_inflow);
				if(ti0==tistart || ti0%save_inflow_period==1) unlink(fname);	// delete existing file
						
				FILE *fp=fopen(fname, "ab");
				if(!fp) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(1);
				
				for(j=0; j<my; j++) fwrite(&user->komega_plane[j][0], sizeof(Cmpnts2), mx, fp);
				fclose(fp);
			}


			// add (xiaolei)
			if(user->temp) {
				char fname[256];
				int ti_name = ( (ti0-1) / save_inflow_period ) * save_inflow_period + 1;
		
				sprintf(fname, "%s/inflow_t_%06d_dt=%g.dat", path_inflow, ti_name, user->dt);
				if(ti0==tistart || ti0%save_inflow_period==1) unlink(fname);	// delete existing file
				
				FILE *fp=fopen(fname, "ab");
				if(!fp) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(1);
		
				for(j=0; j<my; j++) fwrite(&user->tmprt_plane[j][0], sizeof(double), mx, fp);
				fclose(fp);
			}


			// add (xiaolei)
			if(levelset) {
				char fname[256];
				int ti_name = ( (ti0-1) / save_inflow_period ) * save_inflow_period + 1;
		
				sprintf(fname, "%s/inflow_l_%06d_dt=%g.dat", path_inflow, ti_name, user->dt);
				if(ti0==tistart || ti0%save_inflow_period==1) unlink(fname);	// delete existing file
				
				FILE *fp=fopen(fname, "ab");
				if(!fp) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(1);
		
				for(j=0; j<my; j++) fwrite(&user->level_plane[j][0], sizeof(double), mx, fp);
				fclose(fp);
			}




		}
	}
}

void read_inflow_section(UserCtx *user)
{
	DMDALocalInfo info = user->info;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int i, j;
	int rank;
	
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		
	if(nsave_ksection==0) {
		nsave_ksection=1;
		save_ksection[0] = mz/2;
	}
	
	if ( !ucat_plane_allocated ) {
		ucat_plane_allocated = 1;
		
		user->ucat_plane = (Cmpnts **)malloc( sizeof(Cmpnts *) * my );
		if(rans) user->komega_plane = (Cmpnts2 **)malloc( sizeof(Cmpnts2 *) * my );
		if(user->temp) user->tmprt_plane = (double **)malloc( sizeof(double *) * my ); // xiaolei
		if(levelset) user->level_plane = (double **)malloc( sizeof(double *) * my ); // xiaolei

		for(j=0; j<my; j++) {
			user->ucat_plane[j] = (Cmpnts *)malloc( sizeof(Cmpnts) * mx );
			if(rans) user->komega_plane[j] = (Cmpnts2 *)malloc( sizeof(Cmpnts2) * mx );
			if(user->temp) user->tmprt_plane[j] = (double *)malloc( sizeof(double) * mx ); // xiaolei
			if(levelset) user->level_plane[j] = (double *)malloc( sizeof(double) * mx ); // xiaolei
		}
	}
	
	std::vector< std::vector<Cmpnts> > ucat_plane_tmp (my);
	std::vector< std::vector<Cmpnts2> > komega_plane_tmp (my);
	std::vector< std::vector<double> > tmprt_plane_tmp (my);
	std::vector< std::vector<double> > level_plane_tmp (my);
	
	for( j=0; j<my; j++) ucat_plane_tmp[j].resize(mx);
	for( j=0; j<my; j++) komega_plane_tmp[j].resize(mx);
	for( j=0; j<my; j++) tmprt_plane_tmp[j].resize(mx);
	for( j=0; j<my; j++) level_plane_tmp[j].resize(mx);
	
	
	for(j=0; j<my; j++)
	for(i=0; i<mx; i++) {
		ucat_plane_tmp[j][i].x = ucat_plane_tmp[j][i].y = ucat_plane_tmp[j][i].z = 0;
		komega_plane_tmp[j][i].x = komega_plane_tmp[j][i].y = 0;
		tmprt_plane_tmp[j][i] = 0;
		level_plane_tmp[j][i] = 0;
	}
	
	char fname[256];
	//int ti2=ti-save_inflow_minus;
	int ti2=ti+ti_lastsave;

	if(ti2==0) ti2=1;
		
	if(ti2>inflow_recycle_perioid) {
		ti2 -= (ti2/inflow_recycle_perioid) * inflow_recycle_perioid;
	}
	int ti_name = ( (ti2-1) / read_inflow_period ) * read_inflow_period + 1;

	int kplane = save_ksection[0];
//	if (kplane<1 || kplane>mz-2) continue;
			
	// sprintf(fname, "%s/k%04d_inflow_%06d_dt=%g.dat", path_inflow, kplane, ti_name, /*user->dt*/dt_inflow);

	sprintf(fname, "%s/inflow_%06d_dt=%g.dat", path_inflow, ti_name, /*user->dt*/dt_inflow);
        

        PetscPrintf(PETSC_COMM_WORLD, "read_inflow_section\n");
        PetscPrintf(PETSC_COMM_WORLD, "ti=%d, ti2=%d, tistart=%d, ti_lastsave=%d, ti_name=%d, inflow_recycle_perioid=%d, read_inflow_period=%d\n", ti, ti2, tistart, ti_lastsave, ti_name, inflow_recycle_perioid, read_inflow_period);

        
	if(!rank) {
		if(ti==tistart || ti2%read_inflow_period==1) {
			if(ti!=tistart) fclose(user->fp_inflow_u);
			user->fp_inflow_u=fopen(fname, "rb");
			if(!user->fp_inflow_u) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(1);
			
			if(ti==tistart) {	// move position
				for(int it=0; it<(ti2-1)%read_inflow_period; it++)
                                {
                                  for(j=0; j<my; j++)
                                  {
                                    size_t read = fread(&ucat_plane_tmp[j][0], sizeof(Cmpnts), mx, user->fp_inflow_u);
                                    if (read != mx)
                                    {
                                      PetscPrintf(PETSC_COMM_WORLD, "Error reading file %s (most likely end of file)\n", fname);
                                      exit(1);
                                    }
                                  }
				}
			}
		}
		if(tistart==0 && ti==1) {}
		else
                {
                  for(j=0; j<my; j++)
                  {
                    size_t read = fread(&ucat_plane_tmp[j][0], sizeof(Cmpnts), mx, user->fp_inflow_u);
                    if (read != mx)
                    {
                      PetscPrintf(PETSC_COMM_WORLD, "Error reading file %s (most likely end of file)\n", fname);
                      exit(1);
                    }
                  }
                }
	}
	for(j=0; j<my; j++) MPI_Allreduce( &ucat_plane_tmp[j][0], &user->ucat_plane[j][0], mx*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
	PetscPrintf(PETSC_COMM_WORLD, "\nRead inflow data from %s ... \n", fname);
	
	if(rans) {
		if(!rank) {
			char fname[256];
			int ti2=ti;
			
			if(ti2>inflow_recycle_perioid) {
				ti2 -= (ti2/inflow_recycle_perioid) * inflow_recycle_perioid;
			}
			int ti_name = ( (ti2-1) / read_inflow_period ) * read_inflow_period + 1;
			
			sprintf(fname, "%s/inflow_ko_%06d_dt=%g.dat", path_inflow, ti_name, /*user->dt*/dt_inflow);
			
			if(ti==tistart || (ti>tistart+90 && ti2%read_inflow_period==1) ) {
			  //if(tistart!=0 && tistart%read_inflow_period!=1) printf("\n******************* Warning !  rstart should be multiple of 100 *******************\n");
				
				if(ti!=tistart) fclose(user->fp_inflow_ko);
				user->fp_inflow_ko=fopen(fname, "rb");
				if(!user->fp_inflow_ko) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(1);
					
				if(ti==tistart) {	// move position; 100616
					for(int it=0; it<(ti2-1)%read_inflow_period; it++) {
						for(j=0; j<my; j++) fread(&komega_plane_tmp[j][0], sizeof(Cmpnts2), mx, user->fp_inflow_ko);
					}
				}
			}
			
			for(j=0; j<my; j++) fread(&komega_plane_tmp[j][0], sizeof(Cmpnts2), mx, user->fp_inflow_ko);
			
			printf("\nRead inflow data from %s ! \n", fname);
		}
		for(j=0; j<my; j++) MPI_Allreduce( &komega_plane_tmp[j][0], &user->komega_plane[j][0], mx*2, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
	}


	
//	temperature inflow	add (xiaolei)
	if(user->temp) {
		if(!rank) {
			char fname[256];
			int ti2=ti;
			
			if(ti2>inflow_recycle_perioid) {
				ti2 -= (ti2/inflow_recycle_perioid) * inflow_recycle_perioid;
			}
			int ti_name = ( (ti2-1) / save_inflow_period ) * save_inflow_period + 1;
			
			sprintf(fname, "%s/inflow_t_%06d_dt=%g.dat", path_inflow, ti_name, user->dt);
			
			if(ti==tistart || (ti>tistart+90 && ti2%save_inflow_period==1) ) {
			  //if(tistart!=0 && tistart%save_inflow_period!=1) printf("\n******************* Warning !  rstart should be multiple of 100 *******************\n");
				
				if(ti!=tistart) fclose(user->fp_inflow_t);
				user->fp_inflow_t=fopen(fname, "rb");
				if(!user->fp_inflow_t) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(1);
					
				if(ti==tistart) {	// move position; 100616
					for(int it=0; it<(ti2-1)%save_inflow_period; it++) {
						for(j=0; j<my; j++) fread(&tmprt_plane_tmp[j][0], sizeof(double), mx, user->fp_inflow_t);
					}
				}
			}
			
			if(tistart==0 && ti==1) {}
			else for(j=0; j<my; j++) fread(&tmprt_plane_tmp[j][0], sizeof(double), mx, user->fp_inflow_t);
			
			printf("\nRead inflow data from %s ! \n", fname);
		}
		for(j=0; j<my; j++) MPI_Allreduce( &tmprt_plane_tmp[j][0], &user->tmprt_plane[j][0], mx, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
	}


	
//	levelset inflow	add (xiaolei)
	if(levelset && inflow_levelset) {
		if(!rank) {
			char fname[256];
			int ti2=ti;
			
			if(ti2>inflow_recycle_perioid) {
				ti2 -= (ti2/inflow_recycle_perioid) * inflow_recycle_perioid;
			}
			int ti_name = ( (ti2-1) / save_inflow_period ) * save_inflow_period + 1;
			
			sprintf(fname, "%s/inflow_l_%06d_dt=%g.dat", path_inflow, ti_name, user->dt);
			
			if(ti==tistart || (ti>tistart+90 && ti2%save_inflow_period==1) ) {
			  //if(tistart!=0 && tistart%save_inflow_period!=1) printf("\n******************* Warning !  rstart should be multiple of 100 *******************\n");
				
				if(ti!=tistart) fclose(user->fp_inflow_t);
				user->fp_inflow_t=fopen(fname, "rb");
				if(!user->fp_inflow_t) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(1);
					
				if(ti==tistart) {	// move position; 100616
					for(int it=0; it<(ti2-1)%save_inflow_period; it++) {
						for(j=0; j<my; j++) fread(&level_plane_tmp[j][0], sizeof(double), mx, user->fp_inflow_t);
					}
				}
			}
			
			if(tistart==0 && ti==1) {}
			else for(j=0; j<my; j++) fread(&level_plane_tmp[j][0], sizeof(double), mx, user->fp_inflow_t);
			
			printf("\nRead inflow data from %s ! \n", fname);
		}
		for(j=0; j<my; j++) MPI_Allreduce( &level_plane_tmp[j][0], &user->level_plane[j][0], mx, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
	}



}

double lforce_ibm, force_ibm;
double larea_ibm, area_ibm;

void Calc_ShearStress(UserCtx *user)
{
	int i, j, k;

	DM da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int	lxs, lxe, lys, lye, lzs, lze;
	
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;
  
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;
  
	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
  
	Cmpnts ***csi, ***eta, ***zet, ***ucat, ***ucat_sum, ***cent;
	Cmpnts ***icsi, ***ieta, ***izet;
	Cmpnts ***jcsi, ***jeta, ***jzet;
	PetscReal ***nvert, ***aj, ***ustar, ***p, ***p_sum;
	PetscReal ***iaj, ***jaj;

	double N=(double)ti-1.0;
	
	Vec lUcat_sum;

	if(averaging && ti>1) {
	  VecDuplicate(user->lUcat, &lUcat_sum);
	  
	  DMGlobalToLocalBegin(fda, user->Ucat_sum, INSERT_VALUES, lUcat_sum);
	  DMGlobalToLocalEnd(fda, user->Ucat_sum, INSERT_VALUES, lUcat_sum);
	  VecScale(lUcat_sum, 1./N);
		DMDAVecGetArray(fda, lUcat_sum, &ucat_sum);
		DMDAVecGetArray(da, user->P_sum, &p_sum);
	}
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lP, &p);
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);
	DMDAVecGetArray(fda, user->lUcat, &ucat);
	DMDAVecGetArray(fda, user->lCent, &cent);
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(da, user->lUstar, &ustar);
	
	DMDAVecGetArray(fda, user->lICsi, &icsi);
        DMDAVecGetArray(fda, user->lIEta, &ieta);
        DMDAVecGetArray(fda, user->lIZet, &izet);
	DMDAVecGetArray(da, user->lIAj, &iaj);
	
	DMDAVecGetArray(fda, user->lJCsi, &jcsi);
        DMDAVecGetArray(fda, user->lJEta, &jeta);
        DMDAVecGetArray(fda, user->lJZet, &jzet);
	DMDAVecGetArray(da, user->lJAj, &jaj);

	double lforce[6]={0,0,0,0,0,0}, force[6]={0,0,0,0,0,0}, larea[6]={0,0,0,0,0,0}, area[6]={0,0,0,0,0,0};
	double lforce_avg[6]={0,0,0,0,0,0}, force_avg[6]={0,0,0,0,0,0};
	double lforce_avg_ibm=0, force_avg_ibm=0;
	
	double lforce_skin[6]={0,0,0,0,0,0};
        double force_skin[6]={0,0,0,0,0,0};

	double lforce_pressure[6]={0,0,0,0,0,0};
	double force_pressure[6]={0,0,0,0,0,0};
	
	double dudc_avg, dvdc_avg, dwdc_avg, dude_avg, dvde_avg, dwde_avg, dudz_avg, dvdz_avg, dwdz_avg;
	double du_dx_avg, du_dy_avg, du_dz_avg, dv_dx_avg, dv_dy_avg, dv_dz_avg, dw_dx_avg, dw_dy_avg, dw_dz_avg;

	if (user->bctype[0]==1) {	// left wall
		if (xs==0) {
			i = 0;
			for (k=lzs; k<lze; k++)
			for (j=lys; j<lye; j++) {
				if (nvert[k][j][i+1] < 0.1) {
					double W;
					double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
					Compute_du_i (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde,&dudz, &dvdz, &dwdz);

					double ajc = iaj[k][j][i];
					double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;

					double csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
					double eta0 = ieta[k][j][i].x, eta1 = ieta[k][j][i].y, eta2 = ieta[k][j][i].z;
					double zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;

					Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
								&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
					
					double ni[3], nj[3], nk[3];
					double nx, ny, nz;
					Calculate_normal(icsi[k][j][i], ieta[k][j][i], izet[k][j][i], ni, nj, nk);
					nx = ni[0]; //inward normal
					ny = ni[1]; //inward normal
					nz = ni[2]; //inward normal
					double i_area = sqrt( icsi[k][j][i].x*icsi[k][j][i].x + icsi[k][j][i].y*icsi[k][j][i].y + icsi[k][j][i].z*icsi[k][j][i].z );
					double Fs, Fp;
										
					if( fabs(csi0)<1.e-9 && fabs(zet2)<1.e-9 ) { // x_dir is streamwise, z dir is the vertical, y is the spanwise
						Fs = (du_dx * nx + du_dy * ny + du_dz * nz) / user->ren * i_area;
						Fp = - p[k][j][i+1] * csi0;
						larea[0] += fabs(csi1);
					}
					else {
						Fs = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * i_area;
						Fp = - p[k][j][i+1] * csi2;
						larea[0] += fabs(csi0);
					}
					
					
					lforce_skin[0] += Fs;
					lforce_pressure[0] += Fp;
					lforce[0] += Fs + Fp;
				}
			}
		}
	}
	
	if (user->bctype[1]==1) {	// left wall
		if (xe==mx) {
			i = mx-2;
			for (k=lzs; k<lze; k++)
			for (j=lys; j<lye; j++) {
				if (nvert[k][j][i] < 0.1) {
					double W;
					double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
					Compute_du_i (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde,&dudz, &dvdz, &dwdz);

					double ajc = iaj[k][j][i];
					double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;

					double csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
					double eta0 = ieta[k][j][i].x, eta1 = ieta[k][j][i].y, eta2 = ieta[k][j][i].z;
					double zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;

					Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
							 &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
					
					double ni[3], nj[3], nk[3];
					double nx, ny, nz;
					Calculate_normal(icsi[k][j][i], ieta[k][j][i], izet[k][j][i], ni, nj, nk);
					nx = -ni[0]; //inward normal
					ny = -ni[1]; //inward normal
					nz = -ni[2]; //inward normal
					double i_area = sqrt( icsi[k][j][i].x*icsi[k][j][i].x + icsi[k][j][i].y*icsi[k][j][i].y + icsi[k][j][i].z*icsi[k][j][i].z );
					double Fs, Fp;
										
					if( fabs(csi0)<1.e-9 && fabs(zet2)<1.e-9 ) { // x_dir is streamwise, z dir is the vertical, y is the spanwise
						Fs = (du_dx * nx + du_dy * ny + du_dz * nz) / user->ren * i_area;
						Fp = - p[k][j][i] * csi0;
						larea[1] += fabs(csi1);
					}
					else {
						Fs = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * i_area;
						Fp = - p[k][j][i] * csi2;
						larea[1] += fabs(csi0);
					}
					
					lforce_skin[1] += Fs;
					lforce_pressure[1] += Fp;
					lforce[1] += Fs + Fp;
				}
			}
		}
	}

	if (user->bctype[2] == 1 ) {	// bottom wall
		if (ys==0) {
			j = 0;
			for (k=lzs; k<lze; k++)
			for (i=lxs; i<lxe; i++) {
				if (nvert[k][j+1][i] < 0.1) {
					double W, Fz_avg=0;
					double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
					Compute_du_j (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde,&dudz, &dvdz, &dwdz);

					double ajc = jaj[k][j][i];
					double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;

					double csi0 = jcsi[k][j][i].x, csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
					double eta0 = jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
					double zet0 = jzet[k][j][i].x, zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;

					Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
							 &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
					
					double ni[3], nj[3], nk[3];
					double nx, ny, nz;
					Calculate_normal(jcsi[k][j][i], jeta[k][j][i], jzet[k][j][i], ni, nj, nk);
					nx = nj[0]; //inward normal
					ny = nj[1]; //inward normal
					nz = nj[2]; //inward normal
					
					double j_area = sqrt( jeta[k][j][i].x*jeta[k][j][i].x + jeta[k][j][i].y*jeta[k][j][i].y + jeta[k][j][i].z*jeta[k][j][i].z );
					double Fs, Fp;
					
					if( fabs(csi0)<1.e-9 && fabs(zet2)<1.e-9 ) { // x_dir is streamwise, z dir is the vertical, y is the spanwise
						Fs = (du_dx * nx + du_dy * ny + du_dz * nz) / user->ren * j_area;
						Fp = - p[k][j+1][i] * eta0;
						larea[2] += fabs(eta2);
					}
					else {
						Fs = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * j_area;
						Fp = - p[k][j+1][i] * eta2;
						larea[2] += fabs(eta1);
					}

					if(averaging && ti>1) {
					  Compute_du_j (i, j, k, mx, my, mz, ucat_sum, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde,&dudz, &dvdz, &dwdz);
					  Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
							   &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

						double Fs_avg = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * j_area;
						double Fp_avg = - p_sum[k][j+1][i] * eta2 / N;
						dw_dy = ( dwdc * csi[k][j][i].y + dwde * eta[k][j][i].y + dwdz * zet[k][j][i].y ) * aj[k][j][i];
						lforce_avg[2] += ( dw_dy / user->ren ) * fabs(eta[k][j][i].y) + Fp_avg;
                                        }
					
					lforce_skin[2] += Fs;
					lforce_pressure[2] += Fp;
					lforce[2] += Fs + Fp;
					
				}
			}
		}
	}

	if (user->bctype[3] == 1 ) {	// top wall
		if (ye==my) {
			j = my-2;
			for (k=lzs; k<lze; k++)
			for (i=lxs; i<lxe; i++) {

				double A = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
				double dy = 1./aj[k][j][i]/A;
				dy *= 0.5;
				
				if (nvert[k][j][i] < 0.1) {
					double W=0, Fz_avg=0;
					double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
                                        Compute_du_j (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);

					double ajc = jaj[k][j][i];
					double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;

					double csi0 = jcsi[k][j][i].x, csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
					double eta0 = jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
					double zet0 = jzet[k][j][i].x, zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;

					Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
							 &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

					double ni[3], nj[3], nk[3];
					double nx, ny, nz;
					Calculate_normal(csi[k][j-1][i], eta[k][j-1][i], zet[k][j-1][i], ni, nj, nk);
					nx = nj[0]*(-1); //inward normal
					ny = nj[1]*(-1); //inward normal
					nz = nj[2]*(-1); //inward normal
					double j_area = sqrt( jeta[k][j][i].x*jeta[k][j][i].x + jeta[k][j][i].y*jeta[k][j][i].y + jeta[k][j][i].z*jeta[k][j][i].z );
					double Fs = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * j_area;
					//double Fs = dw_dy / user->ren * fabs(eta1);
					double Fp = - p[k][j][i] * eta2; // normal stress
					
					lforce_skin[3] += Fs;
					lforce_pressure[3] += Fp;
					lforce[3] += Fs + Fp;
					larea[3] += /*j_area;*/fabs(eta1);
				}
			}
		}
	}
	
			
	if (user->bctype[0]==-1 || user->bctype[0]==-2 || imin_wm != 0 ) {	// wall function // xyang -3 -4

        	larea[0] = 0.0;
	        lforce[0] = 0.0;


		if (xs==0) {
			i = 1;
			for (k=lzs; k<lze; k++)
			for (j=lys; j<lye; j++) {
				double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
				double sb, sc; 
				double ni[3], nj[3], nk[3];
				Cmpnts Uc;
				
				sb = 0.5/aj[k][j][i]/area;
				
				if(i==1) {
					sc = 2*sb + 0.5/aj[k][j][i+1]/area;
					Uc = ucat[k][j][i+1];
				}
				else {
					sc = 2*sb + 0.5/aj[k][j][i-1]/area;
					Uc = ucat[k][j][i-1];
				}
				
				Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
				if(i==my-2) ni[0]*=-1, ni[1]*=-1, ni[2]*=-1;
				if(j==my-2) nj[0]*=-1, nj[1]*=-1, nj[2]*=-1;
				if(k==mz-2) nk[0]*=-1, nk[1]*=-1, nk[2]*=-1;
					
				double nx = nj[0], ny = nj[1], nz = nj[2];
					
				double A_x = 0.5*(csi[k][j][i].x+csi[k][j][i-1].x) + 0.5*(eta[k][j][i].x+eta[k][j-1][i].x) + 0.5*(zet[k][j][i].x+zet[k-1][j][i].x) ;
				double A_y = 0.5*(csi[k][j][i].y+csi[k][j][i-1].y) + 0.5*(eta[k][j][i].y+eta[k][j-1][i].y) + 0.5*(zet[k][j][i].y+zet[k-1][j][i].y) ;
				double A_z = 0.5*(csi[k][j][i].z+csi[k][j][i-1].z) + 0.5*(eta[k][j][i].z+eta[k][j-1][i].z) + 0.5*(zet[k][j][i].z+zet[k-1][j][i].z) ;
				double A_n = fabs(A_x * nx + A_y * ny + A_z * nz);
				
				//if( (int)(nvert[k][j][i])==1 ) 
				  {
					lforce[0] += ustar[k][j][i] * ustar[k][j][i] * A_n;
					larea[0] += A_n;
				}
			}
		}
	}
	
		
	if (user->bctype[1]==-1 || user->bctype[1]==-2 || imax_wm != 0 ) {	// wall function // xyang -3 -4

        	larea[1] = 0.0;
	        lforce[1] = 0.0;


		if (xe==mx) {
			i = mx-2;
			for (k=lzs; k<lze; k++)
			for (j=lys; j<lye; j++) {
				double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
				double sb, sc; 
				double ni[3], nj[3], nk[3];
				Cmpnts Uc;
				
				sb = 0.5/aj[k][j][i]/area;
				
				if(i==1) {
					sc = 2*sb + 0.5/aj[k][j][i+1]/area;
					Uc = ucat[k][j][i+1];
				}
				else {
					sc = 2*sb + 0.5/aj[k][j][i-1]/area;
					Uc = ucat[k][j][i-1];
				}
				
				Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
				if(i==my-2) ni[0]*=-1, ni[1]*=-1, ni[2]*=-1;
				if(j==my-2) nj[0]*=-1, nj[1]*=-1, nj[2]*=-1;
				if(k==mz-2) nk[0]*=-1, nk[1]*=-1, nk[2]*=-1;
					
				double nx = nj[0], ny = nj[1], nz = nj[2];
					
				double A_x = 0.5*(csi[k][j][i].x+csi[k][j][i-1].x) + 0.5*(eta[k][j][i].x+eta[k][j-1][i].x) + 0.5*(zet[k][j][i].x+zet[k-1][j][i].x) ;
				double A_y = 0.5*(csi[k][j][i].y+csi[k][j][i-1].y) + 0.5*(eta[k][j][i].y+eta[k][j-1][i].y) + 0.5*(zet[k][j][i].y+zet[k-1][j][i].y) ;
				double A_z = 0.5*(csi[k][j][i].z+csi[k][j][i-1].z) + 0.5*(eta[k][j][i].z+eta[k][j-1][i].z) + 0.5*(zet[k][j][i].z+zet[k-1][j][i].z) ;
				double A_n = fabs(A_x * nx + A_y * ny + A_z * nz);
				
				//if( (int)(nvert[k][j][i])==1 ) 
				  {
					lforce[1] += ustar[k][j][i] * ustar[k][j][i] * A_n;
					larea[1] += A_n;
				}
			}
		}
	}
	


	
	if (user->bctype[2]==-1 || user->bctype[2]==-2 || jmin_wm!=0) {	// wall function   jmin_wm (xiaolei)
        	larea[2] = 0.0;
	        lforce[2] = 0.0;


		if (ys==0) {
			j = 1;
			for (k=lzs; k<lze; k++)
			for (i=lxs; i<lxe; i++) {
				double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
				double sb, sc; 
				double ni[3], nj[3], nk[3];
				Cmpnts Uc;
				
				sb = 0.5/aj[k][j][i]/area;
				
				if(j==1) {
					sc = 2*sb + 0.5/aj[k][j+1][i]/area;
					Uc = ucat[k][j+1][i];
				}
				else {
					sc = 2*sb + 0.5/aj[k][j-1][i]/area;
					Uc = ucat[k][j-1][i];
				}
				
				Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
				if(i==my-2) ni[0]*=-1, ni[1]*=-1, ni[2]*=-1;
				if(j==my-2) nj[0]*=-1, nj[1]*=-1, nj[2]*=-1;
				if(k==mz-2) nk[0]*=-1, nk[1]*=-1, nk[2]*=-1;
					
				double nx = nj[0], ny = nj[1], nz = nj[2];
					
				double A_x = 0.5*(csi[k][j][i].x+csi[k][j][i-1].x) + 0.5*(eta[k][j][i].x+eta[k][j-1][i].x) + 0.5*(zet[k][j][i].x+zet[k-1][j][i].x) ;
				double A_y = 0.5*(csi[k][j][i].y+csi[k][j][i-1].y) + 0.5*(eta[k][j][i].y+eta[k][j-1][i].y) + 0.5*(zet[k][j][i].y+zet[k-1][j][i].y) ;
				double A_z = 0.5*(csi[k][j][i].z+csi[k][j][i-1].z) + 0.5*(eta[k][j][i].z+eta[k][j-1][i].z) + 0.5*(zet[k][j][i].z+zet[k-1][j][i].z) ;
				double A_n = fabs(A_x * nx + A_y * ny + A_z * nz);
				
				if(averaging && ti>1) {
					double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
					double sb, sc, ustar_avg;
					double ni[3], nj[3], nk[3];
					Cmpnts Uc, Ua, Ub;
					
					Ua.x = Ua.y = Ua.z = 0;
					sb = 0.5/aj[k][j][i]/area;
					/*
					if(j==1) {
						sc = 2*sb + 0.5/aj[k][j+1][i]/area;
						Uc = ucat_sum[k][j+1][i];
					}
					else {
						sc = 2*sb + 0.5/aj[k][j-1][i]/area;
						Uc = ucat_sum[k][j-1][i];
					}
					Uc.x /= N;
					Uc.y /= N;
					Uc.z /= N;
					
					Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
					if(j==my-2) nj[0]*=-1, nj[1]*=-1, nj[2]*=-1;
					
					double ren = user->ren;
					if(roughness_size>1.e-10) wall_function_roughness_loglaw (1./ren, roughness_size, sc, sb, Ua, Uc, &Ub, &ustar[k][j][i], nj[0], nj[1], nj[2]);
					else {
						wall_function (1./ren, sc, sb, Ua, Uc, &Ub, &ustar_avg, nj[0], nj[1], nj[2]);
					}
					*/
					/*************************************/
					Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
					if(j==my-2) nj[0]*=-1, nj[1]*=-1, nj[2]*=-1;
					
					AxC ( 1., ucat_sum[k][j][i], &Ub);
					double nu = 1./user->ren;
					double nx=nj[0], ny=nj[1], nz=nj[2];
					double Ub_x = Ub.x - Ua.x, Ub_y = Ub.y - Ua.y, Ub_z = Ub.z - Ua.z;
					double un = Ub_x * nx + Ub_y * ny + Ub_z * nz;
					double Ub_t_x = Ub_x - un * nx, Ub_t_y = Ub_y - un * ny, Ub_t_z = Ub_z - un * nz;
					double Ub_t_mag = sqrt( Ub_t_x*Ub_t_x + Ub_t_y*Ub_t_y + Ub_t_z*Ub_t_z );
					if(roughness_size>1.e-10) ustar_avg = find_utau_loglaw_roughness(nu, Ub_t_mag, sb, 0.01, roughness_size);
					else ustar_avg = find_utau_Cabot(nu, Ub_t_mag, sb, 0.01, 0);
					/*************************************/
					
					lforce_avg[2] += ( ustar_avg * ustar_avg ) * A_n;
				}
				
				lforce[2] += ustar[k][j][i] * ustar[k][j][i] * A_n;
				larea[2] += A_n;
			}
		}
	}
	
	if (user->bctype[3]==-1 || user->bctype[3]==-2 || jmax_wm!=0) {	// wall function  jmax_wm (xiaolei)
        	larea[3] = 0.0;
	        lforce[3] = 0.0;


		if (ye==my) {
			j = my-2;
			for (k=lzs; k<lze; k++)
			for (i=lxs; i<lxe; i++) {
				double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
				double sb, sc; 
				double ni[3], nj[3], nk[3];
				Cmpnts Uc;
				
				sb = 0.5/aj[k][j][i]/area;
				
				if(j==1) {
					sc = 2*sb + 0.5/aj[k][j+1][i]/area;
					Uc = ucat[k][j+1][i];
				}
				else {
					sc = 2*sb + 0.5/aj[k][j-1][i]/area;
					Uc = ucat[k][j-1][i];
				}
				
				Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
				if(i==my-2) ni[0]*=-1, ni[1]*=-1, ni[2]*=-1;
				if(j==my-2) nj[0]*=-1, nj[1]*=-1, nj[2]*=-1;
				if(k==mz-2) nk[0]*=-1, nk[1]*=-1, nk[2]*=-1;
					
				double nx = nj[0], ny = nj[1], nz = nj[2];
					
				double A_x = 0.5*(csi[k][j][i].x+csi[k][j][i-1].x) + 0.5*(eta[k][j][i].x+eta[k][j-1][i].x) + 0.5*(zet[k][j][i].x+zet[k-1][j][i].x) ;
				double A_y = 0.5*(csi[k][j][i].y+csi[k][j][i-1].y) + 0.5*(eta[k][j][i].y+eta[k][j-1][i].y) + 0.5*(zet[k][j][i].y+zet[k-1][j][i].y) ;
				double A_z = 0.5*(csi[k][j][i].z+csi[k][j][i-1].z) + 0.5*(eta[k][j][i].z+eta[k][j-1][i].z) + 0.5*(zet[k][j][i].z+zet[k-1][j][i].z) ;
				double A_n = fabs(A_x * nx + A_y * ny + A_z * nz);
				
				//if( (int)(nvert[k][j][i])==1 ) 
				  {
					lforce[3] += ustar[k][j][i] * ustar[k][j][i] * A_n;
					larea[3] += A_n;
				}
			}
		}
	}
	
	// ib shear stress
	lforce_ibm=0, force_ibm=0;
	larea_ibm=0, area_ibm=1.e-10;
	
	
	if(immersed) {
		for(int ibi=0; ibi<NumberOfBodies; ibi++) {
			
			IBMNodes *ibm = &ibm_ptr[ibi];
			
			IBMListNode *current;
			current = user->ibmlist[ibi].head;
			while (current) {
				IBMInfo *ibminfo = &current->ibm_intp;
				
				
				
				int ni = ibminfo->cell;
				current = current->next;
				double sb = ibminfo->d_s;//, sc = sb + ibminfo->d_i;
				
				double nx = ibm->nf_x[ni], ny = ibm->nf_y[ni], nz = ibm->nf_z[ni];
				
				i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;
				
				double u_b = ucat[k][j][i].x, v_b = ucat[k][j][i].y, w_b = ucat[k][j][i].z;
				
				double A_x = csi[k][j][i].x, A_y = eta[k][j][i].y, A_z = zet[k][j][i].z;
				double A_n = fabs(A_x * nx) + fabs(A_y * ny) + fabs(A_z * nz);
				
				/*
				A_n =
					Ai * fabs( n_i[0]*nx + n_i[1]*ny + n_i[2]*nz ) +
					Aj * fabs( n_j[0]*nx + n_j[1]*ny + n_j[2]*nz ) +
					Ak * fabs( n_k[0]*nx + n_k[1]*ny + n_k[2]*nz );
				*/
								
				//printf("%f %f %f\n", Ai * fabs( n_i[0]*nx + n_i[1]*ny + n_i[2]*nz ), Aj * fabs( n_j[0]*nx + n_j[1]*ny + n_j[2]*nz ), Ak * fabs( n_k[0]*nx + n_k[1]*ny + n_k[2]*nz ));
				
				//A_n = sqrt(A_t*A_t + A_s*A_s);
								
				if((wallfunction || IB_wm) && ti>0) {
					lforce_ibm += ustar[k][j][i] * ustar[k][j][i] * A_n;
					//PetscPrintf(PETSC_COMM_WORLD, "ustar=%le\n", ustar[k][j][i]);
				}
				else  {
					// Normal shear force = nu * dW/dn * area 
					//lforce_ibm += 1./user->ren * ( w_c / sc ) * A_n;
					double un = u_b * nx + v_b * ny + w_b * nz;
					double ut = u_b - un * nx, vt = v_b - un * ny, wt = w_b - un * nz;
					double ut_mag = sqrt ( ut*ut + vt*vt + wt*wt );
					
					lforce_ibm += 1./user->ren * ( ut_mag / sb ) * A_n;
					
					if(averaging && ti>1) {
						u_b = ucat_sum[k][j][i].x, v_b = ucat_sum[k][j][i].y, w_b = ucat_sum[k][j][i].z;
						ut = u_b - un * nx, vt = v_b - un * ny, wt = w_b - un * nz;
						ut_mag = sqrt ( ut*ut + vt*vt + wt*wt );
						
						lforce_avg_ibm += 1./user->ren * ( ut_mag / sb ) * A_n;
					}
					
					
					//printf("%f %f %f\n", u_b, v_b, w_b);
					//lforce_ibm += 1./user->ren * ( w_b / sb ) * A_n;
				}
				larea_ibm += A_n;
			};
		}
		
	}
	
	
	double sum_force=0, sum_area=0;//, Re_tau;
	double sum_force_avg=0;
	
	for(i=0; i<6; i++) {
		GlobalSum_All(&lforce[i], &force[i], PETSC_COMM_WORLD);
		GlobalSum_All(&lforce_skin[i], &force_skin[i], PETSC_COMM_WORLD);
		GlobalSum_All(&lforce_pressure[i] , &force_pressure[i], PETSC_COMM_WORLD);
		GlobalSum_All(&lforce_avg[i], &force_avg[i], PETSC_COMM_WORLD);
		GlobalSum_All(&larea[i], &area[i], PETSC_COMM_WORLD);
		
		area[i] = std::max ( area[i], 1.e-10 );
		
		user->shear_stress[i] = force[i] / (area[i]) ;
		user->shear_force_avg[i] = force_avg[i];
		
		if( fabs(force[i]) > 1.e-10 ) {
			sum_force += force[i];		// sum of +z direction shear force
			sum_force_avg += force_avg[i];
			sum_area += area[i];
		} 
	}
	
	if(immersed) {
		GlobalSum_All(&lforce_ibm, &force_ibm, PETSC_COMM_WORLD);
		GlobalSum_All(&lforce_avg_ibm, &force_avg_ibm, PETSC_COMM_WORLD);
		GlobalSum_All(&larea_ibm, &area_ibm, PETSC_COMM_WORLD);
		
		
		//if( force_ibm>1.e-10) 
		{
			sum_force += force_ibm;
			sum_force_avg += force_avg_ibm;
			sum_area += area_ibm;
			
		}
	}
	

	//user->ustar_now = sqrt( sum_force / (sum_area+1.e-30) );
	//Re_tau = user->ren * user->ustar_now;
	
	user->ustar_avg = sqrt( fabs(force_avg[2]) / area[2] );
	user->Re_tau_avg = user->ren * user->ustar_avg;
	if(averaging && user->ustar_avg<1.e-10) {
	  user->Re_tau_avg = user->ren;
	  user->ustar_avg = 1.0;
	}
	
	PetscPrintf(PETSC_COMM_WORLD, "Bottom Re*_avg=%f\n", user->Re_tau_avg);
	/*printf("****************** %f %f\n", force_avg[2], area[2]);
	exit(0);
	*/
	
	// based on bottom
	/*
	ustar = sqrt( force[2] / (user->shear_area[2]+1.e-30) );
	Re_tau = user->ren * force[2] / user->shear_area[2];
	user->ustar_avg = sqrt( force_avg[2] / (user->shear_area[2]+1.e-30) );
	user->Re_tau_avg = user->ren * force_avg[2] / user->shear_area[2];
	*/
       
	
	if(inletprofile==17) {
		double sum_error_u=0, sum_error_v=0, sum_error_p=0;
		double total_sum_error_u=0, total_sum_error_v=0, total_sum_error_p=0;
		
		double inf_u=-100, inf_v=-100, inf_p=-100, ***p;
		double total_inf_u, total_inf_v, total_inf_p;
		
		double _time = user->dt*(ti+1);
		double a = 2.*M_PI;
			
		Vec Coor;
		Cmpnts	***coor, ***ucont;
		DMDAGetGhostedCoordinates(da, &Coor);
	
		DMDAVecGetArray(fda, Coor, &coor);
		DMDAVecGetArray(fda, user->lUcont, &ucont);
		DMDAVecGetArray(da, user->lP, &p);
		
		
		for (k=lzs; k<lze; k++) {
			if(k==1)
			for (j=lys; j<lye; j++)
			for (i=lxs; i<lxe; i++) {
				double xi = (coor[k  ][j  ][i].x + coor[k-1][j  ][i].x + coor[k  ][j-1][i].x + coor[k-1][j-1][i].x) * 0.25;	// centx[].x
				double yi = (coor[k  ][j  ][i].y + coor[k-1][j  ][i].y + coor[k  ][j-1][i].y + coor[k-1][j-1][i].y) * 0.25;	// centx[].y
				double xj = (coor[k  ][j][i  ].x + coor[k-1][j][i  ].x + coor[k  ][j][i-1].x + coor[k-1][j][i-1].x) * 0.25;
				double yj = (coor[k  ][j][i  ].y + coor[k-1][j][i  ].y + coor[k  ][j][i-1].y + coor[k-1][j][i-1].y) * 0.25;
				
				
				double u_exact_i = - cos (xi*a) * sin (yi*a) * exp ( -2./user->ren*a*a*_time );
				double v_exact_j = sin(xj*a) * cos (yj*a) * exp ( -2./user->ren*a*a*_time );
				double p_exact = - 0.25 * ( cos(2.*a*cent[k][j][i].x) + cos(2.*a*cent[k][j][i].y) ) * exp ( -4./user->ren*a*a*_time );
				
				if(i==10 && j==10) {
					printf("u_exact=%e u_computed=%e\n", u_exact_i, ucont[k][j][i].x/csi[k][j][i].x );
					printf("p_exact=%e p_computed=%e\n", p_exact, p[k][j][i] );
				}
				//p_exact -= min_p_exact ;
				
				sum_error_u += pow ( (ucont[k][j][i].x/csi[k][j][i].x - u_exact_i), 2. );	// x Area
				sum_error_v += pow ( (ucont[k][j][i].y/eta[k][j][i].y - v_exact_j), 2. );
				sum_error_p += pow ( (p[k][j][i] - p_exact), 2. );
				
				inf_u = std::max ( inf_u, fabs(ucont[k][j][i].x/csi[k][j][i].x - u_exact_i) ); 
				inf_v = std::max ( inf_v, fabs(ucont[k][j][i].y/eta[k][j][i].y - v_exact_j) ); 
				inf_p = std::max ( inf_p, fabs(p[k][j][i] - p_exact) ); 
			}
		}
		
		GlobalSum_All(&sum_error_u, &total_sum_error_u, PETSC_COMM_WORLD);
		GlobalSum_All(&sum_error_v, &total_sum_error_v, PETSC_COMM_WORLD);
		GlobalSum_All(&sum_error_p, &total_sum_error_p, PETSC_COMM_WORLD);
		
		GlobalMax_All(&inf_u, &total_inf_u, PETSC_COMM_WORLD);
		GlobalMax_All(&inf_v, &total_inf_v, PETSC_COMM_WORLD);
		GlobalMax_All(&inf_p, &total_inf_p, PETSC_COMM_WORLD);
		
		total_sum_error_u = sqrt(total_sum_error_u) / (mx-2.);
		total_sum_error_v = sqrt(total_sum_error_v) / (mx-2.);
		total_sum_error_p = sqrt(total_sum_error_p) / (mx-2.);
		
		PetscPrintf(PETSC_COMM_WORLD, "%dx%d, ti=%d, t=%g\n", mx-2, my-2, ti, _time );
		PetscPrintf(PETSC_COMM_WORLD, "u error (L2, inf) : %.8e %.8e\n", total_sum_error_u, total_inf_u);
		PetscPrintf(PETSC_COMM_WORLD, "v error (L2, inf) : %.8e %.8e\n", total_sum_error_v, total_inf_v);
		PetscPrintf(PETSC_COMM_WORLD, "p error (L2, inf) : %.8e %.8e\n", total_sum_error_p, total_inf_p);
		
		
		DMDAVecRestoreArray(fda, user->lUcont, &ucont);
		DMDAVecRestoreArray(fda, Coor, &coor);
		DMDAVecRestoreArray(da, user->lP, &p);
	}
	
	DMDAVecRestoreArray(da, user->lP, &p);
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	DMDAVecRestoreArray(fda, user->lUcat, &ucat);
	DMDAVecRestoreArray(fda, user->lCent, &cent);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);	//seokkoo 
	DMDAVecRestoreArray(da, user->lUstar, &ustar);
	
	DMDAVecRestoreArray(fda, user->lICsi, &icsi);
        DMDAVecRestoreArray(fda, user->lIEta, &ieta);
        DMDAVecRestoreArray(fda, user->lIZet, &izet);
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	
	DMDAVecRestoreArray(fda, user->lJCsi, &jcsi);
        DMDAVecRestoreArray(fda, user->lJEta, &jeta);
        DMDAVecRestoreArray(fda, user->lJZet, &jzet);
        DMDAVecRestoreArray(da, user->lJAj, &jaj);

	if(averaging && ti>1) {
       		DMDAVecRestoreArray(fda, lUcat_sum, &ucat_sum);
		DMDAVecRestoreArray(da, user->P_sum, &p_sum);
		VecDestroy(&lUcat_sum);
	}
	
	//PetscPrintf(PETSC_COMM_WORLD, "\tForce  : %e %e %e %e %e %e\n", force[0],force[1],force[2],force[3],force[4],force[5]);
	PetscPrintf(PETSC_COMM_WORLD, "\n\tShear Area (Left, Right, Bottom, Top, IB) : %.4e %.4e %.4e %.4e %.4e\n\n", area[0], area[1], area[2], area[3], area_ibm);
	PetscPrintf(PETSC_COMM_WORLD, "\tInstantaneous Total Force        : %.4e %.4e %.4e %.4e\n", force[0], force[1], force[2], force[3]);
	PetscPrintf(PETSC_COMM_WORLD, "\tInstantaneous Total Shear Stress : %.4e %.4e %.4e %.4e %.4e\n", user->shear_stress[0], user->shear_stress[1], user->shear_stress[2],user->shear_stress[3], force_ibm/area_ibm);
	PetscPrintf(PETSC_COMM_WORLD, "\tInstantaneous Frictional Stress  : %.4e %.4e %.4e %.4e\n", force_skin[0]/area[0], force_skin[1]/area[1], force_skin[2]/area[2],force_skin[3]/area[3]);
	PetscPrintf(PETSC_COMM_WORLD, "\tInstantaneous Pressure Stress    : %.4e %.4e %.4e %.4e\n\n", force_pressure[0]/area[0], force_pressure[1]/area[1], force_pressure[2]/area[2],force_pressure[3]/area[3]);
	
	PetscPrintf(PETSC_COMM_WORLD, "\tLeft   u*=%e, Re*=%.2f\n", sqrt(fabs(user->shear_stress[0])), sqrt(fabs(user->shear_stress[0]))*user->ren );
	PetscPrintf(PETSC_COMM_WORLD, "\tRight  u*=%e, Re*=%.2f\n", sqrt(fabs(user->shear_stress[1])), sqrt(fabs(user->shear_stress[1]))*user->ren );
	PetscPrintf(PETSC_COMM_WORLD, "\tBottom u*=%e, Re*=%.2f\n", sqrt(fabs(user->shear_stress[2])), sqrt(fabs(user->shear_stress[2]))*user->ren );
	PetscPrintf(PETSC_COMM_WORLD, "\tTop    u*=%e, Re*=%.2f\n", sqrt(fabs(user->shear_stress[3])), sqrt(fabs(user->shear_stress[3]))*user->ren );
	PetscPrintf(PETSC_COMM_WORLD, "\tIB     u*=%e, Re*=%.2f\n", sqrt(force_ibm/area_ibm), sqrt(force_ibm/area_ibm)*user->ren  );
	PetscPrintf(PETSC_COMM_WORLD, "\n\tTotal  u*=%e, Re*=%.2f\n", sqrt(sum_force/sum_area), sqrt(sum_force/sum_area)*user->ren  );
	
	for(i=0; i<6; i++) user->ustar_now[i] = sqrt( fabs(user->shear_stress[i]) );

	// add begin (xiaolei)
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


        if (!rank) {
                FILE *f;
                char filen[80];
                sprintf(filen, "u_star");
                f = fopen(filen, "a");
                PetscFPrintf(PETSC_COMM_WORLD, f, "%d %le %le %le %le %le \n",ti, sqrt(fabs(user->shear_stress[0])), sqrt(fabs(user->shear_stress[1])), sqrt(fabs(user->shear_stress[2])), sqrt(fabs(user->shear_stress[3])), sqrt(force_ibm/(area_ibm+1.0e-12)) );
                fclose(f);
        }

	// add end (xiaolei)

	if(averaging && ti>1) {
	  //PetscPrintf(PETSC_COMM_WORLD, "\tAveraged  u*=%e, Re*=%.2f\n", user->ustar_avg, user->Re_tau_avg );
	  //PetscPrintf(PETSC_COMM_WORLD, "\tMean Shear Force : Bottom %e, Top %e, IB %e\n", user->shear_force_avg[2], user->shear_force_avg[3], force_avg_ibm );
	}
	
	
};

/*
//savememory version
void write_shear_stress_ibm()
{
	for(int ibi=0; ibi<NumberOfBodies; ibi++) {
		
		IBMNodes *ibm = &ibm_ptr[ibi];
		
		std::vector<double> shear_tmp(ibm->total_n_elmt);
		std::vector<double> mean_shear_tmp(ibm->total_n_elmt);
		std::vector<double> reynolds1_tmp(ibm->total_n_elmt);
		std::vector<double> reynolds2_tmp(ibm->total_n_elmt);
		std::vector<double> reynolds3_tmp(ibm->total_n_elmt);
		
		std::vector<double> shear(ibm->total_n_elmt);
		std::vector<double> mean_shear(ibm->total_n_elmt);
		std::vector<double> reynolds1(ibm->total_n_elmt);
		std::vector<double> reynolds2(ibm->total_n_elmt);
		std::vector<double> reynolds3(ibm->total_n_elmt);			
		
		std::fill ( shear_tmp.begin(), shear_tmp.end(), 0. );
		std::fill ( mean_shear_tmp.begin(), mean_shear_tmp.end(), 0. );
		std::fill ( reynolds1_tmp.begin(), reynolds1_tmp.end(), 0. );
		std::fill ( reynolds2_tmp.begin(), reynolds2_tmp.end(), 0. );
		std::fill ( reynolds3_tmp.begin(), reynolds3_tmp.end(), 0. );
		
		std::vector<int> count(ibm->total_n_elmt), count_tmp(ibm->total_n_elmt);
		std::fill(count_tmp.begin(), count_tmp.end(), 0);
		
		for (int i=0; i<ibm->n_elmt; i++) {
			shear_tmp[ ibm->local2global_elmt [i] ] = ibm->shear[i];
			mean_shear_tmp[ ibm->local2global_elmt [i] ] = ibm->mean_shear[i];
			reynolds1_tmp[ ibm->local2global_elmt [i] ] = ibm->reynolds_stress1[i];
			reynolds2_tmp[ ibm->local2global_elmt [i] ] = ibm->reynolds_stress2[i];
			reynolds3_tmp[ ibm->local2global_elmt [i] ] = ibm->reynolds_stress3[i];
			count_tmp[ ibm->local2global_elmt [i] ] = 1;
		}
	       
		MPI_Reduce ( &count_tmp[0], &count[0], ibm->total_n_elmt, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &shear_tmp[0], &shear[0], ibm->total_n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &mean_shear_tmp[0], &mean_shear[0], ibm->total_n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		
		MPI_Reduce ( &reynolds1_tmp[0], &reynolds1[0], ibm->total_n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &reynolds2_tmp[0], &reynolds2[0], ibm->total_n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &reynolds3_tmp[0], &reynolds3[0], ibm->total_n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		
		
		if(!my_rank) {
			int a=0;
			FILE *fp;
			char filen[256];
				
			sprintf(filen, "%s/shear%06d_%2.2d.dat",path,ti,ibi);
			fp = fopen(filen, "w");
			fprintf(fp, "Variables=x, y, z,shear_stress, mean_shear_stress, UV_ VW_ WU_\n");
			fprintf(fp, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-8]=CELLCENTERED)\n", ibm->total_n_v, ibm->total_n_elmt);
			for (int i=0; i<ibm->total_n_elmt; i++) {
			  if(count[i]) {
			    shear[i]/=(double)count[i];
			    mean_shear[i]/=(double)count[i];
			    reynolds1[i]/=(double)count[i];
			    reynolds2[i]/=(double)count[i];
			    reynolds3[i]/=(double)count[i];
			  }
			}
			for (int i=0; i<ibm->total_n_v; i++) fprintf(fp, "%e\n", ibm->_x_bp[i]);
			for (int i=0; i<ibm->total_n_v; i++) fprintf(fp, "%e\n", ibm->_y_bp[i]);
			for (int i=0; i<ibm->total_n_v; i++) fprintf(fp, "%e\n", ibm->_z_bp[i]);
			
			for (int i=0; i<ibm->total_n_elmt; i++) fprintf(fp, "%.6e\n", shear[i]);
			for (int i=0; i<ibm->total_n_elmt; i++) fprintf(fp, "%.6e\n", mean_shear[i]);
			for (int i=0; i<ibm->total_n_elmt; i++) fprintf(fp, "%.6e\n", reynolds1[i]);
			for (int i=0; i<ibm->total_n_elmt; i++) fprintf(fp, "%.6e\n", reynolds2[i]);
			for (int i=0; i<ibm->total_n_elmt; i++) fprintf(fp, "%.6e\n", reynolds3[i]);
			
			for (int i=0; i<ibm->total_n_elmt; i++) fprintf(fp, "%d %d %d\n", ibm->_nv1[i]+1, ibm->_nv2[i]+1, ibm->_nv3[i]+1);
			
			fclose(fp);
		}
		
	}
       
	MPI_Barrier ( PETSC_COMM_WORLD );
}
*/

void write_shear_stress_ibm()
{
	for(int ibi=0; ibi<NumberOfBodies; ibi++) {
		
		IBMNodes *ibm = &ibm_ptr[ibi];
		
		std::vector<double> shear_tmp(ibm->n_elmt), shear(ibm->n_elmt);
		std::vector<double> mean_shear_tmp(ibm->n_elmt), mean_shear(ibm->n_elmt);
		std::vector<double> reynolds1_tmp(ibm->n_elmt), reynolds1(ibm->n_elmt);
		std::vector<double> reynolds2_tmp(ibm->n_elmt), reynolds2(ibm->n_elmt);
		std::vector<double> reynolds3_tmp(ibm->n_elmt), reynolds3(ibm->n_elmt);
		std::vector<double> pressure_tmp(ibm->n_elmt), pressure(ibm->n_elmt);
		
		std::vector<double> rel_velocity_x_tmp(ibm->n_elmt), rel_velocity_x(ibm->n_elmt);
		std::vector<double> rel_velocity_y_tmp(ibm->n_elmt), rel_velocity_y(ibm->n_elmt);
		std::vector<double> rel_velocity_z_tmp(ibm->n_elmt), rel_velocity_z(ibm->n_elmt);
		
		std::vector<int> count(ibm->n_elmt), count_tmp(ibm->n_elmt);

		std::fill ( shear_tmp.begin(), shear_tmp.end(), 0. );
		std::fill ( mean_shear_tmp.begin(), mean_shear_tmp.end(), 0. );
		std::fill ( reynolds1_tmp.begin(), reynolds1_tmp.end(), 0. );
		std::fill ( reynolds2_tmp.begin(), reynolds2_tmp.end(), 0. );
		std::fill ( reynolds3_tmp.begin(), reynolds3_tmp.end(), 0. );
		std::fill ( pressure_tmp.begin(), pressure_tmp.end(), 0. );
		
		std::fill ( rel_velocity_x_tmp.begin(), rel_velocity_x_tmp.end(), 0. );
		std::fill ( rel_velocity_y_tmp.begin(), rel_velocity_y_tmp.end(), 0. );
		std::fill ( rel_velocity_z_tmp.begin(), rel_velocity_z_tmp.end(), 0. );
		
		std::fill ( count_tmp.begin(), count_tmp.end(), 0);

		for (int i=0; i<ibm->n_elmt; i++) {
			if( fabs(ibm->shear[i])>1.e-10 ) {
				shear_tmp[i] = ibm->shear[i];
				mean_shear_tmp[i] = ibm->mean_shear[i];
				reynolds1_tmp[i] = ibm->reynolds_stress1[i];
				reynolds2_tmp[i] = ibm->reynolds_stress2[i];
				reynolds3_tmp[i] = ibm->reynolds_stress3[i];
				pressure_tmp[i] = ibm->pressure[i];
				
				rel_velocity_x_tmp[i] = ibm->rel_velocity[i].x;
				rel_velocity_y_tmp[i] = ibm->rel_velocity[i].y;
				rel_velocity_z_tmp[i] = ibm->rel_velocity[i].z;
				
				count_tmp[i] += 1;
			}
		}
	       
		MPI_Reduce ( &shear_tmp[0], &shear[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &mean_shear_tmp[0], &mean_shear[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &reynolds1_tmp[0], &reynolds1[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &reynolds2_tmp[0], &reynolds2[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &reynolds3_tmp[0], &reynolds3[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &pressure_tmp[0], &pressure[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		
		MPI_Reduce ( &rel_velocity_x_tmp[0], &rel_velocity_x[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &rel_velocity_y_tmp[0], &rel_velocity_y[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &rel_velocity_z_tmp[0], &rel_velocity_z[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		
		MPI_Reduce ( &count_tmp[0], &count[0], ibm->n_elmt, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
		
		if(!my_rank) {
			int a=0;
			FILE *fp;
			char filen[256];
				
			sprintf(filen, "%s/shear%06d_%d.dat",path,(int)ti,ibi);
			fp = fopen(filen, "w");
			fprintf(fp, "Variables=x y z shear_stress mean_shear_stress UV VW WU p u_ v_ w_\n");
			fprintf(fp, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)\n", ibm->n_v, ibm->n_elmt);
			
			for (int i=0; i<ibm->n_elmt; i++) {
				if(count[i]) {
					shear[i]/=(double)count[i];
					mean_shear[i]/=(double)count[i];
					reynolds1[i]/=(double)count[i];
					reynolds2[i]/=(double)count[i];
					reynolds3[i]/=(double)count[i];
					pressure[i]/=(double)count[i];
					rel_velocity_x[i]/=(double)count[i];
					rel_velocity_y[i]/=(double)count[i];	
					rel_velocity_z[i]/=(double)count[i];
				}
			}
			
			for (int i=0; i<ibm->n_v; i++) fprintf(fp, "%e\n", ibm->x_bp[i]);
			for (int i=0; i<ibm->n_v; i++) fprintf(fp, "%e\n", ibm->y_bp[i]);
			for (int i=0; i<ibm->n_v; i++) fprintf(fp, "%e\n", ibm->z_bp[i]);
			
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", shear[i]);
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", mean_shear[i]);
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", reynolds1[i]);
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", reynolds2[i]);
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", reynolds3[i]);
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", pressure[i]);
			
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", rel_velocity_x[i]);
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", rel_velocity_y[i]);
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%.6e\n", rel_velocity_z[i]);
			
			for (int i=0; i<ibm->n_elmt; i++) fprintf(fp, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
			
			fclose(fp);
			
			/***********
			double Mx=0, My=0, Mz=0;
			for (int i=0; i<ibm->n_elmt; i++) {
				double x=ibm->cent_x[i], y=ibm->cent_y[i], z=ibm->cent_z[i]; 
				double r_x = x-x_r;
				double r_y = y-y_r;
				double r_z = z-z_r;
			      
				double F_x=(-pressure[i]*ibm->nf_x[i])*ibm->dA[i]; 
				double F_y=(-pressure[i]*ibm->nf_y[i])*ibm->dA[i]; 
				double F_z=(-pressure[i]*ibm->nf_z[i])*ibm->dA[i]; 
			      
				Mx +=   r_y*F_z - r_z*F_y;
				My += -(r_x*F_z - r_z*F_x);
				Mz +=   r_x*F_y - r_y*F_x;
			}
			
			sprintf(filen, "%s/torque_%02d.dat",path,ibi);
			fp = fopen(filen, "a");
			fprintf(fp, "%e %e %e\n", Mx, My, Mz);
			fclose(fp);
			********/
		}
		
	}
       
	MPI_Barrier ( PETSC_COMM_WORLD );
}

void write_torque_ibm()
{
	for(int ibi=0; ibi<NumberOfBodies; ibi++) {
		
		IBMNodes *ibm = &ibm_ptr[ibi];
		
		std::vector<double> pressure_tmp(ibm->n_elmt), pressure(ibm->n_elmt);
		std::vector<int> count(ibm->n_elmt), count_tmp(ibm->n_elmt);

		std::fill ( pressure_tmp.begin(), pressure_tmp.end(), 0. );
		std::fill ( count_tmp.begin(), count_tmp.end(), 0);

		for (int i=0; i<ibm->n_elmt; i++) {
			if( fabs(ibm->shear[i])>1.e-10 ) {
				pressure_tmp[i] = ibm->pressure[i];
				count_tmp[i] += 1;
			}
		}

		MPI_Reduce ( &pressure_tmp[0], &pressure[0], ibm->n_elmt, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce ( &count_tmp[0], &count[0], ibm->n_elmt, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
		
		if(!my_rank) {
			FILE *fp;
			char filen[256];
				
			double Mx=0, My=0, Mz=0;
			for (int i=0; i<ibm->n_elmt; i++) {
			  double P=0;
			  if(count[i]>0) P = pressure[i]/(double)count[i];

				double x=ibm->cent_x[i], y=ibm->cent_y[i], z=ibm->cent_z[i]; 
				double r_x = x-x_r;
				double r_y = y-y_r;
				double r_z = z-z_r;
			      
				double F_x=(-P*ibm->nf_x[i])*ibm->dA[i]; 
				double F_y=(-P*ibm->nf_y[i])*ibm->dA[i]; 
				double F_z=(-P*ibm->nf_z[i])*ibm->dA[i]; 
			      
				Mx +=   r_y*F_z - r_z*F_y;
				My += -(r_x*F_z - r_z*F_x);
				Mz +=   r_x*F_y - r_y*F_x;
			}
			
			sprintf(filen, "%s/torque_%02d.dat",path,ibi);
			fp = fopen(filen, "a");
			fprintf(fp, "%d %e %e %e\n", (int)ti, Mx, My, Mz);
			fclose(fp);
		}
		
	}
       
	MPI_Barrier ( PETSC_COMM_WORLD );
}

void Calc_k_Flux(UserCtx *user)
{
	
	
	int i, j, k;
	DM da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int	lxs, lxe, lys, lye, lzs, lze;
 
	PetscReal	***nvert, ***level, ***aj;
	Cmpnts ***ucont, ***ucat, ***zet, ***kzet, ***eta, ***csi;
	
	
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;
  
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;
  
	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lAj, &aj);
	if(levelset) DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(fda, user->lUcont, &ucont);
	DMDAVecGetArray(fda, user->lUcont, &ucat);
	DMDAVecGetArray(fda, user->lZet, &zet);
	DMDAVecGetArray(fda, user->lKZet, &kzet);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lCsi, &csi);
	
	double lFlux=0, lFlux_ibm=0;
	double Flux, Flux_ibm;
	
	for(k=lzs; k<lze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		if(j>=1 && j<=my-2 && i>=1 && i<=mx-2) {
			if( levelset ) {
				/*
				double dx = 1./aj[k][j][i]/sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
				if(dthick_set) dx*=dthick;
				*/
				double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
				double vf = H(level[k+1][j][i], dx);
				if (nvert[k+1][j][i]+nvert[k][j][i] < 0.1) lFlux += ucont[k][j][i].z * vf;
				else if (nvert[k+1][j][i]+nvert[k][j][i] < 2.1) {
					double ucx = (ucat[k+1][j][i].x + ucat[k][j][i].x) * 0.5;
					double ucy = (ucat[k+1][j][i].y + ucat[k][j][i].y) * 0.5;
					double ucz = (ucat[k+1][j][i].z + ucat[k][j][i].z) * 0.5;
					lFlux_ibm += (ucx * zet[k][j][i].x + ucy * zet[k][j][i].y + ucz * zet[k][j][i].z) * vf;
				}
			}	
			else {
				if (nvert[k+1][j][i]+nvert[k][j][i] < 0.1) lFlux += ucont[k][j][i].z;
				else if (nvert[k+1][j][i]+nvert[k][j][i] < 2.1) {
					//lFlux_ibm += ucont[k][j][i].z;
					double ucx = (ucat[k+1][j][i].x + ucat[k][j][i].x) * 0.5;
					double ucy = (ucat[k+1][j][i].y + ucat[k][j][i].y) * 0.5;
					double ucz = (ucat[k+1][j][i].z + ucat[k][j][i].z) * 0.5;
					lFlux_ibm += (ucx * zet[k][j][i].x + ucy * zet[k][j][i].y + ucz * zet[k][j][i].z);
				}
			}
		}
	}

	GlobalSum_All(&lFlux, &Flux, PETSC_COMM_WORLD);
	GlobalSum_All(&lFlux_ibm, &Flux_ibm, PETSC_COMM_WORLD);
	
	Flux /= (mz-2);
	Flux_ibm /= (mz-2);
	
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	if(levelset) DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(fda, user->lUcont, &ucont);
	DMDAVecRestoreArray(fda, user->lUcont, &ucat);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	DMDAVecRestoreArray(fda, user->lKZet, &kzet);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	
	PetscPrintf(PETSC_COMM_WORLD, "\n*** Mean k Flux:%f, Flux ibnode:%f, k=1 area: %f, k=1 area ibnode: %f\n\n", Flux, Flux_ibm, user->k_area[1], user->k_area_ibnode[1]);
	
	user->mean_k_flux = Flux;
	user->mean_k_flux_ibnode = Flux_ibm;
};

void Compute_Q(UserCtx *user, Vec Q)
{
	DMDALocalInfo	info;
	Vec Csi = user->lCsi, Eta = user->lEta, Zet = user->lZet;
	Cmpnts	***ucat;
	Cmpnts	***csi, ***eta, ***zet;
	PetscReal	***nvert, ***aj, ***q;
	DM da = user->da, fda = user->fda;
	
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
  	int	lxs, lxe, lys, lye, lzs, lze;

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

	VecSet(Q, 0);

	DMDAVecGetArray(fda, user->lUcat,  &ucat);
	DMDAVecGetArray(fda, Csi, &csi);
	DMDAVecGetArray(fda, Eta, &eta);
	DMDAVecGetArray(fda, Zet, &zet);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lAj, &aj); 
	DMDAVecGetArray(da, Q, &q);
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if( nvert[k][j][i]>1.1) {
			continue;
		}
		
		double ajc = aj[k][j][i];
		double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
		double eta0 = eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
		double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
		double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
		double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
		
		Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
		Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
							&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
		
		double w11 = 0;
		double w12 = 0.5*(du_dy - dv_dx);
		double w13 = 0.5*(du_dz - dw_dx);
		double w21 = -w12;
		double w22 = 0.;
		double w23 = 0.5*(dv_dz - dw_dy);
		double w31 = -w13;
		double w32 = -w23;
		double w33 = 0.;
	
		q[k][j][i] = w11*w11 + w12*w12 + w13*w13 + w21*w21 + w22*w22 + w23*w23 + w31*w31 + w32*w32 + w33*w33;
	}
	
	DMDAVecRestoreArray(fda, user->lUcat,  &ucat);
	DMDAVecRestoreArray(fda, Csi, &csi);
	DMDAVecRestoreArray(fda, Eta, &eta);
	DMDAVecRestoreArray(fda, Zet, &zet);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lAj, &aj); 
	DMDAVecGetArray(da, Q, &q);
	
	MPI_Barrier ( PETSC_COMM_WORLD );
}
